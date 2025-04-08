import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,  
    StoppingCriteriaList 
)
from peft import PeftModel, PeftConfig



class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        # CPU 텐서로 저장하여 device 불일치 방지
        self.stop_token_ids = [torch.tensor(ids, dtype=torch.long).to('cpu') for ids in stop_token_ids]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # 생성된 각 시퀀스(배치 내)에 대해 검사
        for stop_ids in self.stop_token_ids:
            # 생성된 토큰의 끝 부분이 중단 토큰과 일치하는지 확인
            # CPU에서 비교 수행
            current_input_ids = input_ids[0].to('cpu') # 현재 시퀀스를 CPU로 이동
            if len(current_input_ids) >= len(stop_ids):
                 # 끝에서부터 stop_ids 길이만큼 잘라서 비교
                if torch.equal(current_input_ids[-len(stop_ids):], stop_ids):
                    return True # 중단 조건 충족
        return False # 중단 조건 미충족


class MengziChatbot:
    def __init__(self, base_dir="/kaggle/working/mengzi_lora_finetuned", checkpoint="checkpoint-250"):
        if checkpoint:
            self.model_path = os.path.join(base_dir, checkpoint)
        else:
            self.model_path = base_dir

        print(f"모델 경로: {self.model_path}")

        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        try:
            config = PeftConfig.from_pretrained(self.model_path)
            base_model_name = config.base_model_name_or_path

            print(f"기본 모델 로딩 시작: {base_model_name} (8-bit 양자화, device_map='auto')")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=quantization_config,
                device_map='auto',
            )
            print("기본 모델 로딩 완료.")

            print("LoRA 어댑터 적용 시작...")
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            self.model.eval()
            print("LoRA 어댑터 적용 완료.")

            print("토크나이저 로딩 시작...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)
            
            if self.tokenizer.pad_token is None:
                print("패딩 토큰이 없어 EOS 토큰으로 설정합니다.")
                self.tokenizer.pad_token = self.tokenizer.eos_token
                # 모델 설정에도 반영 (generate에서 pad_token_id를 자동으로 설정하지만 명시)
                self.model.config.pad_token_id = self.tokenizer.eos_token_id
          
            print("토크나이저 로딩 완료.")

            self.device = next(self.model.parameters()).device
            print(f"모델 로드 완료. 추론에 사용할 장치: {self.device}")

           
            # "\n질문:" 문자열을 토큰 ID 리스트로 변환
            stop_token_strings = ["\n질문:", "\n사용자 질문:", "\nUser:"] # 멈추고 싶은 문자열 추가 가능
            self.stop_token_ids = [self.tokenizer(stop_str, return_tensors='pt', add_special_tokens=False).input_ids.squeeze().tolist() for stop_str in stop_token_strings]
            # StoppingCriteriaList 생성
            self.stopping_criteria = StoppingCriteriaList([StopOnTokens(self.stop_token_ids)])
            print(f"생성 중단 트리거 설정 완료: {stop_token_strings}")
           

        except Exception as e:
            print(f"모델 로딩 중 오류 발생: {e}")
            if "CUDA out of memory" in str(e):
                 print("GPU 메모리가 부족합니다. 더 작은 모델을 사용하거나, 4-bit 양자화를 시도하거나, 더 많은 GPU 메모리가 있는 환경에서 실행하세요.")
            else:
                print("지정된 경로, 설정 파일, 라이브러리(bitsandbytes, accelerate) 설치 여부를 확인하세요.")
            raise e

        self.conversation_history = [
            "당신은 맹자입니다. 인(仁)과 의(義)를 가장 중요하게 여기는 철학자로서 대화합니다.",
            "당신의 목표는 백성들을 올바른 길로 인도하고, 통치자들에게 올바른 통치의 길을 제시하는 것입니다.",
            "질문에 대해 철학적이고 명확하게 답변하십시오."
        ]
        print(f"'{base_model_name}' 기반 모델(8-bit 양자화)과 '{self.model_path}'의 LoRA 어댑터를 성공적으로 로드했습니다.")

    def generate_response(self, user_input, max_length=200):
        context = "\n".join(self.conversation_history)
        # 프롬프트 마지막에 "맹자의 답변:"을 붙여 모델이 답변을 시작하도록 유도
        full_prompt = f"{context}\n질문: {user_input}\n맹자의 답변:"

        inputs = self.tokenizer(full_prompt, return_tensors='pt', max_length=512, truncation=True).to(self.device)
        prompt_length = inputs['input_ids'].shape[1] # 프롬프트의 원래 길이 저장

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=max_length,  # max_length 대신 max_new_tokens 사용 권장
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id, # 명시적으로 추가 (경고 방지)
                stopping_criteria=self.stopping_criteria # 생성 중단 기준 전달
            )
        

        # 프롬프트를 제외한 생성된 부분만 디코딩
        response_ids = outputs[0][prompt_length:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        response = response.strip()

        # 대화 기록 업데이트
        self.conversation_history.append(f"질문: {user_input}")
        self.conversation_history.append(f"맹자의 답변: {response}") # 실제 생성된 답변을 저장

        # 대화 기록 길이 제한 (가장 오래된 질문+답변 쌍 제거)
        if len(self.conversation_history) > 7: # 페르소나 3줄 + Q&A 2쌍 = 7줄. 초과 시 제거
            self.conversation_history = self.conversation_history[:3] + self.conversation_history[-4:]

        return response

    def chat(self):
        print("\n--- 맹자 챗봇과의 대화 시작 ---")
        print("대화를 종료하려면 '종료'를 입력하세요.")
        while True:
            try:
                user_input = input("당신: ")
            except EOFError:
                print("\n입력 스트림이 종료되었습니다.")
                break
            if user_input.lower() == '종료':
                print("맹자: 평안하시길 바랍니다.")
                break
            if not user_input.strip():
                continue
            try:
                response = self.generate_response(user_input)
                # 최종 출력에서 "맹자의 답변:" 접두어는 제거하고 내용만 보여주기 
                # print("맹자:", response.replace("맹자의 답변:", "").strip())
                print("맹자:", response) # 위에서 이미 "맹자의 답변:"을 붙여서 저장했으므로 그대로 출력
            except Exception as e:
                print(f"응답 생성 중 오류 발생: {e}")
                break

if __name__ == '__main__':
    torch.cuda.empty_cache()
    chatbot = MengziChatbot()
    chatbot.chat()