import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig, # 양자화를 위해 추가
    # EarlyStoppingCallback 추가
    EarlyStoppingCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training, # k-bit 학습 준비를 위해 추가
    TaskType
)
import bitsandbytes as bnb # target_modules 검색 시 타입 확인용
from data_processing import MengziDataProcessor 


import warnings
warnings.filterwarnings("ignore")
import gc # 가비지 컬렉션

# DVC 관련 환경 변수
os.environ["DVC_NO_ANALYTICS"] = "true"
os.environ["DVC_NO_TRY_EXCEPT"] = "true"

# --- Kaggle 경로 설정 ---
KAGGLE_INPUT_DATA_PATH = '/kaggle/input/mencious-dataset/mencius_dataset.json'
KAGGLE_OUTPUT_DIR = '/kaggle/working/mengzi_lora_finetuned' # 쓰기 가능한 경로


class MengziModelTrainer:
    def __init__(
        self,
        data_path=KAGGLE_INPUT_DATA_PATH,
        model_name='beomi/KoAlpaca-Polyglot-5.8B',
        output_dir=KAGGLE_OUTPUT_DIR,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        use_quantization=True,
        use_gradient_checkpointing=True,
        # Early Stopping 관련 파라미터 추가
        early_stopping_patience=2,
        early_stopping_threshold=0.0, # 기본값 0.0 (개선이 없으면 중단)
    ):
        self.data_path = data_path
        self.model_name = model_name
        self.output_dir = output_dir
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.use_quantization = use_quantization
        self.use_gradient_checkpointing = use_gradient_checkpointing
        # Early Stopping 파라미터 저장
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print(f"Using data path: {self.data_path}")
        print(f"Using output directory: {self.output_dir}")
        os.makedirs(self.output_dir, exist_ok=True)

        # --- 양자화 설정 ---
        bnb_config = None
        compute_dtype = torch.float32 # 기본값
        if self.use_quantization:
            if self.device == 'cpu':
                print("Warning: Quantization requires CUDA, but running on CPU. Disabling quantization.")
                self.use_quantization = False
            else:
                print("Applying 4-bit quantization...")
                compute_dtype = torch.bfloat16 # 기본적으로 bfloat16 시도
                if not torch.cuda.is_bf16_supported():
                    print("BF16 not supported on this GPU, falling back to FP16 for compute dtype.")
                    compute_dtype = torch.float16

                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=compute_dtype
                )

        # --- 모델 로드 ---
        print(f"Loading model: {self.model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config, # None이면 적용 안됨
            device_map="auto", # GPU 메모리에 맞게 자동 분배
            torch_dtype=compute_dtype if not bnb_config else None, # 양자화 시 None 권장, 아닐 경우 compute_dtype 사용
            trust_remote_code=True
        )

        # --- 토크나이저 로드 및 설정 ---
        print(f"Loading tokenizer: {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.model.config.pad_token_id = self.tokenizer.eos_token_id # 모델 설정에도 반영
            print(f"Tokenizer pad_token_id set to eos_token_id: {self.tokenizer.eos_token_id}")

        # --- 양자화 모델 학습 준비 ---
        if self.use_quantization:
            print("Preparing model for k-bit training...")
            # gradient_checkpointing은 TrainingArguments에서 제어하므로 여기서 False 설정
            self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=False)

        # --- LoRA 설정 (target_modules 동적 검색) ---
        print("Finding LoRA target modules...")
        lora_module_names = set()
        for name, module in self.model.named_modules():
            # Linear 레이어 또는 양자화된 Linear 레이어 찾기
            if isinstance(module, bnb.nn.Linear4bit) or isinstance(module, torch.nn.Linear):
                names = name.split('.')
                # 마지막 모듈 이름 (예: 'query_key_value', 'dense')을 타겟으로 추가
                lora_module_names.add(names[-1])

        # 일반적으로 사용되는 모듈 이름 우선순위 (Polyglot/GPT-NeoX 기반)
        potential_targets = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
        target_modules = [name for name in potential_targets if name in lora_module_names]

        if not target_modules:
            # 예비로 찾은 모든 Linear 모듈 이름 사용 (주의 필요)
            # target_modules = list(lora_module_names)
            # 또는 기본값 사용 및 경고
            target_modules = ["query_key_value", "dense"] # 기본값 시도
            print(f"Warning: Could not find typical LoRA targets ({potential_targets}). "
                  f"Using default: {target_modules}. Check model architecture if training fails.")
        else:
            print(f"Found LoRA target modules: {target_modules}")

        self.lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=target_modules, # 동적으로 찾은 모듈 사용
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        # --- PEFT 모델 생성 ---
        try:
            self.model = get_peft_model(self.model, self.lora_config)
            print("PEFT model created successfully.")
            self.model.print_trainable_parameters() # 학습 가능한 파라미터 수 확인!
        except Exception as e:
            print(f"Error creating PEFT model: {e}")
            print("Please check if the target_modules are correctly identified and exist in the base model.")
            print(f"Identified Linear/Linear4bit module names: {lora_module_names}")
            raise e

        # --- 데이터 프로세서 초기화 ---
        try:
            # <<-- 1. 실제 MengziDataProcessor를 초기화 시도 -->>
            self.data_processor = MengziDataProcessor(
                data_path=self.data_path,
                tokenizer=self.tokenizer, # Trainer의 토크나이저 전달
                # 만약 MengziDataProcessor의 __init__에 다른 필수 인자가 있다면 여기에 추가
                # 예: model_name=self.model_name (선택적 인자라면 필요 없을 수 있음)
                # 예: text_column="text" (만약 사용한다면)
            )
            print("Successfully initialized MengziDataProcessor.")
        
        except NameError:
            print("\n!!! ERROR: MengziDataProcessor class is not defined !!!")
            print("Please ensure the MengziDataProcessor class definition code")
            print("is executed in a cell *before* this code block.\n")
            raise
        
        except TypeError as e:
            print(f"\n!!! ERROR: MengziDataProcessor __init__ might have missing arguments or other issues: {e} !!!")
            print("Please check the MengziDataProcessor class definition and ensure its __init__")
            print("method correctly accepts the required arguments (like 'tokenizer').\n")
            raise
        
        except Exception as e:
            # 예상치 못한 다른 에러 처리
            print(f"\n!!! An unexpected error occurred during MengziDataProcessor initialization: {e} !!!")
            import traceback
            traceback.print_exc()
            raise




    def prepare_training(self, test_size=0.1):
        """데이터셋 로드, 처리 및 토큰화 수행"""
        print("Preparing dataset...")
        # MengziDataProcessor가 데이터를 로드하고 train/test로 분할해야 함
        dataset = self.data_processor.create_dataset(test_size=test_size)
        if dataset is None:
             print("Dataset creation failed (check MengziDataProcessor implementation). Exiting.")
             return None
        # Early stopping을 사용하려면 'test' 데이터셋이 반드시 필요합니다.
        if 'test' not in dataset:
            print("Warning: Early stopping requires a 'test' split in the dataset, but it was not found.")
            print("Disabling early stopping features (load_best_model_at_end).")
            # 또는 에러 발생: raise ValueError("Early stopping requires a 'test' split.")

        # DataProcessor가 내부적으로 전달받은 self.tokenizer를 사용해야 함
        tokenized_dataset = self.data_processor.tokenize_dataset(dataset)
        if tokenized_dataset is None:
            print("Dataset tokenization failed (check MengziDataProcessor implementation). Exiting.")
            return None

        print("Dataset preparation finished.")
        return tokenized_dataset

    def train(self, tokenized_dataset):
        """모델 학습 수행"""
        if tokenized_dataset is None or 'train' not in tokenized_dataset:
             print("Cannot train without a valid tokenized 'train' dataset.")
             return

        # --- 데이터 콜레이터 설정 ---
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, # Trainer의 토크나이저 사용
            mlm=False,
        )

        # --- TrainingArguments 설정 ---
        print("Setting up Training Arguments...")
        # Kaggle GPU (T4/P100 - 16GB) 고려 설정
        batch_size = 1
        gradient_accumulation_steps = 8 # 실질 배치 크기 = batch_size * grad_accum (여기선 8)
        effective_batch_size = batch_size * gradient_accumulation_steps
        print(f"Using per_device_train_batch_size={batch_size}, gradient_accumulation_steps={gradient_accumulation_steps} (Effective batch size: {effective_batch_size})")

        # FP16/BF16 설정 (init에서 결정된 compute_dtype 기반)
        use_fp16 = (self.device == 'cuda' and self.model.dtype == torch.float16)
        use_bf16 = (self.device == 'cuda' and self.model.dtype == torch.bfloat16)

        if use_bf16: print("Using BF16 training.")
        elif use_fp16: print("Using FP16 training.")

        # 검증 데이터셋 유무 확인
        has_eval_dataset = 'test' in tokenized_dataset
        if not has_eval_dataset:
            print("Evaluation dataset ('test') not found. Early stopping and loading best model will be disabled.")

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=3, # 최대 에폭 수 (Early stopping으로 조기 종료 가능)
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2, # 평가 시 조금 더 크게 가능
            gradient_accumulation_steps=gradient_accumulation_steps,
            optim="paged_adamw_8bit" if self.use_quantization else "adamw_torch",
            learning_rate=2e-4,
            weight_decay=0.01,
            fp16=use_fp16,
            bf16=use_bf16,
            logging_dir=os.path.join(self.output_dir, 'logs'),
            logging_strategy="steps",
            logging_steps=10, # 로그 출력 빈도
            remove_unused_columns=False,
            # --- Early Stopping 관련 설정 ---
            evaluation_strategy="steps" if has_eval_dataset else "no", # 검증 데이터가 있어야 평가 가능
            eval_steps=50 if has_eval_dataset else None, # 평가 빈도 (데이터셋 크기에 따라 조절) 
            save_strategy="steps",
            save_steps=50, # 체크포인트 저장 빈도 (eval_steps와 맞추는 것이 일반적)
            save_total_limit=2, # 최대 체크포인트 저장 개수
            load_best_model_at_end=has_eval_dataset, # 검증셋 있을 때만 종료 시 최적 모델 로드 
            metric_for_best_model="eval_loss" if has_eval_dataset else None, # 최적 모델 판단 기준 
            greater_is_better=False, # eval_loss는 낮을수록 좋음 
            # ------------------------------
            gradient_checkpointing=False, # CPU에서는 비활성화
            # group_by_length=True, # 패딩 효율 높임 (데이터셋 형태에 따라 효과 다름)
            report_to="none", # 외부 로깅 서비스 비활성화
            # dataloader_num_workers=2, # Kaggle 환경에 따라 조절
            # dataloader_pin_memory=True, # GPU 사용 시 고려
        )

        # --- Early Stopping Callback 설정 ---
        # TrainingArguments의 load_best_model_at_end=True와 함께 사용
        # EarlyStoppingCallback은 patience와 threshold 조건에 따라 학습을 '중단'시키는 역할
        # load_best_model_at_end=True는 중단되거나 정상 종료되었을 때, 저장된 체크포인트 중
        # metric_for_best_model 기준으로 가장 좋았던 모델을 로드하는 역할
        early_stopping_callback = None
        if has_eval_dataset:
             print(f"Early stopping enabled with patience={self.early_stopping_patience} and threshold={self.early_stopping_threshold}")
             early_stopping_callback = EarlyStoppingCallback(
                 early_stopping_patience=self.early_stopping_patience,
                 early_stopping_threshold=self.early_stopping_threshold,
             )
        else:
             print("Early stopping disabled because no evaluation dataset is available.")


        # --- Trainer 초기화 ---
        print("Initializing Trainer...")
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset['train'],
            eval_dataset=tokenized_dataset.get('test'), # 없으면 None 전달
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            # EarlyStoppingCallback 추가 (리스트 형태로 전달)
            callbacks=[early_stopping_callback] if early_stopping_callback else []
        )

        # 메모리 최적화를 위해 불필요한 객체 삭제
        del tokenized_dataset
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()

        # --- 학습 시작 ---
        print("Starting training...")
        # 체크포인트 재개 로직
        last_checkpoint = None
        if os.path.isdir(training_args.output_dir):
             from transformers.trainer_utils import get_last_checkpoint
             last_checkpoint = get_last_checkpoint(training_args.output_dir)
             if last_checkpoint:
                 print(f"Resuming training from checkpoint: {last_checkpoint}")

        try:
            train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
            print("Training finished.")

            metrics = train_result.metrics
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)

            # trainer.save_model()은 load_best_model_at_end=True 설정에 따라
            # 학습 종료 시점의 모델이 아닌, 가장 좋았던 체크포인트의 모델을 저장
            print(f"Saving final (best) LoRA adapter model to {self.output_dir}...")
            trainer.save_model(self.output_dir) # PEFT 어댑터와 설정을 저장
            self.tokenizer.save_pretrained(self.output_dir) # 토크나이저도 함께 저장
            print("Final (best) model adapter and tokenizer saved.")

        except Exception as e:
            print(f"\n!!! An error occurred during training: {e} !!!")
            import traceback
            traceback.print_exc()
            if "CUDA out of memory" in str(e):
                print("\n--- OOM Error Suggestion ---")
                print("1. Reduce `per_device_train_batch_size` (currently {}).".format(batch_size))
                print("2. Increase `gradient_accumulation_steps` (currently {}).".format(gradient_accumulation_steps))
                print("3. Ensure 4-bit quantization (`use_quantization=True`) and gradient checkpointing (`use_gradient_checkpointing=True`) are enabled.")
                print("4. Try reducing `lora_r` (currently {}).".format(self.lora_r))
                # print("5. Check `MengziDataProcessor` for max sequence length, maybe reduce it.")
                print("----------------------------\n")

    def generate_response(self, prompt, max_new_tokens=150):
        """주어진 프롬프트에 대한 응답 생성"""
        if not hasattr(self, 'model') or not hasattr(self, 'tokenizer'):
            print("Model or tokenizer not initialized. Cannot generate response.")
            return None

        print(f"\nGenerating response for prompt: '{prompt[:100]}...'")

        # 모델 컨텍스트 길이 확인
        model_max_length = getattr(self.model.config, 'max_position_embeddings', 512)
        # 생성될 토큰 고려하여 입력 길이 제한
        max_input_length = model_max_length - max_new_tokens
        if max_input_length <= 0:
             print(f"Warning: max_new_tokens ({max_new_tokens}) is too large for model max length ({model_max_length}). Reducing max_new_tokens.")
             max_new_tokens = model_max_length // 2
             max_input_length = model_max_length - max_new_tokens

        # 입력 토큰화
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=max_input_length # 계산된 최대 입력 길이
        )

        # 중요: token_type_ids 제거
        if 'token_type_ids' in inputs:
            del inputs['token_type_ids']

        # 입력 텐서를 모델이 있는 장치로 이동 (device_map='auto' 고려)
        try:
            if hasattr(self.model, 'device'): # PEFT 모델은 device 속성 가짐
                 model_device = self.model.device
            elif hasattr(self.model, 'base_model'): # base_model 확인
                 first_param_device = next(self.model.base_model.parameters()).device
                 model_device = first_param_device
            else: # 일반 모델 확인
                 first_param_device = next(self.model.parameters()).device
                 model_device = first_param_device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            print(f"Input tensors moved to device: {model_device}")
        except Exception as e:
             print(f"Could not automatically move inputs to model device: {e}. "
                   "Keeping inputs on CPU, generate() might handle it.")

        # 생성 설정
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "num_return_sequences": 1,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id
        }

        print(f"Generating with config: {generation_config}")

        try:
            with torch.no_grad(): # 추론 시에는 그래디언트 계산 필요 없음
                 outputs = self.model.generate(**inputs, **generation_config)

            input_ids_len = inputs['input_ids'].shape[1]
            response = self.tokenizer.decode(outputs[0][input_ids_len:], skip_special_tokens=True)
            print("Generated response (decoded).")
            return response.strip()

        except Exception as e:
            print(f"\n!!! Error during generation: {e} !!!")
            import traceback
            traceback.print_exc()
            if "CUDA out of memory" in str(e):
                print("\n--- OOM Error Suggestion (Generation) ---")
                print(f"1. Reduce `max_new_tokens` (currently {max_new_tokens}).")
                print("2. Ensure the model was loaded with quantization if intended.")
                print("3. Try generating with a shorter prompt.")
                print("-----------------------------------------\n")
            return None


# --- Kaggle 노트북에서 실행 ---
if __name__ == '__main__':
    print("Starting LoRA Fine-tuning Process on Kaggle...")

    # --- 환경 확인 ---
    use_gpu = torch.cuda.is_available()
    print(f"GPU Available: {use_gpu}")
    if use_gpu:
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        print(f"BF16 Supported: {torch.cuda.is_bf16_supported()}")
    else:
        print("Running on CPU. Training will be very slow, quantization is disabled, and early stopping requires an eval set (which might not work well on CPU).")

    # --- 트레이너 인스턴스 생성 ---
    try:
        trainer_instance = MengziModelTrainer(
            data_path=KAGGLE_INPUT_DATA_PATH,
            output_dir=KAGGLE_OUTPUT_DIR,
            model_name='beomi/KoAlpaca-Polyglot-5.8B', # 또는 다른 모델
            lora_r=8,          # 메모리 부족 시 4 등으로 줄이기
            lora_alpha=16,       # 보통 lora_r * 2
            lora_dropout=0.05,   # 0.05 ~ 0.1
            use_quantization=use_gpu, # GPU 사용 시에만 양자화 활성화
            use_gradient_checkpointing=use_gpu, # GPU 사용 시에만 활성화 (CPU에서 True면 에러날 수 있음)
            # Early stopping 파라미터 전달
            early_stopping_patience=2, # 2번의 평가 동안 개선 없으면 중단
            early_stopping_threshold=0.001, # loss가 0.001 이상 개선되어야 함 (선택적)
        )
    except Exception as e:
        print(f"Failed to initialize MengziModelTrainer: {e}")
        print("Please check the error messages above, especially regarding MengziDataProcessor or PEFT setup.")
        raise SystemExit("Trainer initialization failed.")


    # --- 데이터 준비 ---
    # test_size=0.1 로 설정하여 검증 데이터셋 생성 (Early stopping에 필요)
    tokenized_dataset = trainer_instance.prepare_training(test_size=0.1)

    # --- 학습 실행 ---
    if tokenized_dataset:
        trainer_instance.train(tokenized_dataset)

        # --- 학습 후 메모리 정리 (선택 사항) ---
        # del trainer_instance.model # 필요 시 모델 객체 삭제
        # gc.collect()
        # if use_gpu:
        #     torch.cuda.empty_cache()
        # 만약 삭제했다면 아래 테스트 생성 전에 다시 로드해야 함

        # --- 테스트 응답 생성 ---
        # load_best_model_at_end=True 이므로, trainer_instance.model은
        # 검증 데이터셋에서 가장 좋은 성능을 보인 시점의 모델 상태입
        print("\n--- Testing Response Generation After Training (Using Best Model) ---")

        test_prompt = "백성을 다스리는 가장 중요한 원칙은 무엇입니까?"
        response = trainer_instance.generate_response(test_prompt, max_new_tokens=150)
        if response:
             print("-" * 30)
             print(f"질문: {test_prompt}")
             print(f"맹자의 응답 (Fine-tuned): {response}")
             print("-" * 30)

        test_prompt_2 = "군자는 무엇을 경계해야 합니까?"
        response_2 = trainer_instance.generate_response(test_prompt_2, max_new_tokens=100)
        if response_2:
            print(f"질문: {test_prompt_2}")
            print(f"맹자의 응답 (Fine-tuned): {response_2}")
            print("-" * 30)

    else:
        print("Dataset preparation failed. Training and generation skipped.")

    print("Fine-tuning process finished.")

    # Kaggle 출력 디렉토리 내용 확인
    print("\n--- Output Directory Contents ---")
    if os.path.exists(KAGGLE_OUTPUT_DIR):
        output_items = os.listdir(KAGGLE_OUTPUT_DIR)
        if output_items:
            # 체크포인트 폴더 확인
            checkpoint_folders = [item for item in output_items if item.startswith('checkpoint-')]
            print(f"Checkpoints saved: {len(checkpoint_folders)}")
            print("Other files/folders:")
            for item in output_items:
                 # adapter_model.bin 또는 adapter_model.safetensors 가 있는지 확인
                 if item == "adapter_model.bin" or item == "adapter_model.safetensors":
                     print(f"- {item} (BEST MODEL ADAPTER)")
                 elif not item.startswith('checkpoint-'):
                    print(f"- {item}")
        else:
            print(f"Output directory {KAGGLE_OUTPUT_DIR} is empty.")
    else:
        print(f"Output directory {KAGGLE_OUTPUT_DIR} not found (Training might have failed before saving).")