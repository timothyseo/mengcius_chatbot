프로젝트 제목: 맹자의 지혜를 담다: LoRA 기반 맹자 챗봇 파인튜닝 프로젝트

**프로젝트 개요 (Project Overview)**
프로젝트 요약: KoAlpaca-Polyglot-5.8B 모델을 맹자 관련 데이터로 파인튜닝(LoRA)하여, 맹자의 사상과 어조를 반영하여 대화하는 인공지능 챗봇 개발
개발 동기 및 목표:
동양 고전, 특히 맹자 사상의 현대적 접근성 향상
텍스트 기반 학습을 넘어선 인터랙티브 학습/탐구 경험 제공
대규모 언어 모델(LLM)의 한국어 파인튜닝 기술(LoRA, 양자화) 적용 및 실험
목표: 맹자의 철학적 관점을 바탕으로 질문에 답변하고, 맹자 특유의 어조를 모방하는 챗봇 구현
기대 효과:
사용자의 맹자 및 동양 철학 이해 증진
교육 및 연구 보조 도구로서의 활용 가능성 탐색
LLM 파인튜닝 및 경량화 기술 역량 확보
**주요 기능 (Key Features)**
맹자 사상 기반 답변: 맹자의 핵심 사상(인, 의 등), 특정 개념, 가상 상황에 대한 맹자의 관점 관련 질문에 답변 (스크린샷 참고)
자연스러운 대화 흐름:
챗봇 페르소나(맹자) 설정 및 유지 (chatbot.py의 conversation_history)
이전 대화 맥락을 일부 반영하여 답변 생성
맹자 어조 모방: 파인튜닝을 통해 맹자 특유의 설득적이고 철학적인 어조 구현 시도
답변 생성 제어: 불필요한 반복 질문 생성을 방지하기 위해 특정 패턴(\n질문: 등) 감지 시 답변 생성을 중단하는 StoppingCriteria 적용 (chatbot.py의 StopOnTokens)
**타겟 사용자 (Target Audience)**
맹자, 동양 철학에 관심 있는 학생 및 일반 대중
동양 철학 관련 연구자 또는 교육자
AI 챗봇 기술, LLM 파인튜닝(LoRA) 및 양자화 기술에 관심 있는 개발자/학습자
**기술 스택 (Technology Stack)**
언어: Python
핵심 라이브러리: transformers, peft, torch, bitsandbytes, datasets, pandas
기반 모델: beomi/KoAlpaca-Polyglot-5.8B
파인튜닝 기법: LoRA (Low-Rank Adaptation)
  lora_r=8, lora_alpha=16, lora_dropout=0.05
  Target Modules: query_key_value, dense (자동 탐색 기반)
모델 경량화:
  학습: 4-bit Quantization (NF4, compute_dtype=bfloat16) (training.py)
  추론: 8-bit Quantization (chatbot.py)
개발 환경: Kaggle Notebooks
하드웨어: NVIDIA Tesla T4 GPU x 2
**시스템 아키텍처 (System Architecture) - (간략)**
[사용자 입력] → [Chatbot Interface (chatbot.py)] → [프롬프트 구성 (페르소나 + 대화기록 + 질문)] → [PEFT 모델 (KoAlpaca-5.8B + LoRA Adapter) (8-bit Quantized)] → [토크나이저] → [답변 생성] → [StopOnTokens 체크] → [응답 텍스트] → [사용자 출력]
주요 흐름:
  사용자 입력을 받아 챗봇 페르소나, 대화 기록과 결합하여 모델 입력 프롬프트 생성.
  사전 학습된 KoAlpaca 모델에 파인튜닝된 LoRA 어댑터를 적용하고 8-bit 양자화된 PEFT 모델을 로드.
  토크나이저를 사용하여 프롬프트를 토큰 ID로 변환 후 모델에 입력.
  모델이 답변 토큰 생성. StopOnTokens 조건 만족 시 생성 중단.
  생성된 토큰 ID를 텍스트로 디코딩하여 사용자에게 최종 응답 전달.
**개발 과정 (Development Process)**
1. 기획 및 데이터 준비: 맹자 챗봇 아이디어 구체화, 논어, 대학, 중용, 맹자 국역판으로부터 추출한 데이터 기반으로 맹자 관련 대화 데이터셋 (mencius_dataset.json) 구축 (단일 턴 QA 및 3단계 심층 대화 형식 포함).
2. 데이터 전처리 (data_processing.py):
  JSON 데이터 로드 및 형식 통일 (멀티턴 → 시퀀셜 턴 변환).
  pandas 활용 데이터 정제 및 datasets 라이브러리로 Train/Test 분할 (Test size 10%).
  AutoTokenizer (beomi/KoAlpaca-Polyglot-5.8B) 활용, Instruction-Output 쌍 토큰화, 패딩, Truncation.
  Loss 계산을 위해 Padding 토큰 레이블 -100으로 처리, token_type_ids 컬럼 제거.
3. 모델 및 학습 환경 설정 (training.py):
  Base LLM (beomi/KoAlpaca-Polyglot-5.8B) 선정.
  Kaggle T4 x 2 GPU 환경 구성.
  학습 최적화: 4-bit 양자화(NF4, BitsAndBytesConfig), PEFT(LoRA), prepare_model_for_kbit_training 적용.
4. 모델 파인튜닝 (training.py):
  transformers.Trainer 활용 LoRA 파인튜닝 수행.
  학습 설정: paged_adamw_8bit 옵티마이저, batch_size=1, gradient_accumulation_steps=8 (Effective Batch Size: 8), learning_rate=2e-4, num_train_epochs=3.
  검증 및 최적 모델 저장: evaluation_strategy="steps", eval_steps=50, save_steps=50, load_best_model_at_end=True, metric_for_best_model="eval_loss".
  조기 종료: EarlyStoppingCallback 적용 (patience=2, threshold=0.001)으로 과적합 방지 및 학습 시간 단축.
5. 챗봇 인터페이스 개발 (chatbot.py):
  학습된 최적 LoRA 어댑터 (checkpoint-250)를 PeftModel로 로드 (추론 시 8-bit 양자화).
  챗봇 페르소나 정의, 대화 기록 관리 기능 구현.
  StopOnTokens 구현하여 답변 품질 관리.
  사용자 인터랙티브 chat() 함수 구현.
6. 테스트 및 결과 확인: 학습 중 Validation Loss 모니터링, 학습 완료 후 샘플 질문 통해 답변 생성 품질 정성적 평가.
**개발 중 어려움 & 해결 (Challenges & Solutions)**
Challenge 1: 제한된 GPU 메모리 (T4 16GB x 2)로 5.8B 모델 파인튜닝
Solution:
  4-bit Quantization (NF4): bitsandbytes 라이브러리를 사용하여 모델 로드 시 메모리 사용량 대폭 감소.
  PEFT (LoRA): 전체 모델 파라미터가 아닌, 소수의 LoRA 파라미터만 학습하여 메모리 효율 증대.
  Gradient Accumulation: 작은 배치 사이즈(1)로 학습하되, 여러 스텝의 그래디언트를 누적(8)하여 실질적인 배치 사이즈 효과(8) 달성.
  Paged AdamW Optimizer: 옵티마이저 상태를 CPU 메모리로 페이징하여 GPU 메모리 절약.
Challenge 2: 챗봇의 불필요한 다음 질문 생성 또는 장황한 답변
Solution:
  Stopping Criteria: transformers.StoppingCriteria 상속받아 StopOnTokens 클래스 구현. 답변 생성 중 \n질문:, \n사용자 질문: 등 특정 토큰 시퀀스가 나타나면 생성을 즉시 중단시켜 간결하고 명확한 답변 유도.
Challenge 3: 최적의 학습 지점 탐색 및 과적합 방지
Solution:
  Evaluation & Early Stopping: TrainingArguments에서 evaluation_strategy="steps" 설정하여 일정 스텝(50)마다 검증 데이터셋(test)으로 성능(eval_loss) 평가. EarlyStoppingCallback(patience=2)을 사용하여 특정 횟수 이상 성능 개선이 없으면 학습 조기 중단. load_best_model_at_end=True로 설정하여 가장 좋았던 시점의 모델(checkpoint-250) 자동 저장.
Challenge 4: 다양한 형태의 맹자 데이터(단일 QA, 심층 대화) 통합 처리
Solution:
  Data Processor 로직 구현 (data_processing.py): JSON 파일 내 다른 형식의 데이터를 파싱하여 일관된 'instruction'-'output' 쌍으로 변환. 특히 3단계 대화는 이전 대화 내용을 'instruction'에 누적 포함시켜 문맥 학습 유도.
**결과 및 평가 (Results & Evaluation)**
학습 결과:
최종 학습된 체크포인트: checkpoint-250 (Early Stopping에 의해 결정됨)
학습 시간: 약 3시간 46분 (Epoch 2/3 완료 시점)
Train Loss (at step 350): ~0.3620
Validation Loss (at step 350): ~0.8993 (이후 개선 없어 중단)
Trainable Parameters: ~14.6M (0.2488%) - LoRA의 효율성 확인
결과물:
파인튜닝된 LoRA 어댑터 (adapter_model.safetensors) 및 설정 파일 성공적으로 저장 (/kaggle/working/mengzi_lora_finetuned).
학습 로그 및 메트릭 저장 완료.
정성적 평가 (Qualitative):
챗봇은 맹자의 핵심 사상(본성, 긍휼지심, 군자의 덕목 등)에 대해 철학적 관점을 담아 답변 생성.
맹자 특유의 어조를 일부 모방하는 경향 확인.
StopOnTokens 적용으로 불필요한 내용 생성 방지 효과 확인.
**향후 개선 방향 (Future Work)**
데이터셋 고도화:
  더 많은 맹자 원문, 주석, 해설서 데이터 추가 학습.
  다양한 질문 유형(비교, 비판적 질문 등) 데이터 보강.
  데이터 정제 및 일관성 검증 강화.
모델 성능 향상:
  LoRA 외 다른 PEFT 기법(예: QLoRA 심층 적용, IA3 등) 실험.
  더 큰 규모 또는 최신 한국어 LLM 기반 파인튜닝 시도.
  Hyperparameter tuning (LoRA rank, alpha, learning rate 스케줄 등) 추가 진행.
평가 체계 강화:
  ROUGE, BLEU 등 정량적 평가 지표 도입.
  맹자 전문가 또는 사용자 대상 블라인드 테스트 및 만족도 조사 수행.
기능 확장:
  특정 맹자 구절 인용 또는 출처 제시 기능 추가.
  웹/앱 인터페이스 개발 통한 사용자 접근성 향상.
**배운 점 및 느낀 점 (Learnings & Takeaways)**
기술 역량 강화:
  대규모 언어 모델(5.8B) 파인튜닝 전 과정(데이터 처리, 학습, 추론) 경험.
  LoRA 및 PEFT: 개념 이해 및 실제 적용, 효율성 체감.
  양자화(4-bit/8-bit): bitsandbytes 활용 모델 경량화 및 제한된 환경(T4 GPU)에서의 LLM 활용 능력 증대.
  transformers 라이브러리 심층 활용 (Trainer, TrainingArguments, StoppingCriteria 커스터마이징 등).
  Kaggle 환경에서의 GPU 활용 및 실험 관리 능력 향상.
도메인 지식 및 데이터:
  맹자 사상 및 고전 텍스트에 대한 이해 증진.
  고전 텍스트를 LLM 학습에 적합한 대화형 데이터셋으로 가공하는 것의 중요성 및 어려움 체감.
문제 해결 및 프로젝트 관리:
  메모리 부족(OOM) 등 기술적 문제 발생 시 다양한 해결 전략(양자화, PEFT, 옵티마이저 선택 등) 적용 및 트러블슈팅 경험.
  프로젝트 목표 설정부터 결과 도출까지 전 과정 관리 및 문서화(코드, 로그, PPT) 경험.
