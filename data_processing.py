import json
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoConfig, AutoTokenizer # AutoConfig는 최대 길이 확인 위해 유지
import os
import warnings

# Kaggle 환경의 데이터 경로 
KAGGLE_INPUT_DIR = '/kaggle/input/mencious-dataset'
DATA_FILE_NAME = 'mencius_dataset.json'   

class MengziDataProcessor:
    def __init__(self,
                 data_path=None,
                 tokenizer=None, # <<-- 1. tokenizer 인자 추가
                 model_name=None, # model_name은 이제 선택사항 (Config 로드용)
                 max_source_length=512,
                 max_target_length=512):
        """
        맹자 데이터를 전처리하고 모델 학습을 위해 준비하는 클래스 (수정됨)

        :param data_path: JSON 데이터 파일의 **전체 경로**. None이면 오류 발생.
        :param tokenizer: 외부에서 **미리 로드된** Tokenizer 객체. None이면 오류 발생.
        :param model_name: 모델 설정 참조용 모델 이름 (선택 사항).
        :param max_source_length: 입력 시퀀스의 최대 토큰 길이.
        :param max_target_length: 출력(레이블) 시퀀스의 최대 토큰 길이.
        """
        if data_path is None:
             raise ValueError("`data_path` must be provided.")
        # <<-- 2. 외부 tokenizer 객체 유효성 검사 추가 -->>
        if tokenizer is None:
             raise ValueError("A pre-loaded `tokenizer` object must be provided.")

        self.data_path = data_path
        # <<-- 3. 외부에서 받은 tokenizer 사용 -->>
        self.tokenizer = tokenizer
        self.model_name = model_name # model_name은 여전히 저장 (config 로드 등에 사용될 수 있음)
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

        # 모델의 실제 최대 입력 길이 확인 (tokenizer나 model_name 기반으로 시도)
        try:
            # tokenizer.name_or_path 또는 model_name을 사용
            config_source = self.model_name if self.model_name else self.tokenizer.name_or_path
            if config_source: # 이름/경로가 있어야 config 로드 가능
                config = AutoConfig.from_pretrained(config_source)
                model_max_len = getattr(config, 'max_position_embeddings', 512)
                if self.max_source_length > model_max_len:
                     warnings.warn(
                         f"Warning: max_source_length ({self.max_source_length}) exceeds model's maximum position embeddings ({model_max_len}). "
                         f"Ensure the model architecture supports this length or truncation might occur unexpectedly."
                     )
                if self.max_target_length > model_max_len:
                     warnings.warn(
                         f"Warning: max_target_length ({self.max_target_length}) exceeds model's maximum position embeddings ({model_max_len}). "
                         f"Consider if this is intended, as labels are often truncated based on model capacity."
                     )
            else:
                print("Warning: Cannot determine model config source (model_name or tokenizer.name_or_path). Skipping max length check.")
        except Exception as e:
            print(f"Could not check model's max length from config (this is often okay): {e}")


    def load_data(self):
        """
        JSON 데이터 로드 및 형식 통일 (단일 턴 + 3단계 대화 처리)
        """
        raw_data = None
        print(f"Attempting to load data from: {self.data_path}")
        try:
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"Input file not found at the specified path: {self.data_path}")
            with open(self.data_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            if not isinstance(raw_data, list):
                 raise ValueError("JSON data root must be a list.")
            print(f"Successfully loaded {len(raw_data)} raw entries from {self.data_path}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Hint: Ensure the dataset is correctly added to your Kaggle notebook")
            print(f"Expected path structure: /kaggle/input/<your-dataset-slug>/{DATA_FILE_NAME}")
            print(f"Currently checking: {self.data_path}")
            print("\nCurrent working directory:", os.getcwd())
            if os.path.exists('/kaggle/input'):
                print("\nContents of /kaggle/input:")
                for item in os.listdir('/kaggle/input'): print(f"- {item}")
            else: print("\n/kaggle/input directory not found.")
            return None
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error loading or validating root of JSON data from {self.data_path}: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during data loading: {e}")
            return None

        processed_data = []
        processed_count = 0
        skipped_count = 0
        print("Processing and transforming data entries...")
        for item in raw_data:
            if not isinstance(item, dict):
                print(f"Warning: Skipping non-dictionary item: {item}")
                skipped_count += 1
                continue
            # 형식 2: 3단계 대화 데이터 처리
            if "initial_instruction" in item and "initial_output" in item and \
               "follow_up_1_instruction" in item and "follow_up_1_output" in item and \
               "follow_up_2_instruction" in item and "follow_up_2_output" in item:
                try:
                    q1 = item["initial_instruction"]; a1 = item["initial_output"]
                    q2 = item["follow_up_1_instruction"]; a2 = item["follow_up_1_output"]
                    q3 = item["follow_up_2_instruction"]; a3 = item["follow_up_2_output"]
                    if None in [q1, a1, q2, a2, q3, a3]: raise TypeError("None value found in multi-turn")
                    processed_data.append({"instruction": q1, "output": a1}); processed_count += 1
                    instruction_turn_2 = f"질문: {q1}\n답변: {a1}\n질문: {q2}"
                    processed_data.append({"instruction": instruction_turn_2, "output": a2}); processed_count += 1
                    instruction_turn_3 = f"{instruction_turn_2}\n답변: {a2}\n질문: {q3}"
                    processed_data.append({"instruction": instruction_turn_3, "output": a3}); processed_count += 1
                except KeyError as e:
                    print(f"Warning: Skipping multi-turn item due to missing key {e}: {item.get('initial_instruction', 'N/A')[:50]}...")
                    skipped_count += 1
                except TypeError as e:
                    print(f"Warning: Skipping multi-turn item due to unexpected type (potentially None value): {e} in item starting with {item.get('initial_instruction', 'N/A')[:50]}...")
                    skipped_count += 1
            # 형식 1: 단일 턴 데이터 처리
            elif "instruction" in item and "output" in item:
                if item["instruction"] is not None and item["output"] is not None:
                    processed_data.append({"instruction": item["instruction"], "output": item["output"]}); processed_count += 1
                else:
                    print(f"Warning: Skipping single-turn item with None value(s): {item.get('instruction', 'N/A')[:50]}...")
                    skipped_count += 1
            else: print(f"Warning: Skipping item with unrecognized format: {list(item.keys())}"); skipped_count += 1
        print(f"Data processing finished. Processed {processed_count} instruction-output pairs. Skipped {skipped_count} entries.")
        if not processed_data: print("Warning: No data could be processed.")
        return processed_data

    def create_dataset(self, test_size=0.2):
        """
        데이터셋을 훈련용과 검증용으로 분할
        """
        data = self.load_data()
        if data is None or not data:
            print("Error: No data loaded or processed to create dataset.")
            return None
        df = pd.DataFrame(data)
        df.dropna(subset=['instruction', 'output'], inplace=True)
        df = df[(df['instruction'].str.strip() != '') & (df['output'].str.strip() != '')]
        if len(df) == 0:
            print("Error: No valid data remaining after cleaning to create dataset.")
            return None
        if len(df) < 2 :
             print("Warning: Very few data points (<2), cannot split into train/test. Returning as train set only.")
             train_dataset = Dataset.from_pandas(df)
             return DatasetDict({'train': train_dataset})
        try:
            n_total = len(df); n_test = int(n_total * test_size)
            if n_test == 0 and n_total > 1: n_test = 1
            if n_test >= n_total: n_test = n_total - 1
            n_train = n_total - n_test
            if n_train <= 0:
                print("Warning: Calculated train size is zero or negative. Returning full data as train set.")
                train_dataset = Dataset.from_pandas(df)
                return DatasetDict({'train': train_dataset})

            train_df = df.sample(n=n_train, random_state=42)
            test_df = df.drop(train_df.index)
            if train_df.empty or test_df.empty:
                 print("Warning: Train or Test DataFrame became empty after sampling. Returning full data as train set.")
                 train_dataset = Dataset.from_pandas(df)
                 return DatasetDict({'train': train_dataset})
            train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
            test_dataset = Dataset.from_pandas(test_df, preserve_index=False)
            return DatasetDict({'train': train_dataset, 'test': test_dataset})
        except Exception as e:
            print(f"Error during dataset splitting: {e}")
            print("Returning the entire dataset as 'train' set.")
            train_dataset = Dataset.from_pandas(df)
            return DatasetDict({'train': train_dataset})

    def tokenize_dataset(self, dataset):
        """
        데이터셋 토큰화 및 불필요한 token_type_ids 컬럼 제거.
        max_length는 __init__에서 설정된 값을 사용합니다.
        """
        if dataset is None:
            print("Error: Cannot tokenize None dataset.")
            return None
        if 'train' not in dataset or not dataset['train']:
             print("Error: 'train' dataset is missing or empty in the input DatasetDict.")
             return None
    
        # 토큰화 함수 
        def tokenize_function(examples):
            # 입력 (Instruction) 토큰화
            model_inputs = self.tokenizer(
                examples['instruction'],
                max_length=self.max_source_length,
                truncation=True,
                padding='max_length' # 참고: Language Modeling에서는 보통 False로 두고 DataCollator에서 처리하는 것이 더 효율적일 수 있습니다.
            )
            # 출력 (Output) 토큰화 (레이블용)
            labels = self.tokenizer(
                text_target=examples['output'],
                max_length=self.max_target_length,
                truncation=True,
                padding='max_length' # 참고: 여기도 마찬가지입니다.
            )
            # 중요: 패딩된 레이블은 손실 계산에서 제외 (-100으로 변경)
            label_input_ids = labels['input_ids']
            padded_labels = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label_row]
                for label_row in label_input_ids
            ]
            model_inputs['labels'] = padded_labels
            return model_inputs
    
        try:
            required_columns = ['instruction', 'output']
            if not all(col in dataset['train'].column_names for col in required_columns):
                missing_cols = [col for col in required_columns if col not in dataset['train'].column_names]
                print(f"Error: Missing required columns for tokenization: {missing_cols}"); print(f"Available columns: {dataset['train'].column_names}"); return None
    
            # 원본 컬럼 이름 가져오기
            original_columns = list(dataset['train'].column_names)
            # map 함수에서 제거할 원본 컬럼들 계산
            # (tokenize_function이 반환하는 키: input_ids, attention_mask, labels)
            remove_cols = [col for col in original_columns] # 모든 원본 컬럼을 제거 대상으로 지정
    
            print(f"Applying tokenization and removing original columns: {remove_cols}")
            # .map() 적용: tokenize_function의 반환값으로 컬럼이 대체되고, remove_columns는 원본 컬럼을 제거
            tokenized_dataset_intermediate = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=remove_cols, # 원본 'instruction', 'output' 등 제거
            )
            print("Tokenization mapping complete. Checking for 'token_type_ids'...")
            print(f"Columns after mapping: {tokenized_dataset_intermediate['train'].column_names}") # 매핑 후 컬럼 확인
    
            # --- token_type_ids 제거 로직 ---
            final_tokenized_dataset = DatasetDict()
            for split in tokenized_dataset_intermediate.keys():
                current_split_dataset = tokenized_dataset_intermediate[split]
                # 현재 스플릿에 'token_type_ids' 컬럼이 있는지 확인
                if 'token_type_ids' in current_split_dataset.column_names:
                    print(f"Removing 'token_type_ids' column from '{split}' split.")
                    # .remove_columns()를 사용하여 해당 컬럼 제거 후 새 DatasetDict에 저장
                    final_tokenized_dataset[split] = current_split_dataset.remove_columns(['token_type_ids'])
                else:
                    # 없다면 그대로 새 DatasetDict에 저장
                    print(f"'token_type_ids' column not found in '{split}' split. Keeping as is.")
                    final_tokenized_dataset[split] = current_split_dataset
            # --- 제거 로직 끝 ---
    
            print("Finished processing splits for 'token_type_ids'.")
            print(f"Final columns in train split: {final_tokenized_dataset['train'].column_names}") # 최종 컬럼 확인
    
            return final_tokenized_dataset # token_type_ids가 제거된 최종 데이터셋 반환
    
        except Exception as e:
            print(f"Error during tokenization or column removal: {e}")
            import traceback
            traceback.print_exc()
            return None
        
# --- Kaggle 노트북에서 실행 시 (테스트용) ---
if __name__ == '__main__':
    # --- 설정 ---
    final_data_path = os.path.join(KAGGLE_INPUT_DIR, DATA_FILE_NAME)
    # 테스트용 모델 이름 
    test_model_name = 'beomi/KoAlpaca-Polyglot-5.8B' # Trainer와 같은 모델 사용 
    max_input_len = 512
    max_output_len = 512
    validation_split_size = 0.1
    # --------------------------

    print(f"--- Data Processing Test Start ---")
    print(f"Using model for tokenizer: {test_model_name}")
    print(f"Target data file path: {final_data_path}")
    print("-" * 30)

    # <<-- 테스트를 위해 토크나이저를 외부에서 로드 -->>
    try:
        external_tokenizer = AutoTokenizer.from_pretrained(test_model_name)
        print("External tokenizer loaded successfully.")
    except Exception as e:
        print(f"Failed to load tokenizer for testing: {e}")
        external_tokenizer = None # 실패 시 None 처리

    if external_tokenizer:
        # <<-- 수정된 Processor 생성 방식: 로드된 tokenizer 전달 -->>
        processor = MengziDataProcessor(
            data_path=final_data_path,
            tokenizer=external_tokenizer, # 로드된 토크나이저 전달
            model_name=test_model_name,   # model_name도 전달 (선택 사항)
            max_source_length=max_input_len,
            max_target_length=max_output_len
        )

        dataset = processor.create_dataset(test_size=validation_split_size)
        if dataset:
            print("\n--- Dataset Created ---"); print(dataset)
            tokenized_dataset = processor.tokenize_dataset(dataset)
            if tokenized_dataset:
                print("\n--- Tokenization Complete ---"); print(tokenized_dataset)
                if 'train' in tokenized_dataset and len(tokenized_dataset['train']) > 0:
                    print("\n--- Tokenized Sample (Train[0]) ---")
                    sample = tokenized_dataset['train'][0]
                    print("Input IDs:", sample['input_ids'][:50], "...")
                    print("Attention Mask:", sample['attention_mask'][:50], "...")
                    if 'labels' in sample:
                        label_ids = [label for label in sample['labels'] if label != -100]
                        # 외부 토크나이저(processor.tokenizer와 동일)로 디코딩
                        decoded_label = processor.tokenizer.decode(label_ids, skip_special_tokens=True)
                        print("Decoded Labels (Target):", decoded_label[:100], "...")
                else: print("No training samples available.")
            else: print("\n--- Tokenization Failed ---")
        else: print("\n--- Dataset Creation Failed ---")
    else:
        print("Tokenizer loading failed, cannot proceed with DataProcessor test.")

    print("\n--- Data Processing Test End ---")