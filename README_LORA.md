# EAGLE + LoRA 사용 가이드

## 소개

이 가이드는 EAGLE 모델에 LoRA를 적용하여 학습하고 추론하는 방법을 설명합니다. EAGLE(Efficient Accelerated Generation via Lookahead Eager Decoding)은 빠른 추론을 위한 모델로, LoRA를 적용하면 더 적은 자원으로 모델을 조정할 수 있습니다.

## 구성 요소

- **lora_main.py**: LoRA를 사용하여 EAGLE 모델을 학습하는 스크립트
- **lora_ea_model.py**: 학습된 LoRA 가중치를 적용한 EAGLE 모델로 추론하는 스크립트

## 학습 과정

1. **LoRA 학습**

```bash
python -m eagle.train.lora_main \
    --basepath /path/to/base_model \
    --configpath /path/to/config.json \
    --pretrained_model_path /path/to/eagle_model \
    --lr 3e-5 \
    --bs 4 \
    --tmpdir /path/to/training_data \
    --cpdir /path/to/save_checkpoints \
    --lora-r 8 \
    --lora-alpha 16 \
    --lora-dropout 0.05 \
    --total-token 59 \
    --depth 5 \
    --top-k 10 \
    --threshold 1.0
```

## 추론 과정

1. **LoRA 적용 EAGLE 모델로 추론**

```bash
python -m eagle.model.lora_ea_model \
    --base_model_path /path/to/base_model \
    --ea_model_path /path/to/eagle_model \
    --lora_path /path/to/lora_weights \
    --prompt "Hello, I am" \
    --device cuda:0
```

## 코드에서 직접 사용하기

```python
from eagle.model.lora_ea_model import EaModel
import torch

# 모델 설정
base_model_path = "/path/to/base_model"
ea_model_path = "/path/to/eagle_model"
lora_path = "/path/to/lora_weights"

# LoRA 설정
lora_config = {
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
}

# 모델 로드
model = EaModel.from_pretrained(
    base_model_path=base_model_path,
    ea_model_path=ea_model_path,
    lora_path=lora_path,
    lora_config=lora_config,
    total_token=59,
    depth=5,
    top_k=10,
    threshold=1.0
)

# 토크나이저 준비
tokenizer = model.get_tokenizer()

# 입력 인코딩
prompt = "Hello, I am"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# 텍스트 생성
outputs = model.eagenerate(
    input_ids=inputs["input_ids"],
    temperature=0.7,
    top_p=0.9,
    max_new_tokens=100
)

# 결과 디코딩
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

## 매개변수 설명

### 학습 매개변수
- `--basepath`: 기본 언어 모델 경로
- `--configpath`: EAGLE 모델 설정 파일 경로
- `--pretrained_model_path`: 사전 학습된 EAGLE 모델 경로
- `--lr`: 학습률
- `--bs`: 배치 크기
- `--tmpdir`: 학습 데이터 경로
- `--cpdir`: 체크포인트 저장 경로
- `--lora-r`: LoRA 랭크
- `--lora-alpha`: LoRA 알파
- `--lora-dropout`: LoRA 드롭아웃
- `--total-token`: EAGLE 모델의 총 토큰 수
- `--depth`: EAGLE 모델 깊이
- `--top-k`: 토큰 생성시 고려할 상위 k개 토큰
- `--threshold`: 토큰 선택 임계값

### 추론 매개변수
- `--base_model_path`: 기본 언어 모델 경로
- `--ea_model_path`: EAGLE 모델 경로
- `--lora_path`: LoRA 가중치 경로
- `--prompt`: 입력 프롬프트
- `--device`: 사용할 장치 (cuda:0, cpu 등)
- `--total_token`, `--depth`, `--top_k`, `--threshold`: EAGLE 모델 파라미터

## 주의사항

1. LoRA 가중치는 보통 `adapter_model.bin` 또는 `adapter_model.safetensors` 파일로 저장됩니다.
2. 학습과 추론 시 동일한 EAGLE 모델 파라미터(`total_token`, `depth`, `top_k`, `threshold`)를 사용해야 합니다.
3. 메모리 최적화를 위해 필요시 `device_map="auto"`를 사용할 수 있습니다. 