#!/bin/bash

# EAGLE 모델에서 LoRA 파라미터만 학습하는 스크립트

# # 캐시 디렉토리 설정
# export TRANSFORMERS_CACHE="/home/chokwans99/.cache/huggingface/hub"
# export HF_HOME="/home/chokwans99/.cache/huggingface"

# # GPU 설정 - GPU 1번만 사용
# export CUDA_VISIBLE_DEVICES=1

# 필요한 경로 설정
BASE_MODEL_PATH="/home/chokwans99/EAGLE-2/EAGLE/mymodels"  # 베이스 모델 경로 (LLaMA, Qwen 등)
# EAGLE_MODEL_PATH="/home/chokwans99/EAGLE/models/EAGLE-Vicuna-7B-v1.3"  # 사전 학습된 EAGLE 모델 경로
EAGLE_MODEL_PATH="yuhuili/EAGLE-Vicuna-7B-v1.3"  # 사전 학습된 EAGLE 모델 경로
CONFIG_PATH="/home/chokwans99/EAGLE-2/EAGLE/mymodels"  # 설정 파일 경로
TRAIN_DATA_PATH="/home/chokwans99/EAGLE-2/EAGLE/gendata/sharegpt_0_67_mufp16"  # 학습 데이터 경로

CHECKPOINT_DIR="checkpoints/lora_$(date +%Y%m%d_%H%M%S)"  # 체크포인트 저장 경로

# 학습 관련 설정
LEARNING_RATE=3e-5
BATCH_SIZE=1  # 단일 GPU에서 실행할 배치 사이즈
GRADIENT_ACCUMULATION_STEPS=1

# LoRA 관련 설정
LORA_RANK=8
LORA_ALPHA=16
LORA_DROPOUT=0.05

# 디버깅 및 메모리 관리 설정
# export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"  # 메모리 단편화 방지

# 디렉토리 생성
mkdir -p $CHECKPOINT_DIR

# 먼저 모델 다운로드
python -c "from transformers import AutoTokenizer, AutoConfig; AutoTokenizer.from_pretrained('$BASE_MODEL_PATH', trust_remote_code=True); AutoConfig.from_pretrained('$BASE_MODEL_PATH', trust_remote_code=True)"

# lora_main.py를 수정하여 사전 학습된 EAGLE 모델을 로드할 수 있도록 추가 인자 전달
accelerate launch \
    --gpu_ids 0 \
    --mixed_precision bf16 \
    -m eagle.train.lora_main \
    --basepath $BASE_MODEL_PATH \
    --configpath $CONFIG_PATH \
    --pretrained_model_path $EAGLE_MODEL_PATH \
    --lr $LEARNING_RATE \
    --bs $BATCH_SIZE \
    --gradient-accumulation-steps $GRADIENT_ACCUMULATION_STEPS \
    --tmpdir $TRAIN_DATA_PATH \
    --cpdir $CHECKPOINT_DIR \
    --lora-r $LORA_RANK \
    --lora-alpha $LORA_ALPHA \
    --lora-dropout $LORA_DROPOUT

echo "LoRA 학습이 완료되었습니다. 체크포인트는 $CHECKPOINT_DIR 에 저장되었습니다."