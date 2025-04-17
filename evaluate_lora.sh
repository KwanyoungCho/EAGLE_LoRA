CUDA_VISIBLE_DEVICES=0 python -m eagle.evaluation.lora_gen_ea_answer_vicuna \
  --base-model-path /home/chokwans99/LoRA_Eagle/EAGLE_LoRA/mymodels/vicuna-7b-v1.3 \
  --ea-model-path "yuhuili/EAGLE-Vicuna-7B-v1.3" \
  --lora-path /home/chokwans99/LoRA_Eagle/EAGLE_LoRA/checkpoints/cnn_dailymail_lora_20250417_133301/lora_weights_19 \
  --num-gpus-per-model 1 \
  --num-gpus-total 1 \
  --model-id "lora-mt-bench-new" \
  