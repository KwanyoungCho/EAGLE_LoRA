python -m eagle.evaluation.lora_gen_baseline_answer_vicuna \
  --base-model-path /home/chokwans99/EAGLE-2/EAGLE/mymodels \
  --ea-model-path "yuhuili/EAGLE-Vicuna-7B-v1.3" \
  --lora-path /home/chokwans99/EAGLE-2/EAGLE/checkpoints/lora_20250416_115832/lora_weights_19 \
  --model-id lora-vicuna-13b-fp16-baseline \
