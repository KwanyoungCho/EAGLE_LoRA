CUDA_VISIBLE_DEVICES=0 python -m eagle.evaluation.gen_baseline_answer_vicuna \
  --base-model-path /home/chokwans99/EAGLE-2/EAGLE/mymodels \
  --ea-model-path "yuhuili/EAGLE-Vicuna-7B-v1.3" \
  --num-gpus-per-model 1 \
  --num-gpus-total 1 \
  --model-id "baseline-mt-bench" \