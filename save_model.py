# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "vicuna-7b-v1.3"
save_dir = f"/home/chokwans99/LoRA_Eagle/EAGLE_LoRA/mymodels/{model_name}"

tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.3")
model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.3")

model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)