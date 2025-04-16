# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

save_dir = "/home/chokwans99/EAGLE-2/EAGLE/mymodels"

tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.3")
model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.3")

model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)