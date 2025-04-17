import argparse
import copy

parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=100)
parser.add_argument('--index', type=int, default=1)
parser.add_argument('--gpu_index', type=int, nargs='+', default=[0])
parser.add_argument('--outdir', type=str, default='outdir0')
args = parser.parse_args()
import os

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)[1:-1]
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import json
from fastchat.model.model_adapter import get_conversation_template

bigname="/home/chokwans99/LoRA_Eagle/EAGLE_LoRA/mymodels/vicuna-7b-v1.3"
# bigname = "/home/lyh/weights/hf/llama/7B/"
# smallname = "/home/lyh/weights/hf/llama/7B/"



def longest_common_prefix(list1, list2):
    prefix_length = 0
    min_length = min(len(list1), len(list2))

    for i in range(min_length):
        if list1[i] == list2[i]:
            prefix_length += 1
        else:
            break

    common_prefix = list1[:prefix_length]
    return common_prefix, prefix_length


def build_dataset_rank(
        tokenizer, split="train",
        select=None,
):
    # WMT16 영어-독일어 데이터셋 로드
    ds = load_dataset("wmt16", "de-en", split=split)
    ds = ds.shuffle(seed=42)
    
    # 시작과 끝 인덱스로 데이터 선택
    ds1 = ds.select(range(args.start, args.end))
    original_columns1 = ds1.column_names
    num_proc = 4

    def preprocess_function(examples):
        new_examples = {
            "conversation":[],
            "input_ids": [],
            "loss_mask": []
        }
        
        for i in range(len(examples['translation'])):
            # vicuna 대화 템플릿 가져오기
            conv = get_conversation_template("vicuna")
            
            # 영어 원문과 독일어 번역문 추출
            english_text = examples['translation'][i]['en']
            german_text = examples['translation'][i]['de']
            
            # 번역을 위한 프롬프트 생성
            instruction = f"Translate the following English text to German:\n\n{english_text}"
            
            # 대화 형식으로 변환
            conv.append_message(conv.roles[0], instruction)
            conv.append_message(conv.roles[1], german_text)
            
            conversation = conv.get_prompt()
            
            # 토큰화
            input_ids = tokenizer(
                conversation,
                return_tensors="pt",
                max_length=tokenizer.model_max_length,
                truncation=True,
            ).input_ids[0]
            
            # loss mask 생성 - 독일어 출력 부분만 학습
            loss_mask = torch.ones_like(input_ids)
            
            sep = conv.sep + conv.roles[1] + ": "
            total_len = int(input_ids.ne(tokenizer.pad_token_id).sum())

            turns = conversation.split(conv.sep2)
            cur_len = 1
            loss_mask[:cur_len] = 0
            
            for j, turn in enumerate(turns):
                if turn == "":
                    break
                turn_len = len(tokenizer(turn).input_ids)

                parts = turn.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

                if j != 0 and not tokenizer.legacy:
                    # The legacy and non-legacy modes handle special tokens differently
                    instruction_len -= 1

                # 지시문(영어 원문) 부분은 loss 계산에서 제외
                loss_mask[cur_len: cur_len + instruction_len] = 0
                cur_len += turn_len

                if j != 0 and not tokenizer.legacy:
                    # The legacy and non-legacy modes handle special tokens differently
                    cur_len -= 1

            loss_mask[cur_len:] = 0

            # 문장이 너무 길면 건너뛰기
            if len(input_ids) > tokenizer.model_max_length * 0.9:  # 90% 이상 차지하면 스킵
                continue
                
            new_examples["conversation"].append(conversation)
            new_examples["input_ids"].append(input_ids[None,:])
            new_examples["loss_mask"].append(loss_mask[None,:])

        return new_examples

    ds1 = ds1.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns1,
        load_from_cache_file=False
    )

    # 필터링: 빈 예제 제거
    ds1 = ds1.filter(lambda x: len(x["input_ids"]) > 0, batched=False)
    
    ds1.set_format(type="torch")
    return ds1

bigtokenizer = AutoTokenizer.from_pretrained(bigname, use_fast=False)
ds = build_dataset_rank(bigtokenizer)
print(ds)
bigmodel = AutoModelForCausalLM.from_pretrained(bigname, device_map="auto", torch_dtype=torch.float16)
bigmodel.eval()

@torch.no_grad()
def ge(data):
    input_ids = data["input_ids"]
    outs_big = bigmodel(input_ids.cuda(), output_hidden_states=True)
    hidden_state_big = outs_big.hidden_states[-1]
    max_prob_tokens_big = torch.argmax(outs_big.logits, dim=-1)
    probs = torch.softmax(outs_big.logits, dim=-1)
    maxp = probs[0].max(dim=1).values
    td = {
        "input_ids": input_ids.cpu()[0],
        "hidden_state": hidden_state_big.cpu()[0],
        "loss_mask": data["loss_mask"].cpu()[0]
    }
    return td

outdir = f'{args.outdir}/{args.index}'
if not os.path.exists(outdir):
    os.makedirs(outdir)

def writedata(name, data_point):
    if not os.path.exists(name):
        os.makedirs(name)
    current_length = len(os.listdir(name))
    idx = current_length
    torch.save(data_point, f'{name}/data_{idx}.ckpt')

for data in ds:
    outdata = ge(data)
    writedata(outdir, outdata)


