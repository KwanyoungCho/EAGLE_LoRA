import argparse

parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--basepath', type=str, default='/home/lyh/weights/hf/vicuna_v13/7B/')
parser.add_argument('--configpath', type=str, default="config.json")
parser.add_argument('--pretrained_model_path', type=str, default=None, help='사전 학습된 EAGLE 모델 경로')
parser.add_argument('--lr', type=float, default=3e-5)
parser.add_argument('--bs', type=int, default=4)
parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
parser.add_argument('--tmpdir', type=str, default='0')
parser.add_argument('--cpdir', type=str, default='0')
parser.add_argument('--lora-r', type=int, default=8, help='LoRA 랭크')
parser.add_argument('--lora-alpha', type=int, default=16, help='LoRA 알파')
parser.add_argument('--lora-dropout', type=float, default=0.05, help='LoRA 드롭아웃')
# EAGLE 모델 관련 매개변수 추가
parser.add_argument('--total-token', type=int, default=59, help='EAGLE 모델의 총 토큰 수')
parser.add_argument('--depth', type=int, default=5, help='EAGLE 모델 depth')
parser.add_argument('--top-k', type=int, default=10, help='EAGLE 모델의 top-k 값')
parser.add_argument('--threshold', type=float, default=1.0, help='EAGLE 모델의 threshold 값')
args = parser.parse_args()

train_config = {
    "lr": args.lr,
    "bs": args.bs,
    "gradient_accumulation_steps": args.gradient_accumulation_steps,
    "datapath": f"{args.tmpdir}",
    "is_warmup": True,
    "num_epochs": 20,
    "num_warmup_steps": 2000,
    "total_steps": 800000,
    "p_w": 0.1,
    "v_w": 1.0,
    "head_w": 0.1,
    "num_workers": 2,
    "embeding": True,
    "act": "No",
    "data_noise": True,
    "noise": "uniform",
    "mean": 0.0,
    "std": 0.2,
    "residual": "true,norm",
    "max_len": 2048,
    "config_path": args.configpath,
    "b1": 0.9,
    "b2": 0.95,
    "grad_clip": 0.5,
    "save_freq": 5,
    # LoRA 파라미터
    "lora_r": args.lora_r,
    "lora_alpha": args.lora_alpha,
    "lora_dropout": args.lora_dropout,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "pretrained_model_path": args.pretrained_model_path,
    # EAGLE 모델 파라미터 추가
    "total_token": args.total_token,
    "depth": args.depth,
    "top_k": args.top_k,
    "threshold": args.threshold,
}
import json
from safetensors import safe_open
import os
import torch

torch.backends.cuda.matmul.allow_tf32 = True
from accelerate import Accelerator
from accelerate.utils import set_seed

set_seed(0)
accelerator = Accelerator(mixed_precision='bf16',
                          gradient_accumulation_steps=train_config["gradient_accumulation_steps"])
from ..model.cnets import Model
from ..model.configs import EConfig
from typing import Any, Dict, List

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from transformers import get_linear_schedule_with_warmup, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType, PeftModelForFeatureExtraction
from ..model.lora_utils import CustomPeftModelForFeatureExtraction

if accelerator.is_main_process:
    import wandb
    wandb.init(project="eagle-lora-cnn_dailymail", entity="chaile9983", config=train_config)

# 기본 모델 설정 로드
baseconfig = AutoConfig.from_pretrained(args.basepath)

# 언어 모델 헤드 초기화 
head = torch.nn.Linear(baseconfig.hidden_size, baseconfig.vocab_size, bias=False)

# 사전학습된 언어 모델 헤드 가중치 로드 시도
try:
    # safetensors 형식에서 로드 시도
    with open(os.path.join(args.basepath, "model.safetensors.index.json"), "r") as f:
        index_json = json.loads(f.read())
        head_path = index_json["weight_map"]["lm_head.weight"]
    with safe_open(os.path.join(args.basepath, head_path),
                   framework="pt",
                   device="cpu") as f:
        tensor_slice = f.get_slice("lm_head.weight")
        vocab_size, hidden_dim = tensor_slice.get_shape()
        tensor = tensor_slice[:, :hidden_dim].float()
except:
    # pytorch 모델 형식에서 로드 시도
    with open(os.path.join(args.basepath, "pytorch_model.bin.index.json"), "r") as f:
        index_json = json.loads(f.read())
        head_path = index_json["weight_map"]["lm_head.weight"]
    weights = torch.load(os.path.join(args.basepath, head_path))
    tensor = weights["lm_head.weight"].float()

# 로드된 가중치로 헤드 초기화
head.weight.data = tensor
head.eval()

# 헤드의 파라미터는 학습하지 않음
for param in head.parameters():
    param.requires_grad = False



def list_files(path):
    datapath = []
    for root, directories, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            datapath.append(file_path)
    return datapath


class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.0):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        noise = torch.randn(tensor.size()) * self.std + self.mean
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data


class AddUniformNoise:
    def __init__(self, std=0.0):
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        # 균일 분포(-0.5~0.5)에서 노이즈 생성 후 스케일링 
        # 시퀀스 길이에 따라 노이즈 크기 조정 (512/길이)
        noise = (torch.rand_like(tensor) - 0.5) * self.std * 512 / tensor.shape[1]
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data


class CustomDataset(Dataset):
    def __init__(self, datapath, transform=None):
        self.data = datapath
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = torch.load(self.data[index])
        new_data = {}
        hidden_state = data['hidden_state'][:train_config["max_len"]][None, :]
        input_ids = data['input_ids'][:train_config["max_len"]][None, :]
        loss_mask = data["loss_mask"][:train_config["max_len"]][None, :]

        length = hidden_state.shape[1]
        attention_mask = [1] * length
        loss_mask = loss_mask[0].tolist()
        loss_mask[-1] = 0

        input_ids_target = input_ids[:, 1:]
        zeropadding = torch.tensor([[0]])
        input_ids_target = torch.cat((input_ids_target, zeropadding), dim=1)

        target = hidden_state[:, 1:, :]
        zeropadding = torch.zeros(1, 1, target.shape[2])
        target = torch.cat((target, zeropadding), dim=1)
        loss_mask[-1] = 0
        
        new_data["attention_mask"] = attention_mask
        new_data["loss_mask"] = loss_mask
        new_data["target"] = target
        new_data["hidden_state_big"] = hidden_state
        new_data["input_ids"] = input_ids_target

        if self.transform:
            new_data = self.transform(new_data)

        return new_data


class DataCollatorWithPadding:
    def paddingtensor(self, intensors, N):
        B, n, S = intensors.shape
        padding_tensor = torch.zeros(B, N - n, S)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors, N):
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item['hidden_state_big'].shape[1] for item in features)
        batch_input_ids = torch.cat([self.paddingtensor2D(item['input_ids'], max_length) for item in features])
        batch_hidden_states = torch.cat([self.paddingtensor(item['hidden_state_big'], max_length) for item in features])
        batch_target = torch.cat([self.paddingtensor(item['target'], max_length) for item in features])
        batch_loss_mask = torch.tensor(
            [item['loss_mask'] + [0] * (max_length - len(item['loss_mask'])) for item in features])
        batch_attention_mask = torch.tensor(
            [item['attention_mask'] + [0] * (max_length - len(item['attention_mask'])) for item in features])
        
        batch = {
            "input_ids": batch_input_ids,
            "hidden_states": batch_hidden_states,
            "target": batch_target,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
        }
        return batch


def top_accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res

def compute_loss(target, target_p, predict, loss_mask):
    out_head = head(predict)
    out_logp = nn.LogSoftmax(dim=2)(out_head)
    plogp = target_p * out_logp
    ploss = -torch.sum(torch.sum(loss_mask * plogp, 2)) / (loss_mask.sum() + 1e-5)
    vloss = criterion(predict, target)
    vloss = torch.sum(torch.mean(loss_mask * vloss, 2)) / (loss_mask.sum() + 1e-5)
    return vloss, ploss, out_head

@torch.no_grad()
def getkacc(model, data, head, max_length=5):
    def generate(hidden_states, input_ids, head, max_length=4, use_cache=True):
        if use_cache:
            past_key_values = None
            for i in range(max_length):
                if past_key_values != None:
                    out_hidden, past_key_values = model(last_hidden, input_ids=token, past_key_values=past_key_values,
                                               use_cache=True)
                else:
                    out_hidden, past_key_values = model(hidden_states, input_ids=input_ids, use_cache=True)
                last_hidden = out_hidden[:, -1:]
                last_headout = head(last_hidden)
                token = torch.argmax(last_headout, dim=-1)
                input_ids = torch.cat((input_ids, token), dim=1)
        else:
            raise NotImplementedError

        return input_ids

    hidden_states = data["hidden_states"]
    input_ids = data["input_ids"]
    loss_mask = data["loss_mask"]
    target = data["target"]
    total = [0 for _ in range(max_length)]
    correct = [0 for _ in range(max_length)]
    bs, seq_len = hidden_states.shape[0], hidden_states.shape[1]
    target_headout = head(target)
    target_ids = target_headout.argmax(dim=2)

    for pre_len in range(1, seq_len):
        if loss_mask[:, pre_len].sum() == 0:
            continue
        pre_hidden_states = hidden_states[:, :pre_len]
        pre_input_ids = input_ids[:, :pre_len]
        outs = generate(pre_hidden_states, pre_input_ids, head, max_length=max_length)
        generate_ids = outs[:, pre_len:]
        for bid in range(bs):
            for k in range(max_length):
                if loss_mask[bid, pre_len + k] == 0:
                    break
                if pre_len + k >= seq_len:
                    break
                total[k] += 1
                if generate_ids[bid, k] == target_ids[bid, pre_len + k - 1]:
                    correct[k] += 1
                else:
                    for kk in range(k + 1, max_length):
                        total[kk] += 1
                    break

    acc = [correct[i] / total[i] for i in range(len(correct))]
    return acc


if train_config["data_noise"]:
    if train_config["noise"] == "uniform":
        aug = AddUniformNoise(std=train_config["std"])
    else:
        aug = AddGaussianNoise(mean=train_config["mean"], std=train_config["std"])
else:
    aug = None

datapath = list_files(train_config["datapath"])

traindatapath = datapath[:int(len(datapath) * 0.95)]
testdatapath = datapath[int(len(datapath) * 0.95):]

traindataset = CustomDataset(traindatapath, transform=aug)
testdataset = CustomDataset(testdatapath)
train_loader = DataLoader(traindataset, batch_size=train_config["bs"], shuffle=True,
                          collate_fn=DataCollatorWithPadding(), num_workers=train_config["num_workers"],
                          pin_memory=True)
test_loader = DataLoader(testdataset, batch_size=train_config["bs"], shuffle=False,
                         collate_fn=DataCollatorWithPadding(), num_workers=train_config["num_workers"], pin_memory=True)

if accelerator.is_main_process:
    if not os.path.exists(args.cpdir):
        os.makedirs(args.cpdir)

# EAGLE 모델 설정 및 초기화
if accelerator.is_main_process:
    print(f"EAGLE 모델 초기화 시작")

# config 로드 및 모델 초기화
if args.pretrained_model_path:
    configpath = os.path.join(args.pretrained_model_path, "config.json")
    if not os.path.exists(configpath):
        from huggingface_hub import hf_hub_download
        configpath = hf_hub_download(args.pretrained_model_path, "config.json")
        
    config = EConfig.from_pretrained(configpath)
    
    # ea_model.py와 동일한 방식으로 bias 값 설정
    with open(configpath, "r") as f:
        con = json.loads(f.read())
    try:
        bias = con["bias"]
    except:
        bias = True
    
    # ea_model.py와 동일한 방식으로 모델 초기화
    model = Model(
        config, 
        bias=bias, 
        total_tokens=train_config["total_token"],
        depth=train_config["depth"],
        top_k=train_config["top_k"],
        threshold=train_config["threshold"]
    )
    
    # 가중치 로드
    if accelerator.is_main_process:
        print(f"사전 학습된 EAGLE 모델 로드 중: {args.pretrained_model_path}")
    
    try:
        # 가중치 파일 찾기 및 로드
        try:
            load_model_path = os.path.join(args.pretrained_model_path, "pytorch_model.bin")
            if not os.path.exists(load_model_path):
                from huggingface_hub import hf_hub_download
                load_model_path = hf_hub_download(args.pretrained_model_path, "pytorch_model.bin")
            
            if accelerator.is_main_process:
                print(f"PyTorch 모델 파일에서 가중치 로드: {load_model_path}")
            
            ea_layer_state_dict = torch.load(load_model_path, map_location="cpu")
        except:
            from safetensors.torch import load_file
            load_model_path = os.path.join(args.pretrained_model_path, "model.safetensors")
            if not os.path.exists(load_model_path):
                from huggingface_hub import hf_hub_download
                load_model_path = hf_hub_download(args.pretrained_model_path, "model.safetensors")
            
            if accelerator.is_main_process:
                print(f"Safetensors 파일에서 가중치 로드: {load_model_path}")
            
            ea_layer_state_dict = load_file(load_model_path)
        
        # ea_model.py와 동일하게 strict=True로 설정
        model.load_state_dict(ea_layer_state_dict, strict=True)
        
        # 모델 초기화 완료
        if accelerator.is_main_process:
            print(f"EAGLE 모델 로드 완료")
            
        # 가중치 로드 후 LoRA 적용
        if accelerator.is_main_process:
            print(f"LoRA 적용 시작")
        
        # 모델에 LoRA 적용
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=train_config["lora_r"],
            lora_alpha=train_config["lora_alpha"],
            lora_dropout=train_config["lora_dropout"],
            target_modules=train_config["lora_target_modules"]
        )
        
        # 모델에 LoRA 적용
        model = get_peft_model(model, peft_config)
        
        # 커스텀 PeftModel로 래핑하여 forward 메서드 오버라이드
        model = CustomPeftModelForFeatureExtraction(model, peft_config)
        
        # 기존 모델 파라미터는 학습하지 않도록 동결
        for name, param in model.named_parameters():
            if 'lora_' not in name:  # LoRA 파라미터가 아닌 경우
                param.requires_grad = False
            else:
                param.requires_grad = True  # LoRA 파라미터만 학습 가능하게 설정
        
        if accelerator.is_main_process:
            model.print_trainable_parameters()  # 학습 가능한 파라미터 수 출력
            print(f"LoRA 적용 완료")
            
    except Exception as e:
        if accelerator.is_main_process:
            print(f"모델 로드 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
else:
    # 사전 학습된 모델이 없는 경우 기본 초기화
    config = EConfig.from_pretrained(train_config["config_path"])
    model = Model(
        config, 
        bias=True,
        total_tokens=train_config["total_token"],
        depth=train_config["depth"],
        top_k=train_config["top_k"],
        threshold=train_config["threshold"],
        load_emb=True, 
        path=args.basepath
    )
    if accelerator.is_main_process:
        print(f"기본 EAGLE 모델 초기화 완료")
        
    # 초기화 후 LoRA 적용
    if accelerator.is_main_process:
        print(f"LoRA 적용 시작")
    
    # 모델에 LoRA 적용
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=train_config["lora_r"],
        lora_alpha=train_config["lora_alpha"],
        lora_dropout=train_config["lora_dropout"],
        target_modules=train_config["lora_target_modules"]
    )
    
    # 모델에 LoRA 적용
    model = get_peft_model(model, peft_config)
    
    # 커스텀 PeftModel로 래핑하여 forward 메서드 오버라이드
    model = CustomPeftModelForFeatureExtraction(model, peft_config)
    
    # 기존 모델 파라미터는 학습하지 않도록 동결
    for name, param in model.named_parameters():
        if 'lora_' not in name:  # LoRA 파라미터가 아닌 경우
            param.requires_grad = False
        else:
            param.requires_grad = True  # LoRA 파라미터만 학습 가능하게 설정
    
    if accelerator.is_main_process:
        model.print_trainable_parameters()  # 학습 가능한 파라미터 수 출력
        print(f"LoRA 적용 완료")

# ea_model.py처럼 초기화 추가
model.init_tree()

# 모델 구조 출력
# print(model)

# LoRA 파라미터 상태 확인을 위한 디버깅 코드
if accelerator.is_main_process:
    print("\n===== LoRA 파라미터 상태 확인 =====")
    lora_params = 0
    all_params = 0
    trainable_params = 0
    
    # LoRA 파라미터 확인
    lora_param_names = []
    non_lora_param_names = []
    
    for name, param in model.named_parameters():
        all_params += param.numel()
        
        if 'lora_' in name:
            lora_params += param.numel()
            lora_param_names.append(name)
        else:
            non_lora_param_names.append(name)
        
        if param.requires_grad:
            trainable_params += param.numel()
    
    # 결과 출력
    print(f"전체 파라미터 수: {all_params:,}")
    print(f"LoRA 파라미터 수: {lora_params:,} ({lora_params/all_params*100:.4f}%)")
    print(f"학습 가능한 파라미터 수: {trainable_params:,} ({trainable_params/all_params*100:.4f}%)")
    print(f"LoRA 파라미터 이름 (처음 10개): {lora_param_names[:10]}")
    
    # LoRA 파라미터가 없는 경우 경고
    if len(lora_param_names) == 0:
        print("\n경고: LoRA 파라미터가 발견되지 않았습니다!")
        print("PEFT 라이브러리가 제대로 적용되었는지 확인하세요.")
        
        # 모델 구조 확인
        print("\n모델 구조 정보:")
        print(f"모델 타입: {type(model)}")
        
        # 첫 번째 레이어 확인
        for name, module in model.named_modules():
            if "base_model.model.model.layers.0" in name:
                print(f"첫 번째 레이어 이름: {name}")
                print(f"첫 번째 레이어 타입: {type(module)}")
                break
    else:
        print("\nLoRA 파라미터가 성공적으로 적용되었습니다.")
    print("============================\n")

# 손실 함수, 옵티마이저 설정
criterion = nn.SmoothL1Loss(reduction="none")
# 학습 가능한 LoRA 파라미터만 최적화
optimizer = optim.AdamW(
    [p for n, p in model.named_parameters() if p.requires_grad and 'lora_' in n], 
    lr=train_config["lr"], 
    betas=(train_config["b1"], train_config["b2"])
)
num_epochs = train_config["num_epochs"]
num_warmup_steps = train_config["num_warmup_steps"]
total_steps = train_config["total_steps"]
is_warmup = train_config["is_warmup"]

if is_warmup:
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=total_steps)

    model, head, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
        model, head, optimizer, train_loader, test_loader, scheduler
    )
else:
    model, head, optimizer, train_loader, test_loader = accelerator.prepare(
        model, head, optimizer, train_loader, test_loader
    )
for epoch in range(num_epochs + 1):
    top_3acc = [0 for _ in range(3)]
    correct = 0
    total = 0
    epoch_loss = 0
    num_batches = 0
    model.train()
    for batch_idx, data in enumerate(tqdm(train_loader)):
        with accelerator.accumulate(model):
            optimizer.zero_grad()
            predict = model(data["hidden_states"], input_ids=data["input_ids"], attention_mask=data["attention_mask"])
            with torch.no_grad():
                target_head = head(data["target"])
                target_p = nn.Softmax(dim=2)(target_head)
                target_p = target_p.detach()
            loss_mask = data["loss_mask"][:, :, None]
            vloss, ploss, out_head = compute_loss(data["target"], target_p, predict, loss_mask)
            loss = train_config["v_w"] * vloss + train_config["p_w"] * ploss
            accelerator.backward(loss)
            accelerator.clip_grad_value_(model.parameters(), train_config["grad_clip"])
            optimizer.step()
            if is_warmup:
                scheduler.step()
        with torch.no_grad():
            _, predicted = torch.max(out_head, 2)
            _, target = torch.max(target_head, 2)
            ct = loss_mask.sum().item()
            cc = ((predicted == target) * loss_mask.squeeze()).sum().item()
            out_head = out_head.view(-1, target_head.shape[-1])[loss_mask.view(-1) == 1]
            target = target.view(-1)[loss_mask.view(-1) == 1]
            topkacc = top_accuracy(out_head, target, (1, 2, 3))
            for top_i in range(len(topkacc)):
                top_3acc[top_i] += topkacc[top_i]
            total += ct
            correct += cc
        if accelerator.is_main_process and ct != 0:
            logdict = {"train/lr": optimizer.param_groups[0]["lr"], "train/vloss": vloss.item(),
                       "train/ploss": ploss.item(), "train/loss": loss.item(), "train/acc": cc / ct}
            for id, i in enumerate(top_3acc):
                logdict[f'train/top_{id + 1}_acc'] = topkacc[id].item() / ct
            wandb.log(logdict)

        del ploss, vloss
        epoch_loss += loss.item()
        num_batches += 1

    correct, total = torch.tensor(correct).cuda(), torch.tensor(total).cuda()
    correct, total = accelerator.gather_for_metrics((correct, total))
    correct, total = correct.sum().item(), total.sum().item()
    epoch_loss /= num_batches
    top_3acc = accelerator.gather_for_metrics(top_3acc)
    if accelerator.is_local_main_process:
        for id, i in enumerate(top_3acc):
            wandb.log({f'train/epochtop_{id + 1}_acc': i.sum().item() / total})
    if accelerator.is_local_main_process:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
        print('Train Accuracy: {:.2f}%'.format(100 * correct / total))
        wandb.log({"train/epochacc": correct / total, "train/epochloss": epoch_loss})

    if (epoch + 1) % train_config["save_freq"] == 0:
        top_3acc = [0 for _ in range(3)]
        correct = 0
        total = 0
        epoch_loss = 0
        num_batches = 0
        model.eval()

        k_acc = [[] for i in range(5)]
        for batch_idx, data in enumerate(tqdm(test_loader)):
            with torch.no_grad():
                if batch_idx < 10:
                    acces = getkacc(model, data, head, max_length=5)
                    for i in range(len(acces)):
                        k_acc[i].append(acces[i])
                predict = model(data["hidden_states"], input_ids=data["input_ids"],
                                attention_mask=data["attention_mask"])
                target_head = head(data["target"])
                target_p = nn.Softmax(dim=2)(target_head)
                target_p = target_p.detach()
                loss_mask = data["loss_mask"][:, :, None]
                vloss, ploss, out_head = compute_loss(data["target"], target_p, predict, loss_mask)
                loss = train_config["v_w"] * vloss + train_config["p_w"] * ploss
                _, predicted = torch.max(out_head, 2)
                _, target = torch.max(target_head, 2)
                ct = loss_mask.sum().item()
                cc = ((predicted == target) * loss_mask.squeeze()).sum().item()
                out_head = out_head.view(-1, target_head.shape[-1])[loss_mask.view(-1) == 1]
                target = target.view(-1)[loss_mask.view(-1) == 1]
                topkacc = top_accuracy(out_head, target, (1, 2, 3))
                for top_i in range(len(topkacc)):
                    top_3acc[top_i] += topkacc[top_i]
                total += ct
                correct += cc
            epoch_loss += loss.item()
            num_batches += 1

        mean_acces = []
        for id, i in enumerate(k_acc):
            mean_acc = np.array(i).mean()
            mean_acc = torch.tensor(mean_acc).cuda()
            mean_acces.append(mean_acc)

        mean_acces = accelerator.gather_for_metrics(mean_acces)
        if accelerator.is_local_main_process:
            for id, i in enumerate(mean_acces):
                mean_acc = i.mean().item()
                wandb.log({f"test/{id}_acc": mean_acc})

        correct, total = torch.tensor(correct).cuda(), torch.tensor(total).cuda()
        correct, total = accelerator.gather_for_metrics((correct, total))
        correct, total = correct.sum().item(), total.sum().item()
        top_3acc = accelerator.gather_for_metrics(top_3acc)
        if accelerator.is_local_main_process:
            for id, i in enumerate(top_3acc):
                wandb.log({f'test/top_{id + 1}_acc': i.sum().item() / total})
        epoch_loss /= num_batches
        if accelerator.is_local_main_process:
            print('Test Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
            print('Test Accuracy: {:.2f}%'.format(100 * correct / total))
            wandb.log({"test/epochacc": correct / total, "test/epochloss": epoch_loss})
            
            # LoRA 가중치 저장
            model.save_pretrained(f"{args.cpdir}/lora_weights_{epoch}")
            # 전체 상태 저장
            accelerator.save_state(output_dir=f"{args.cpdir}/state_{epoch}")
