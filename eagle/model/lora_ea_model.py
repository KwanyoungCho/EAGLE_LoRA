import copy
import json
import time
import os
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, PreTrainedModel, PretrainedConfig, AutoConfig

# LoRA 관련 라이브러리 추가
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from peft.utils import _get_submodules

from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .modeling_mixtral_kv import MixtralForCausalLM as KVMixtralForCausalLM
from .modeling_qwen2_kv import LlamaForCausalLM as KVQwen2ForCausalLM
from .utils import *
from .kv_cache import initialize_past_key_values

from .cnets import Model
from .configs import EConfig

# 커스텀 LoRA 모델 래퍼 클래스 (학습/추론 모드 함께 지원)
from .lora_utils import CustomPeftModelForFeatureExtraction


# 안전한 디바이스 감지 함수
def get_safe_device(device=None):
    """
    시스템에서 사용 가능한 안전한 디바이스를 반환합니다.
    지정된 디바이스가 없거나 유효하지 않은 경우 대체 디바이스를 반환합니다.
    """
    # 디바이스가 명시적으로 지정된 경우
    if device is not None:
        try:
            if device.type == 'cuda' and device.index is not None:
                if device.index >= torch.cuda.device_count():
                    print(f"경고: {device}는 유효하지 않습니다. 대체 디바이스를 사용합니다.")
                    device = None
                else:
                    return device
            else:
                return device
        except:
            device = None
    
    # 디바이스가 지정되지 않았거나 유효하지 않은 경우
    if torch.cuda.is_available():
        # 사용 가능한 GPU 중 첫 번째 사용
        return torch.device('cuda:0')
    else:
        # GPU가 없으면 CPU 사용
        return torch.device('cpu')


class EaModel(nn.Module):
    def __init__(
            self,
            base_model,
            base_model_name_or_path,
            ea_model_path,
            total_token,
            depth,
            top_k,
            threshold,
            ea_layer_state_dict,
            lora_path=None,
            lora_config=None
    ):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path, use_fast=False)

        # 안전한 디바이스 감지
        try:
            # 기본 모델의 레이어 디바이스 가져오기 시도
            layer_device = base_model.model.layers[-1].self_attn.q_proj.weight.device
            self.device = get_safe_device(layer_device)
        except:
            # 오류 발생 시 기본 디바이스 사용
            self.device = get_safe_device()
        
        print(f"모델이 사용할 디바이스: {self.device}")
        self.dtype = self.base_model.dtype

        # EA 모델 구성
        config = EConfig.from_pretrained(ea_model_path)
        with open(ea_model_path, "r") as f:
            con = json.loads(f.read())
        try:
            bias = con["bias"]
        except:
            bias = True
        self.ea_layer = Model(
            config,
            bias=bias,
            total_tokens=total_token,
            depth=depth,
            top_k=top_k,
            threshold=threshold
        )

        # ea_model.py와 동일한 방식으로 디바이스 관리
        low_memory = False
        try:
            layer_device = base_model.model.layers[-1].self_attn.q_proj.weight.device
            head_device = base_model.lm_head.weight.device
            
            # 디바이스 유효성 확인
            layer_device = get_safe_device(layer_device)
            head_device = get_safe_device(head_device)
            
            if layer_device != head_device:
                self.ea_layer.diff_device = True
                if not low_memory:
                    self.ea_layer.headweight = base_model.lm_head.weight.clone().to(layer_device)
                else:
                    self.ea_layer.layer_device = layer_device
            else:
                self.ea_layer.diff_device = False
        except Exception as e:
            print(f"디바이스 설정 중 오류 발생: {e}, 기본 설정 사용")
            self.ea_layer.diff_device = False

        # 상태 딕셔너리 로드
        try:
            # 가능한 에러 방지를 위해 CPU에 먼저 로드
            if isinstance(ea_layer_state_dict, dict):
                # 이미 메모리에 로드된 상태 딕셔너리인 경우
                self.ea_layer.load_state_dict(ea_layer_state_dict, strict=True)
            else:
                raise ValueError("ea_layer_state_dict는 딕셔너리여야 합니다")
        except Exception as e:
            print(f"상태 딕셔너리 로드 중 오류 발생: {e}")
            raise

        # 모델을 적절한 디바이스와 데이터 타입으로 이동
        self.ea_layer = self.ea_layer.to(self.dtype).to(self.device)

        # LoRA 가중치 적용 (존재하는 경우)
        if lora_path is not None:
            self.apply_lora(lora_path, lora_config)
            # 최종 디바이스 확인
            self.ea_layer = self.ea_layer.to(self.dtype).to(self.device)

        # 마지막으로 init_tree() 호출
        self.ea_layer.init_tree()
        
    def apply_lora(self, lora_path, lora_config=None):
        """
        EAGLE 모델에 LoRA 가중치를 적용합니다.
        Args:
            lora_path (str): LoRA 가중치 파일 경로
            lora_config (dict, optional): LoRA 설정. 지정하지 않으면 기본값 사용
        """
        # lora_main.py와 유사하게 디바이스 관리
        print(f"LoRA 가중치 적용 시작: {lora_path}")
        
        # 안전한 디바이스 확인
        device = self.device
        dtype = self.dtype

        # 기본 LoRA 설정
        if lora_config is None:
            lora_config = {
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            }

        # LoRA 설정 생성 (lora_main.py와 유사)
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=True,  # 추론 모드
            r=lora_config["r"],
            lora_alpha=lora_config["lora_alpha"],
            lora_dropout=lora_config["lora_dropout"],
            target_modules=lora_config["target_modules"]
        )

        # ea_layer를 올바른 디바이스로 이동
        self.ea_layer = self.ea_layer.to(device)
        
        # LoRA 적용: EA layer를 PEFT 래핑
        self.ea_layer = get_peft_model(self.ea_layer, peft_config)

        # 추론용 커스텀 래퍼로 감싸기 (inference_mode=True로 설정)
        self.ea_layer = CustomPeftModelForFeatureExtraction(self.ea_layer, peft_config, inference_mode=True)

        # LoRA 가중치 로드 - lora_main.py에서의 접근 방식과 유사하게
        try:
            if os.path.isdir(lora_path):
                # 디렉토리인 경우 - adapter_model.bin 또는 adapter_model.safetensors 파일 확인
                adapter_path = None
                if os.path.exists(os.path.join(lora_path, "adapter_model.bin")):
                    adapter_path = os.path.join(lora_path, "adapter_model.bin")
                    print(f"LoRA 바이너리 파일 로드: {adapter_path}")
                    state_dict = torch.load(adapter_path, map_location='cpu')
                elif os.path.exists(os.path.join(lora_path, "adapter_model.safetensors")):
                    adapter_path = os.path.join(lora_path, "adapter_model.safetensors")
                    print(f"LoRA safetensors 파일 로드: {adapter_path}")
                    from safetensors.torch import load_file
                    # 안전하게 CPU에 먼저 로드
                    state_dict = load_file(adapter_path)
                else:
                    raise ValueError(f"LoRA 파일을 찾을 수 없습니다: {lora_path}")
            else:
                # 직접적인 파일 경로인 경우
                adapter_path = lora_path
                if lora_path.endswith(".bin"):
                    print(f"LoRA 바이너리 파일 로드: {adapter_path}")
                    state_dict = torch.load(adapter_path, map_location='cpu')
                elif lora_path.endswith(".safetensors"):
                    print(f"LoRA safetensors 파일 로드: {adapter_path}")
                    from safetensors.torch import load_file
                    # 안전하게 CPU에 먼저 로드
                    state_dict = load_file(adapter_path)
                else:
                    raise ValueError(f"지원되지 않는 파일 형식: {lora_path}")
                
            # 로드 후 텐서를 타겟 디바이스로 이동
            print(f"LoRA 가중치를 {device}로 이동")
            state_dict = {k: v.to(device) for k, v in state_dict.items()}
            
        except Exception as e:
            print(f"LoRA 가중치 로드 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            raise

        # 키 구조 불일치 문제 해결 - 키 변환
        new_state_dict = {}
        for key, value in state_dict.items():
            # 중첩 구조 처리 (base_model.model.layers -> base_model.base_model.model.layers)
            if key.startswith("base_model.model.layers"):
                new_key = "base_model." + key
                
                # LoRA 어댑터 이름 처리 (.weight -> .default.weight)
                if ".lora_A.weight" in new_key:
                    new_key = new_key.replace(".lora_A.weight", ".lora_A.default.weight")
                elif ".lora_B.weight" in new_key:
                    new_key = new_key.replace(".lora_B.weight", ".lora_B.default.weight")
                
                new_state_dict[new_key] = value
            else:
                # 기타 키는 그대로 유지
                new_state_dict[key] = value
        
        print(f"LoRA 가중치 키 구조 변환: {len(state_dict)}개 -> {len(new_state_dict)}개")
        
        # 디버깅을 위한 키 샘플 출력
        if len(state_dict) > 0:
            orig_keys = list(state_dict.keys())
            converted_keys = [k for k in new_state_dict.keys() if k.startswith("base_model.base_model")]
            
            if orig_keys and converted_keys:
                print("키 변환 예시:")
                print(f"  원본: {orig_keys[0]}")
                print(f"  변환: {converted_keys[0] if converted_keys else '변환된 키 없음'}")

        # state_dict 적용 시도
        try:
            # 변환된 state_dict를 적용하고, 키 매칭 정보 확인
            load_result = self.ea_layer.load_state_dict(new_state_dict, strict=False)
            if load_result.missing_keys:
                print("Warning: The following keys are missing in the loaded LoRA state dict:")
                for key in load_result.missing_keys[:10]:  # 너무 많을 경우 일부만 출력
                    print(f"  - {key}")
                if len(load_result.missing_keys) > 10:
                    print(f"  ... 외 {len(load_result.missing_keys) - 10}개 더 있음")
                    
            if load_result.unexpected_keys:
                print("Warning: The following keys are unexpected in the loaded LoRA state dict:")
                for key in load_result.unexpected_keys[:10]:
                    print(f"  - {key}")
                if len(load_result.unexpected_keys) > 10:
                    print(f"  ... 외 {len(load_result.unexpected_keys) - 10}개 더 있음")
        except Exception as e:
            print(f"LoRA 가중치 적용 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            raise

        print(f"LoRA 가중치가 성공적으로 로드되었습니다: {adapter_path}")

        # 명시적인 디바이스/타입 설정
        self.ea_layer = self.ea_layer.to(dtype).to(device)
        
        # 파라미터 중 하나를 확인하여 올바른 디바이스에 있는지 확인
        for name, param in self.ea_layer.named_parameters():
            if 'lora_' in name:
                print(f"LoRA 파라미터 '{name}' 디바이스: {param.device}")
                break

    @classmethod
    def from_pretrained(
            cls,
            Type="LLaMA",
            base_model_path=None,
            ea_model_path=None,
            total_token=59,
            depth=5,
            top_k=10,
            threshold=1.0,
            # LoRA 관련 매개변수
            lora_path=None,
            lora_config=None,
            **kwargs,
    ):
        # lora_main.py와 유사하게 모델 로드
        print(f"기본 모델 로드 시작: {base_model_path}")
        
        # 아키텍처 타입 확인
        arch_type = AutoConfig.from_pretrained(base_model_path).architectures[0]
        
        # 기본 모델 로드
        try:
            if arch_type == 'LlamaForCausalLM':
                base_model = KVLlamaForCausalLM.from_pretrained(base_model_path, **kwargs)
            elif arch_type == 'Qwen2ForCausalLM':
                base_model = KVQwen2ForCausalLM.from_pretrained(base_model_path, **kwargs)
            else:
                base_model = KVMixtralForCausalLM.from_pretrained(base_model_path, **kwargs)
                
            print(f"기본 모델 타입: {arch_type}")
        except Exception as e:
            print(f"기본 모델 로드 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            raise

        # EA 모델 구성 파일 경로
        try:
            configpath = os.path.join(ea_model_path, "config.json")
            if not os.path.exists(configpath):
                configpath = hf_hub_download(ea_model_path, "config.json")
                
            print(f"EA 모델 구성 파일 로드: {configpath}")
        except Exception as e:
            print(f"구성 파일 로드 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            raise

        # 장치 감지 및 설정
        device = get_safe_device()
        print(f"가중치 로드에 사용할 디바이스: {device}")

        # EA 모델 가중치 로드
        try:
            try:
                load_model_path = os.path.join(ea_model_path, "pytorch_model.bin")
                if not os.path.exists(load_model_path):
                    load_model_path = hf_hub_download(ea_model_path, "pytorch_model.bin")
                print(f"PyTorch 모델 파일에서 가중치 로드: {load_model_path}")
                ea_layer_state_dict = torch.load(load_model_path, map_location='cpu')
            except:
                from safetensors.torch import load_file
                load_model_path = os.path.join(ea_model_path, "model.safetensors")
                if not os.path.exists(load_model_path):
                    load_model_path = hf_hub_download(ea_model_path, "model.safetensors")
                print(f"Safetensors 파일에서 가중치 로드: {load_model_path}")
                ea_layer_state_dict = load_file(load_model_path)
        except Exception as e:
            print(f"모델 가중치 로드 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            raise

        # LoRA 설정이 제공된 경우 구성
        if lora_path is not None and lora_config is None:
            lora_config = {
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            }

        # 최종 모델 생성
        try:
            model = cls(
                base_model,
                base_model_path,
                configpath,
                total_token,
                depth,
                top_k,
                threshold,
                ea_layer_state_dict,
                lora_path=lora_path,
                lora_config=lora_config
            )
            print("EAGLE 모델 초기화 성공")
        except Exception as e:
            print(f"EAGLE 모델 초기화 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            raise

        # (옵션) total_token이 -1이면, 성능 측정 후 적절한 토큰 길이 설정
        if total_token == -1:
            try:
                device = get_safe_device()  # 안전한 디바이스 사용
                cans = [40, 48, 50, 56, 60]
                x = [1, 1.05, 1.07, 1.1, 1.13]
                times = []
                for i in range(len(cans)):
                    length = cans[i]
                    input_ids = torch.randint(0, model.config.vocab_size - 200, (1, length)).to(device)
                    torch.cuda.synchronize()
                    start_time = time.time()
                    for _ in range(20):
                        torch.cuda.synchronize()
                        with torch.no_grad():
                            outputs = model.base_model(input_ids)
                        torch.cuda.synchronize()
                    torch.cuda.synchronize()
                    end_time = time.time()
                    times.append((end_time - start_time) / x[i])
                total_token = cans[times.index(min(times))]
                model.ea_layer.total_tokens = total_token - 1
                print(f"최적의 토큰 수로 설정됨: {total_token}")
            except Exception as e:
                print(f"토큰 수 최적화 중 오류 발생: {e}")
                # 오류 발생 시 기본값 사용
                model.ea_layer.total_tokens = 59
                print("기본 토큰 수 59로 설정됨")

        return model

    def get_tokenizer(self):
        """
        Get the tokenizer of the base model.
        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            past_key_values=None,
            output_orig=False,
            position_ids=None,
    ):
        with torch.inference_mode():
            # base_model의 언어 모델 부분에 입력 전달
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
            hidden_states = outputs[0]
        if output_orig:
            return outputs, orig, hidden_states
        else:
            return outputs, hidden_states

    @torch.no_grad()
    def eagenerate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,
    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        max_length = max_length - self.ea_layer.total_tokens - 10

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()



        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token = initialize_tree(
            input_ids, self, past_key_values, logits_processor
        )
        new_token = 0

        for idx in range(max_length):
            #with Timer("all"):
            self.base_model.model.tree_mask = tree_mask

            draft_tokens=draft_tokens.to(input_ids.device)
            #with Timer("tree_decoding"):
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )
            #retrieve_indices=tree_buffers["retrieve_indices"]
            #logits = logits[0, retrieve_indices]
            draft_tokens=torch.cat((draft_tokens,padding),dim=1)
            candidates=draft_tokens[0,retrieve_indices]
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )
            # print(accept_length)
            #with Timer("update_inference_inputs"):
            input_ids, draft_tokens, retrieve_indices,tree_mask,tree_position_ids, new_token, hidden_state, sample_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state_new,
                sample_p
            )

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
        if not log:
            return input_ids
        else:
            return input_ids, new_token, idx

    # 이하의 naivegenerate, ea_generate, naive_generate 함수들은 동일한 구조를 유지합니다.
    @torch.no_grad()
    def naivegenerate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        max_length = max_length - self.ea_layer.total_tokens - 10

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()



        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        outputs = self.base_model(input_ids, past_key_values=past_key_values, use_cache=True)
        new_token = 0

        for idx in range(max_length):
            if logits_processor is not None:
                logits = outputs.logits[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = outputs.logits[:, -1:].argmax(dim=-1)
            outputs = self.base_model(input_id, use_cache=True, past_key_values=past_key_values)
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            new_token+=1

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
        if not log:
            return input_ids
        else:
            return input_ids, new_token, idx

    @torch.no_grad()
    def ea_generate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        max_length=max_length-self.ea_layer.total_tokens-10

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        #assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding=(torch.zeros(1,1,dtype=torch.long)-1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()



        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        draft_tokens, retrieve_indices,tree_mask,tree_position_ids, logits, hidden_state, sample_token = initialize_tree(
            input_ids, self, past_key_values, logits_processor
        )
        new_token = 0

        for idx in range(max_length):
            #with Timer("all"):
            self.base_model.model.tree_mask = tree_mask

            draft_tokens=draft_tokens.to(input_ids.device)
            #with Timer("tree_decoding"):
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )
            #retrieve_indices=tree_buffers["retrieve_indices"]
            #logits = logits[0, retrieve_indices]
            draft_tokens=torch.cat((draft_tokens,padding),dim=1)
            candidates=draft_tokens[0,retrieve_indices]
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )
            # print(accept_length)
            #with Timer("update_inference_inputs"):
            input_ids, draft_tokens, retrieve_indices,tree_mask,tree_position_ids, new_token, hidden_state, sample_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_key_values_data,
                current_length_data,
                self,
                hidden_state_new,
                sample_p
            )

            yield input_ids

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break


    @torch.no_grad()
    def naive_generate(
            self,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,

    ):
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        max_length = max_length - self.ea_layer.total_tokens - 10

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        # assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        self.ea_layer.reset_kv()

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)
        outputs = self.base_model(input_ids, past_key_values=past_key_values, use_cache=True)
        new_token = 0


        for idx in range(max_length):
            if logits_processor is not None:
                logits = outputs.logits[:, -1]
                logits = logits_processor(None, logits)
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                input_id = torch.multinomial(probabilities, 1)
            else:
                input_id = outputs.logits[:, -1:].argmax(dim=-1)

            outputs = self.base_model(input_id, use_cache=True, past_key_values=past_key_values)
            input_ids = torch.cat([input_ids, input_id], dim=-1)
            new_token += 1

            yield input_ids



            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
