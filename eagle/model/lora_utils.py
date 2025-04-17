import torch
import torch.nn as nn

# EAGLE 모델을 위한 커스텀 PeftModel 클래스 정의
class CustomPeftModelForFeatureExtraction(nn.Module):
    def __init__(self, model, peft_config, inference_mode=False):
        super().__init__()
        self.base_model = model
        self.peft_config = peft_config
        self.inference_mode = inference_mode  # 추론 모드 플래그
        
        # 필요한 메서드 직접 연결 (중복 참조 없이)
        self.init_tree = getattr(model, 'init_tree', None)
        self.topK_genrate = getattr(model, 'topK_genrate', None)
        self.reset_kv = getattr(model, 'reset_kv', None)
        self.reset = getattr(model, 'reset', None)
        
        # EAGLE 모델에 필요한 중요 속성들 직접 복사
        # 이렇게 하면 속성을 검색할 때 무한 재귀를 방지할 수 있음
        if hasattr(model, 'total_tokens'):
            self.total_tokens = model.total_tokens
        if hasattr(model, 'diff_device'):
            self.diff_device = model.diff_device
        if hasattr(model, 'headweight'):
            self.headweight = model.headweight
        if hasattr(model, 'layer_device'):
            self.layer_device = model.layer_device
        if hasattr(model, 'depth'):
            self.depth = model.depth
        if hasattr(model, 'top_k'):
            self.top_k = model.top_k
        if hasattr(model, 'threshold'):
            self.threshold = model.threshold
        
        # 학습 가능한 매개변수 출력 기능 (lora_main.py와 호환)
        self.print_trainable_parameters = getattr(model, 'print_trainable_parameters', None)
    
    def __getattr__(self, name):
        """
        이 클래스에 없는 속성이나 메서드에 접근할 때 base_model에서 찾아 전달합니다.
        무한 재귀를 방지하기 위해 안전장치를 추가했습니다.
        """
        # 'base_model'에 대한 접근은 무한 재귀를 발생시킬 수 있으므로 즉시 예외 발생
        if name == 'base_model':
            raise AttributeError(f"'{self.__class__.__name__}'에 'base_model' 속성이 없습니다")
        
        # 기본 모델에서 속성을 가져옴
        if hasattr(self.base_model, name):
            return getattr(self.base_model, name)
        
        # 명시적인 오류 메시지
        raise AttributeError(f"'{self.__class__.__name__}' 또는 base_model에 '{name}' 속성이 없습니다")
    
    def save_pretrained(self, output_dir):
        if hasattr(self.base_model, 'save_pretrained'):
            return self.base_model.save_pretrained(output_dir)
        else:
            print("Warning: save_pretrained not implemented for this model")
    
    def forward(
        self,
        hidden_states,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        # 추론 모드에서는 requires_grad 설정 건너뜀
        if not self.inference_mode and hidden_states is not None and not hidden_states.requires_grad:
            hidden_states = hidden_states.detach().requires_grad_(True)
        
        # base_model의 forward 직접 호출
        outputs = self.base_model(
            hidden_states=hidden_states,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
        
        # 학습 모드에서만 outputs의 requires_grad 설정
        if not self.inference_mode:
            if use_cache and isinstance(outputs, tuple):
                hidden_out, cache = outputs
                if not hidden_out.requires_grad:
                    hidden_out = hidden_out.detach().requires_grad_(True)
                return hidden_out, cache
            
            if not isinstance(outputs, tuple) and not outputs.requires_grad:
                outputs = outputs.detach().requires_grad_(True)
        
        return outputs
