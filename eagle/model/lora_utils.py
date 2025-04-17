import torch
import torch.nn as nn

# EAGLE 모델을 위한 커스텀 PeftModel 클래스 정의
class CustomPeftModelForFeatureExtraction(nn.Module):
    def __init__(self, model, peft_config, inference_mode=False):
        super().__init__()
        self.base_model = model
        self.peft_config = peft_config
        self.inference_mode = inference_mode
        
        # 필요한 메서드들을 직접 참조 - 중복 참조 없이
        # 원본 모델 또는 내부 모델에서 메서드 찾기
        target_model = model.base_model if hasattr(model, 'base_model') else model
        
        # EAGLE 특수 메서드 복사
        self.init_tree = getattr(target_model, 'init_tree', None)
        self.topK_genrate = getattr(target_model, 'topK_genrate', None)
        self.reset_kv = getattr(target_model, 'reset_kv', None)
        self.reset = getattr(target_model, 'reset', None)
        
        # 학습 가능한 매개변수 출력 기능
        self.print_trainable_parameters = getattr(model, 'print_trainable_parameters', None)
            
        # EAGLE 모델에 필요한 중요 속성들 복사 - 간결한 방식으로
        for attr in ['total_tokens', 'diff_device', 'headweight', 
                    'layer_device', 'depth', 'top_k', 'threshold', 
                    'hf_device_map']:
            if hasattr(target_model, attr):
                setattr(self, attr, getattr(target_model, attr))
    
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
        # 추론 모드가 아닐 때만 그래디언트 흐름을 위해 hidden_states의 requires_grad 확인
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
