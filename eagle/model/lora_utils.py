import torch
import torch.nn as nn

# EAGLE 모델을 위한 커스텀 PeftModel 클래스 정의
class CustomPeftModelForFeatureExtraction(nn.Module):
    def __init__(self, model, peft_config):
        super().__init__()
        self.base_model = model
        self.peft_config = peft_config
        # 필요한 메서드와 속성 복사
        self.print_trainable_parameters = getattr(model, 'print_trainable_parameters', None)
        
        # init_tree와 기타 메서드 설정
        if hasattr(model, 'base_model'):
            self.base_tree = model.base_model
            self.init_tree = self.base_tree.init_tree if hasattr(self.base_tree, 'init_tree') else None
            self.topK_genrate = self.base_tree.topK_genrate if hasattr(self.base_tree, 'topK_genrate') else None
            self.reset_kv = self.base_tree.reset_kv if hasattr(self.base_tree, 'reset_kv') else None
            self.reset = self.base_tree.reset if hasattr(self.base_tree, 'reset') else None
        else:
            self.base_tree = model
            self.init_tree = model.init_tree if hasattr(model, 'init_tree') else None
            self.topK_genrate = model.topK_genrate if hasattr(model, 'topK_genrate') else None
            self.reset_kv = model.reset_kv if hasattr(model, 'reset_kv') else None
            self.reset = model.reset if hasattr(model, 'reset') else None
    
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
        # 그래디언트 흐름을 위해 hidden_states의 requires_grad 확인
        if hidden_states is not None and not hidden_states.requires_grad:
            hidden_states = hidden_states.detach().requires_grad_(True)
        
        # base_model의 forward 직접 호출 - 이미 LoRA가 적용된 레이어들이 포함됨
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
        
        # 그래디언트 흐름 유지를 위해 출력 텐서의 requires_grad 확인
        if use_cache and isinstance(outputs, tuple):
            hidden_out, cache = outputs
            if not hidden_out.requires_grad:
                hidden_out = hidden_out.detach().requires_grad_(True)
            return hidden_out, cache
        
        if not outputs.requires_grad:
            outputs = outputs.detach().requires_grad_(True)
        
        return outputs

# 추론 시 requires_grad 설정을 건너뛰는 버전의 클래스
class InferenceCustomPeftModelForFeatureExtraction(CustomPeftModelForFeatureExtraction):
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
        # 추론 모드에서는 requires_grad 설정 없이 바로 forward 호출
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
        
        return outputs
