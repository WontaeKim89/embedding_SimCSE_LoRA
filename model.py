import torch
from torch import nn
from transformers import RobertaModel, RobertaTokenizer
from transformers.models.roberta.modeling_roberta import RobertaSelfAttention


# class LoRA(nn.Module):
#     def __init__(self, model, rank=8, dropout_rate=0.1):
#         super(LoRA, self).__init__()
#         self.model = model
#         self.rank = rank

#         # LoRA 적용을 위한 저차원 행렬 정의
#         self.lora_q = nn.Linear(model.config.hidden_size, rank, bias=False)
#         self.lora_k = nn.Linear(rank, model.config.hidden_size, bias=False)

#         # 드롭아웃 레이어 추가
#         self.dropout = nn.Dropout(p=dropout_rate)

#         # 저차원 행렬 가중치 초기화
#         nn.init.normal_(self.lora_q.weight, std=model.config.initializer_range)
#         nn.init.zeros_(self.lora_k.weight)

#     def forward(self, input_ids, attention_mask, **kwargs):
        
#         # 모델에 입력할 데이터의 차원을 [batch_size, sequence_length]로 맞출것
#         if input_ids.ndimension() == 3:
#             input_ids = input_ids.squeeze(0)
#         if input_ids.ndimension() == 3:
#             attention_mask = attention_mask.squeeze(0)
#         outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
#         hidden_states = outputs.last_hidden_state
#         # LoRA 적용: 저차원 행렬을 사용하여 attention 수정
#         lora_q_output = self.lora_q(hidden_states)
#         lora_k_output = self.lora_k(lora_q_output)
#         # 드롭아웃 적용
#         lora_k_output = self.dropout(lora_k_output)

#         # LoRA 결과를 원본 히든 상태에 추가
#         adapted_hidden_states = hidden_states + lora_k_output
#         return adapted_hidden_states




class LoRA(nn.Module):
    def __init__(self, model, r=8, alpha=16, dropout_rate=0.1):
        super(LoRA, self).__init__()
        self.model = model
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout_rate = dropout_rate  # 드롭아웃 비율 추가

        # 원본 모델 가중치 고정
        for param in self.model.parameters():
            param.requires_grad = False

        # LoRA 레이어를 어텐션 프로젝션에 주입
        for name, module in self.model.named_modules():
            if isinstance(module, RobertaSelfAttention):
                # 쿼리, 키, 밸류 프로젝션 수정
                self._inject_lora(module)

    def _inject_lora(self, module):
        for proj in ["query", "key", "value"]:
            weight = getattr(module, proj).weight
            bias = getattr(module, proj).bias
            out_features, in_features = weight.shape
    
            # LoRA 레이어 생성
            lora_A = nn.Linear(in_features, self.r, bias=False)
            lora_B = nn.Linear(self.r, out_features, bias=False)
            nn.init.normal_(lora_A.weight, std=0.02)
            nn.init.zeros_(lora_B.weight)
    
            # LoRAProjection에 드롭아웃 비율 전달
            lora_proj = LoRAProjection(weight, bias, lora_A, lora_B, self.scaling, dropout_rate=self.dropout_rate)
            setattr(module, proj, lora_proj)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class LoRAProjection(nn.Module):
    def __init__(self, weight, bias, lora_A, lora_B, scaling, dropout_rate=0.0):
        super(LoRAProjection, self).__init__()
        self.weight = weight
        self.bias = bias
        self.lora_A = lora_A
        self.lora_B = lora_B
        self.scaling = scaling

        # 드롭아웃 레이어 추가
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(p=dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        # LoRA 부분 계산
        lora_out = self.lora_A(x)
        if self.dropout is not None:
            lora_out = self.dropout(lora_out)
        lora_out = self.lora_B(lora_out)

        # 원본 출력과 LoRA 보정값의 합산
        return nn.functional.linear(x, self.weight, self.bias) + self.scaling * lora_out

