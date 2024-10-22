import torch
from torch import nn
from transformers import RobertaModel, RobertaTokenizer
from transformers.models.roberta.modeling_roberta import RobertaSelfAttention


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

