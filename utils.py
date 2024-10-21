from model import LoRA
import torch.nn as nn
import gc
import random
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from safetensors.torch import load_file
import torch


def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()


def info_nce_loss(embeddings_1, embeddings_2, temperature, device):
    # Cosine similarity matrix 계산 (유사도 계산 결과는 batch_size x batch_size 형태가 되어야함)
    sim_matrix = F.cosine_similarity(embeddings_1.unsqueeze(1), embeddings_2.unsqueeze(0), dim=-1) / temperature

    # 같은 문장은 diagonal에 위치함
    batch_size = sim_matrix.size(0)
    labels = torch.arange(batch_size).to(device)  # 같은 문장 쌍의 인덱스가 맞춰져 있음 (diagonal)

    # CrossEntropyLoss로 InfoNCE Loss 계산
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(sim_matrix, labels)
    return loss


def empty_checker(value): #null이면 False를 반환
    none_keyword = ['Nan', 'nan', 'NaN', 'None', 'NAN', 'none']
    try:
        if isinstance(value, list):
            if len(value)==0:
                return False
            else :
                return True
        if isinstance(value, str) and len(value)==0:
            return False
        if not isinstance(value, str) and value is None:
            return False
        if not isinstance(value, str) and math.isnan(value):
            return False
        if isinstance(value, str) and value in none_keyword:
            return False
        return True
    except Exception as e:
        print(f'{e} : value : {value}')


def empty_deleter(values:list):
    none_keyword = ['Nan', 'nan', 'NaN', 'None', 'NAN', 'none']
    result = ''
    for val in values:
        if not isinstance(val, str) and val is None:
            result+= ''
        elif not isinstance(val, str) and math.isnan(val):
            result+= ''
        elif isinstance(val, str) and val in none_keyword:
            result+= ''
        else:
            result+= val
    return result


def origin_model_tokenizer_load(device, model_name='BM-K/KoSimCSE-roberta-multitask'):
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model.to(device), tokenizer


def fine_tuning_model_tokenizer_load(saved_model_path, origin_model, device):
    lora_model = LoRA(origin_model).to(device)
    
    lora_load_state_dict = load_file(saved_model_path)
    lora_model.load_state_dict(lora_load_state_dict, strict=False) # strict=False로 설정하여 일부 가중치만 업데이트

    lora_model.to(device)
    lora_model.eval()
    return lora_model


def get_embed(inputs:list, tokeinzer, model):
    """
    Args.
        inputs(list) : ["sen1", "sen2" ...]
    """
    sen_token = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
    sen_token = {key: value.to(device) for key, value in sen_token.items()}

    if isinstance(model, RobertaModel):
        print(f'....Embedding through "RoBERTa" >>>>')
        embeddings, sen_embed = model(**sen_token, return_dict=False)
    elif isinstance(model, LoRA):
        print(f'....Embedding through "LoRA" >>>>')
        for k, v in sen_token.items():
            print(f'{k} / {v.size()}')
        embeddings = model(**sen_token, return_dict=False) # 리스트내부에 "(max_length, 768) x 문장개수" 로 구성되어있음
        sen_embed = [i[0,:].unsqueeze(0) for i in embeddings] # 리스트내 vector의 cls토큰에 해당하는 첫번째(0)인덱스의 값을 가져오되, (1,768)shape을 가지도록 unsqueeze
        sen_embed = torch.cat(sen_embed, dim=0) # (문장개수 x 768)형태의 데이터로, RoBERTa 모델의 데이터 형태와 동일하게 처리
    else :
        print('예상가능한 모델 형태가 아닙니다. (RoBERTa 또는 LoRA 모델만 사용가능)')
        return False
    return sen_embed


def get_cosine_similarity_score(embed1, embed2):
    if len(embed1.shape) == 1: embed1 = embed1.unsqueeze(0)
    if len(embed2.shape) == 1: embed2 = embed2.unsqueeze(0)

    a_norm = embed1 / embed1.norm(dim=1)[:, None]
    b_norm = embed2 / embed2.norm(dim=1)[:, None]
    return torch.mm(a_norm, b_norm.transpose(0, 1)) * 100


def fisher_yates_shuffle(arr_origin, arr2_origin=None):
    # 원본유지
    arr = arr_origin[:]
    arr2 = arr2_origin[:] if arr2_origin else None
    
    if arr2:
        if len(arr)!=len(arr2):
            raise ValueError(f"함수에 인자로 입력된 두개 목록의 길이가 상이합니다.{len(arr)}/{len(arr2)}")
        n = len(arr)
        for i in range(n-1, 0, -1):
            j = random.randint(0, i)
            arr[i], arr[j] = arr[j], arr[i]
            arr2[i], arr2[j] = arr2[j], arr2[i]
        return (list(arr), list(arr2))
    else :
        n = len(arr)
        for i in range(n-1, 0, -1):
            j = random.randint(0, i)
            arr[i], arr[j] = arr[j], arr[i]
        return arr
        