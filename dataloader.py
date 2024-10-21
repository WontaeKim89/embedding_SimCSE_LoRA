import sys
sys.path.append('/workspace/wontae_kim/RoBERTa_LoRA_SimCSE_FT')
from transformers import AutoModel, AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from dataset import dataset_divider, dataset_generator

model_name = 'BM-K/KoSimCSE-roberta-multitask'
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 자동 패딩 처리
data_collator = DataCollatorWithPadding(tokenizer)

def data_loader(data_type, batch_size=64, test_set_shuffle=True, num_workers=4, pin_memory=True, collate_fn=data_collator, tokenizer=tokenizer):
    """
    Args.
        data_type : 0.일반적인 Sim-CSE 데이터 구조를 의미하며, pair구조가 아닌, 문장의 목록 형태의 데이터를 의미한다.
    """
    if data_type==0:
        common_params = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "collate_fn": collate_fn
        }
        train_dataset, valid_dataset, test_dataset = dataset_generator(tokenizer=tokenizer, data_type=data_type)
        train_dataloader = DataLoader(train_dataset, shuffle=True, **common_params)
        valid_dataloader = DataLoader(valid_dataset, shuffle=True, **common_params)
        test_dataloader = DataLoader(test_dataset, shuffle=test_set_shuffle, **common_params)
        
        return train_dataloader, valid_dataloader, test_dataloader

    else:
        train_dataset_A, valid_dataset_A, test_dataset_A, train_dataset_B, valid_dataset_B, test_dataset_B = dataset_generator(tokenizer=tokenizer, data_type=data_type)
        common_params = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "collate_fn": collate_fn
        }
        train_dataloader_A = DataLoader(train_dataset_A, shuffle=True, **common_params)
        valid_dataloader_A = DataLoader(valid_dataset_A, shuffle=True, **common_params)
        test_dataloader_A = DataLoader(test_dataset_A, shuffle=test_set_shuffle, **common_params)

        train_dataloader_B = DataLoader(train_dataset_B, shuffle=True, **common_params)
        valid_dataloader_B = DataLoader(valid_dataset_B, shuffle=True, **common_params)
        test_dataloader_B = DataLoader(test_dataset_B, shuffle=test_set_shuffle, **common_params)

        return train_dataloader_A, valid_dataloader_A, test_dataloader_A, train_dataloader_B, valid_dataloader_B, test_dataloader_B
        