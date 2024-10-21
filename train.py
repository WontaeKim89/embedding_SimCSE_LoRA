from trainer import trainer
import gc
import os
from datetime import datetime
from safetensors import safe_open
from safetensors.torch import save_file
from dataloader import data_loader
from model import LoRA
import argparse
import torch
import wandb
from transformers import AutoModel, AutoTokenizer
from utils import fine_tuning_model_tokenizer_load, clear_memory

def main(args):
    # Args
    loss_temperature = getattr(args, 'loss_temperature', 0.08)
    model_name = getattr(args, 'model_name', 'BM-K/KoSimCSE-roberta-multitask')
    # model_name = getattr(args, 'model_name', 'jhgan/ko-sroberta-multitask')
    lr = getattr(args, 'lr', 1e-6)
    data_type = getattr(args, 'data_type', 0) # 0.동일문장 비교를 통한 Sim-CSE 학습방식 / 1.실제 유사문장 쌍이 있는 strong pair를 통해 학습할 경우
    test_set_shuffle = getattr(args, 'test_set_shuffle', True)
    num_workers = getattr(args, 'num_workers', 4)
    pin_memory = getattr(args, 'pin_memory', True)
    gpu_num = getattr(args, 'gpu_num', '6')
    batch_size = getattr(args, 'batch_size', 64)
    try:
        epochs = getattr(args, 'epochs', 10)
        fine_tune_model_yn = getattr(args, 'fine_tune_model_yn', '1')
        train_mode = getattr(args, 'train_mode', False)
    except:
        epochs = args.get('epochs', 10)
        fine_tune_model_yn = args.get('fine_tune_model_yn', '1')
        train_mode = args.get('train_mode', False)
    

    """
    Arguments Info Print
    """
    print(f'\n\n<Arguments>\n  - epochs : {epochs}\n  - fine_tune_model_yn : {fine_tune_model_yn}\n  - lr : {lr}\n  - data_type : {data_type}\n  - test_set_shuffle : {test_set_shuffle}\n  - num_workers : {num_workers}\n  - pin_memory : {pin_memory}\n  - gpu_num : {gpu_num}\n  - loss_temperature : {loss_temperature}\n\n')
    
    os.environ["TOKENIZERS_PARALLELISM"]="false"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    model_save_path = "/workspace/wontae_kim/RoBERTa_LoRA_SimCSE_FT/save_model"
    
    now_date = datetime.today().strftime("%Y%m%d")
    device = 'cuda:'+str(gpu_num) if torch.cuda.is_available() else 'cpu'
    print(f'GPU 사용 여부 : {device}')
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    origin_model = AutoModel.from_pretrained(model_name)
    
    if fine_tune_model_yn=='1':
        # fine_tune_model_name = '/workspace/wontae_kim/fine_tuning/save_model/20240919_SimCSE_Lora_Strong.safetensors'# 법령 데이터 Fine-Tuning 모델
        # fine_tune_model_name = "20241016_SimCSE_Lora.safetensors" # 법령 데이터 Fine-Tuning + 질문/API이름 Normal Fine-Tuning 모델
        # fine_tune_model_name = "20241017_SimCSE_Lora_epochs-2_lr-0.0001_batch-64_temperature-0.1_loss-0.57809.safetensors"
        fine_tune_model_name = "bk_20241018_normal_SimCSE_Lora_epochs-5_lr-0.0003_batch-16_temperature-0.1_loss-0.22338.safetensors"
        model = fine_tuning_model_tokenizer_load(os.path.join(model_save_path, fine_tune_model_name), origin_model, device)
        print(f'>>> Fine-Tuning Target Model : {fine_tune_model_name}')
        
    elif fine_tune_model_yn=='0' :
        model = LoRA(origin_model).to(device)
        print(f'>>> Fine-Tuning Target Model : {model_name}')
    else :
        print(f"'[ValueError] fine_tune_model_yn(Fine-Tuning 대상 모델 구분)에 적절한 옵션 값이 설정되지 않았습니다.(0.pre-trained origin model / 1.fine-tuning model)")
        raise ValueError

    model = model.to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-5)
    
    dataloader_config = {
        "data_type": data_type,
        "test_set_shuffle": test_set_shuffle if test_set_shuffle else True,
        "num_workers": num_workers if num_workers else 4,
        "pin_memory": pin_memory if pin_memory else True,
        "tokenizer": tokenizer if tokenizer else None,
        "batch_size": batch_size if batch_size else 64
    }
    
    if data_type==0:
        print(f'--------------[DataLoader] Normal DataLoader 생성')
        train_dataloader_A, valid_dataloader_A, test_dataloader_A = data_loader(**dataloader_config)
        train_dataloader_B, valid_dataloader_B, test_dataloader_B = None, None, None
    
    elif data_type==1:
        print(f'--------------[DataLoader] Strong DataLoader 생성')
        train_dataloader_A, valid_dataloader_A, test_dataloader_A, train_dataloader_B, valid_dataloader_B, test_dataloader_B = data_loader(**dataloader_config)

    trainer_config = {
        "model": model, 
        "train_dataloader_A": train_dataloader_A, 
        "valid_dataloader_A": valid_dataloader_A, 
        "train_dataloader_B": train_dataloader_B, 
        "valid_dataloader_B": valid_dataloader_B, 
        "optimizer": optimizer, 
        "device": device, 
        "epochs": epochs, 
        "loss_temperature": loss_temperature,
        "logging": False if train_mode else True #train mode와 반대(<-> Logging)
    }
    clear_memory() # Memory Initialization
    trainer(**trainer_config)


######################  Wandb Hyper Params Search  ###########################

def wandb_get_sweep_config():
    sweep_config = {
        'method': 'bayes',  # or 'random', 'grid', 'bayes'
        'metric': {
            'name': 'info_nce_loss',  # 기록되는 Loss 이름
            'goal': 'minimize'        # 최소화할지 최대화할지 설정
        },
        'parameters': {
            'learning_rate': {
                'values': [1e-6, 1e-5, 1e-4, 1e-3]
            },
            'batch_size': {
                'values': [32, 64, 128]
            },
            'temperature': {
                'values': [0.05, 0.1, 0.2, 0.5]
            },
        }
    }
    return sweep_config


def wandb_sweep_set(args):
    # 초기 parameter set
    config_defaults = {
        'learning_rate': 1e-4,
        'batch_size': 64,
        'temperature': 0.1,
        "epochs": 1,
        "fine_tune_model_yn": 1,
        "data_type": 0
    }
    # W&B 초기화 (기본값은 Sweep에서 덮어쓰기됨)
    wandb.init(config=config_defaults)
    config_params = {
        "learning_rate": wandb.config.learning_rate,
        "batch_size": wandb.config.batch_size,
        "loss_temperature": wandb.config.temperature,
        }
    args_dict = vars(args)
    additional_params = {k: v for k, v in args_dict.items() if k not in config_params}
    temp_params = {**config_params, **additional_params}
    # temp_params = {**config_params}
    print(f'\n\n<Test Hyper Params>\n{temp_params}\n\n')
    main(temp_params)


def wandb_tune(args):
    wandb_config = wandb_get_sweep_config()
    sweep_id = wandb.sweep(wandb_config, project="embed_fine_tuning")
    wandb.agent(sweep_id, function=lambda: wandb_sweep_set(args), count=10)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embedding Model Fine-Tuning Option")
    
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--model_name', type=str, default='BM-K/KoSimCSE-roberta-multitask', help="Pre-trained model name")
    # parser.add_argument('--model_name', type=str, default='jhgan/ko-sroberta-multitask', help="Pre-trained model name")
    parser.add_argument('--fine_tune_model_yn', default='1', help="0.fine-tuning되지않은 origin model로 fine-tuning 진행 / 1.fine-tuning되어있는 모델로 다시 fine-tuning 진행")
    parser.add_argument('--lr', type=float, default=1e-6, help="Learning rate for the optimizer, default 1e-6")
    parser.add_argument('--data_type', type=int, default=0, help="0.동일문장 비교를 통한 Sim-CSE 학습방식 / 1.실제 유사문장 쌍이 있는 strong pair를 통해 학습할 경우")
    parser.add_argument('--test_set_shuffle', type=bool, default=True, help="Test set 데이터 shuffling 여부")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for data loading")
    parser.add_argument('--pin_memory', type=bool, default=True, help="Pin memory for data loading")
    parser.add_argument('--loss_temperature', type=float, default=.1, help="loss_temperature, default 0.08")
    parser.add_argument('--gpu_num', type=str, default=6, help="gpu number to use")
    parser.add_argument('--batch_size', type=int, default=64, help="Dataloader batch size")
    parser.add_argument('--train_mode', type=str, default=True, help="True : Fine-Tuning 진행 / False : Wandb를 통한 Hyper-parameter Search 진행")

    args = parser.parse_args()
    print(f'Train_mode : {args.train_mode}')
    if str(args.train_mode)==str(True):
        print(f'\n\n>>>>>> Fine-Tuning 학습 모드를 시작합니다')
        main(args)
    elif str(args.train_mode)==str(False) :
        print(f'>>>>>> Wandb Hyper Params Searching 모드를 시작합니다')
        args.epochs = '1' # 여러번 테스트를 위해 1epoch씩만 검증
        wandb_tune(args)
    else :
        print(f'[ValueError] train_mode 옵션은 반드시 True 또는 False로만 입력가능합니다 - * True : Fine-Tuning 진행 / False : Wandb를 통한 Hyper-parameter Search 진행 ')
        raise ValueError
