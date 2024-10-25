from datetime import datetime
import numpy as np
from tqdm import tqdm
import torch
from utils import info_nce_loss, clear_memory
from safetensors.torch import save_file
import wandb

def train_epoch(model, dataloader_A, optimizer, device, loss_temperature, dataloader_B=None, logging=False):
    model.train()
    best_loss = np.inf
    early_stop_limit = 100
    early_stop_num = 0
    total_loss = 0
    deducted_cnt = 0 # loss가 NaN일 경우 마지막 평균 loss 계산때, 카운트에서 차감하기 위함

    if not dataloader_B:
        iterator = tqdm(dataloader_A)
        is_dual = False
        print(f">> Data Type : 'Single(Not Dual/Pair)'")

    else :
        iterator = tqdm(zip(dataloader_A, dataloader_B))
        is_dual = True
        print(f">> Data Type : 'Dual(Pair)'")
    
    desc = "Strong Fine-Tuning" if is_dual else "Normal Fine-Tuning"

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    for cnt, data in enumerate(iterator):
        optimizer.zero_grad()
           
        if not is_dual:
            batch = data
            input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            outputs1 = model(input_ids, attention_mask=attention_mask, return_dict=True)
            hidden_states1 = outputs1.last_hidden_state
            embeddings_1 = hidden_states1[:, 0, :].to(device)

            outputs2 = model(input_ids, attention_mask=attention_mask, return_dict=True)
            hidden_states2 = outputs2.last_hidden_state
            embeddings_2 = hidden_states2[:, 0, :].to(device)
        
        else:
            batch_A, batch_B = data
            # 문장 A와 B의 입력 및 attention mask
            input_ids_A, attention_mask_A = batch_A['input_ids'].to(device), batch_A['attention_mask'].to(device)
            input_ids_B, attention_mask_B = batch_B['input_ids'].to(device), batch_B['attention_mask'].to(device)
            # 드롭아웃 변형을 통해 문장 A와 B의 임베딩 생성
            output_A = model(input_ids_A, attention_mask=attention_mask_A, return_dict=True)
            hidden_states_A = output_A.last_hidden_state # batch x sequence_length x hidden_size
            output_B = model(input_ids_B, attention_mask=attention_mask_B, return_dict=True)
            hidden_states_B = output_B.last_hidden_state

            embeddings_1 = hidden_states_A[:, 0, :].to(device)
            embeddings_2 = hidden_states_B[:, 0, :].to(device)

        loss = info_nce_loss(embeddings_1, embeddings_2, loss_temperature, device)
        if loss<best_loss:
            best_loss = loss
            print(f'>>>>>>>>>>>>>>>> Best Loss Update : {best_loss} <<<<<<<<<<<<<<<<<<!!!')
            early_stop_num=0
        
        early_stop_num+=1
        if early_stop_num>=early_stop_limit:
            print(f'>>>>>>>>>>>>>>>> Early_Stop <<<<<<<<<<<<<<<<<<!!!')
            break
            
        if logging:
            wandb.log({'Nce_loss[Train]': loss.item()})
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if np.isnan(loss.item()):
            deducted_cnt+=1
            print(f'>> {desc}... Train Loss : {"!! Loss NaN 발생 !!"} -> 현재 Epoch에서 NaN 발생횟수 : {deducted_cnt}')
            continue
        else :
            total_loss += loss.item()
        print(f'>> {desc}... Train Loss : {loss.item():.8f}')
    
    avg_loss = total_loss / ((cnt+1)-deducted_cnt)
    if logging:
        wandb.log({'EPOCH-Nce_loss[Train]': avg_loss})
    return avg_loss


def evaluate_epoch(model, dataloader_A, optimizer, device, loss_temperature, dataloader_B=None, logging=False):
    model.eval()
    total_loss = 0
    deducted_cnt = 0

    if not dataloader_B:
        iterator = tqdm(dataloader_A)
        is_dual = False

    else: 
        iterator = tqdm(zip(dataloader_A, dataloader_B))
        is_dual = True

    desc = "Strong Fine-Tuning" if is_dual else "Normal Fine-Tuning"
    with torch.no_grad():
        for data in tqdm(iterator):
            if not is_dual:
                batch = data
                input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
                output = model(input_ids, attention_mask=attention_mask, return_dict=True)
                hidden_states = output.last_hidden_state
                embeddings_1 = hidden_states[:, 0, :].to(device)
                embeddings_2 = hidden_states[:, 0, :].to(device)
                """수정코드 종료"""
            else :
                batch_A, batch_B = data
                input_ids_A, attention_mask_A = batch_A['input_ids'].to(device), batch_A['attention_mask'].to(device)
                input_ids_B, attention_mask_B = batch_B['input_ids'].to(device), batch_B['attention_mask'].to(device)

                output_A = model(input_ids_A, attention_mask=attention_mask_A, return_dict=True)
                output_B = model(input_ids_B, attention_mask=attention_mask_B, return_dict=True)
                hidden_states_A = output_A.last_hidden_state 
                hidden_states_B = output_B.last_hidden_state

                embeddings_1 = hidden_states_A[:, 0, :].to(device)
                embeddings_2 = hidden_states_B[:, 0, :].to(device)
                """수정코드 종료"""

            loss = info_nce_loss(embeddings_1, embeddings_2, loss_temperature, device)
            if logging:
                wandb.log({'Nce_loss[Eval]': loss.item()})
            
            if np.isnan(loss.item()):
                deducted_cnt+=1
                continue
            else :
                total_loss += loss.item()
            print(f'>>  {desc}... Valid Loss : {loss.item():.8f}')

    avg_loss = total_loss / (len(dataloader_A)-deducted_cnt)
    if logging:
        wandb.log({'EPOCH-Nce_loss[Eval]': avg_loss})
    return avg_loss


def trainer(model, train_dataloader_A, valid_dataloader_A, optimizer, device, epochs, loss_temperature, train_dataloader_B=None, valid_dataloader_B=None, logging=False):
    best_loss = np.inf
    best_model = None
    now_date = datetime.today().strftime("%Y%m%d")
    _lr = optimizer.param_groups[0]['lr']
    _batch_size = train_dataloader_A.batch_size
    
    for epoch in range(int(epochs)):
        print(f'Epoch {epoch + 1}/{epochs}')
        train_config = {
            "model": model,
            "dataloader_A": train_dataloader_A,
            "dataloader_B": train_dataloader_B,
            "optimizer": optimizer,
            "device": device,
            "loss_temperature": loss_temperature,
            "logging": logging #wandb logging 여부
        }                
 
        print(f'============ Epoch_{epoch+1} Training Start ============')
        train_loss = train_epoch(**train_config)
        print(f'====== Epoch_{epoch+1} Training Complete >>> Train Loss: {train_loss:.8f} ======')

        valid_config = {
            "model": model,
            "dataloader_A": valid_dataloader_A,
            "dataloader_B": valid_dataloader_B,
            "optimizer": optimizer,
            "device": device,
            "loss_temperature": loss_temperature,
            "logging": logging #wandb logging 여부
        }       
        print(f'============ Epoch_{epoch+1} Validation Start ============')
        valid_loss = evaluate_epoch(**valid_config)
        print(f'====== Epoch_{epoch+1} Validation Complete >>> Valid Loss: {valid_loss:.8f} ======')

        if best_loss>valid_loss:
            print(f'!!!! BEST LOSS 갱신 : {best_loss:.8f} -> {valid_loss:.8f}')
            best_loss = valid_loss
            best_model_state_dict = model.state_dict()
        
        clear_memory()

    fine_tune_method = 'strong' if train_dataloader_B else 'normal'
    save_file(best_model_state_dict, f'/workspace/wontae_kim/RoBERTa_LoRA_SimCSE_FT/save_model/{now_date}_{fine_tune_method}_SimCSE_Lora_epochs-{epochs}_lr-{_lr}_batch-{_batch_size}_temperature-{loss_temperature}_loss-{best_loss:.5f}.safetensors')
    print(f'===========학습이 종료되었습니다============')
    print(f'>> Best Loss : {best_loss:8f}')