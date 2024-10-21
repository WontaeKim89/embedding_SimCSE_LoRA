from torch.utils.data import Dataset
from utils import fisher_yates_shuffle
import pickle as pkl
import sys

sys.path.append('/workspace/wontae_kim/RoBERTa_LoRA_SimCSE_FT')
# NORMAL_DATASET_PATH = './test_dataset.pkl'
NORMAL_DATASET_PATH = './normal_sentences.pkl'
STRONG_DATASET_PATH = './strong_dataset.pkl'


def load_dataset(dataset_type):
    """
    * Data Description
        1) NORMAL_DATASET_PATH : ['sentence1', 'sentence12', 'sentence3', ...]
            - 단순 문장의 목록으로 구성
        2) STRONG_DATASET_PATH : (['sentence_A1', 'sentence_A2', 'sentence_A3', ...], ['sentence_B1', 'sentence_B2', 'sentence_B3', ...])
            - 문장 목록 A와 문장 목록 A와 '관계(positive pair)'가 있는 문장들의 목록 B, 두개의 목록을 Tuple로 묶은 데이터
            - 문장 목록 A의 각 인덱스의 문장은 B목록의 동일 인덱스의 문장과 positive pair 관계를 가지도록 구성한다.
    """
    if dataset_type==1:
        with open(STRONG_DATASET_PATH, 'rb') as f:
            strong_data = pkl.load(f)
            strong_dataset = tuple(map(list, zip(*[(sen1,sen2) for sen1, sen2 in zip(strong_data[0], strong_data[1]) if isinstance(sen1, str) and isinstance(sen2, str)])))
            strong_dataset_shuffle = fisher_yates_shuffle(strong_dataset[0], strong_dataset[1]) # 유사 데이터 밀집되지않도록 pair 유지한채로 shuffling
        return strong_dataset_shuffle
        
    elif dataset_type==0:
        with open(NORMAL_DATASET_PATH, 'rb') as f:
            normal_dataset = pkl.load(f)
            normal_dataset = [i for i in normal_dataset if isinstance(i, str)]
            normal_dataset = fisher_yates_shuffle(normal_dataset)
            # test_size = int(len(normal_dataset)*0.2)
        return normal_dataset
    else:
        raise ValueError(f"[ValueError] data_type 옵션의 값이 올바르지 않습니다. 허용되는 값: 0 또는 1, 입력된 값: {data_type}")


def dataset_divider(data_type, ratio):
    """
    train, valid, test set으로 분할
    """        
    if data_type==0:
        sentences = load_dataset(data_type)
        
        train_cnt = int(len(sentences)*ratio[0])
        valid_cnt = int(len(sentences)*ratio[1])
        test_cnt = len(sentences)-train_cnt-valid_cnt
    
        train_set = sentences[:train_cnt]
        valid_set = sentences[train_cnt:train_cnt+valid_cnt]
        test_set = sentences[train_cnt+valid_cnt:]
        return train_set, valid_set, test_set, [len(train_set), len(valid_set), len(test_set)]
    
    else:
        sentences = load_dataset(data_type)
        print(type(sentences), len(sentences))
        sentences_A, sentences_B = sentences[0], sentences[1]
        train_cnt = int(len(sentences_A)*ratio[0])
        valid_cnt = int(len(sentences_A)*ratio[1])
        test_cnt = len(sentences_A)-train_cnt-valid_cnt

        train_set_A = sentences_A[:train_cnt]
        valid_set_A = sentences_A[train_cnt:train_cnt+valid_cnt]
        test_set_A = sentences_A[train_cnt+valid_cnt:]

        train_set_B = sentences_B[:train_cnt]
        valid_set_B = sentences_B[train_cnt:train_cnt+valid_cnt]
        test_set_B = sentences_B[train_cnt+valid_cnt:]
        return train_set_A, valid_set_A, test_set_A, train_set_B, valid_set_B, test_set_B, [train_cnt, valid_cnt, test_cnt]


class SimCSEDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data  # 데이터셋 리스트
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data[idx]  # 하나의 배치 데이터를 선택
        inputs = self.tokenizer(
            sentence,
            max_length=self.max_length,
            padding='max_length',  # 모든 데이터를 max_length로 패딩
            truncation=True,       # max_length로 자름
            return_tensors='pt'    # tensor로 변환
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),  # 배치 차원을 제거
            'attention_mask': inputs['attention_mask'].squeeze(0)
        }


def dataset_generator(tokenizer, data_type=0, ratio:list=[.6, .2, .2]):
    if sum(ratio)!=1:
        raise ValueError(f"Train / Valid / Test Data set의 분할 비율이 올바르지 않습니다. -> {ratio}")
    if data_type not in [0,1]:
        raise ValueError(f"[ValueError] data_type 옵션의 값이 올바르지 않습니다. 허용되는 값: 0 또는 1, 입력된 값: {data_type}")

    if data_type==0:
        train_set, valid_set, test_set, _ = dataset_divider(data_type, ratio)
        train_dataset = SimCSEDataset(train_set, tokenizer)
        valid_dataset = SimCSEDataset(valid_set, tokenizer)
        test_dataset = SimCSEDataset(test_set, tokenizer)
        return train_dataset, valid_dataset, test_dataset

    else:
        train_set_A, valid_set_A, test_set_A, train_set_B, valid_set_B, test_set_B, _ = dataset_divider(data_type, ratio)
        train_dataset_A = SimCSEDataset(train_set_A, tokenizer)
        valid_dataset_A = SimCSEDataset(valid_set_A, tokenizer)
        test_dataset_A = SimCSEDataset(test_set_A, tokenizer)

        train_dataset_B = SimCSEDataset(train_set_B, tokenizer)
        valid_dataset_B = SimCSEDataset(valid_set_B, tokenizer)
        test_dataset_B = SimCSEDataset(test_set_B, tokenizer)
        return train_dataset_A, valid_dataset_A, test_dataset_A, train_dataset_B, valid_dataset_B, test_dataset_B
    