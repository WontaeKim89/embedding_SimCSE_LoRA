<h1 align="center"> <p>🤗 LoRA와 SimCSE를 이용한<br>한국어 임베딩 모델 Fine-Tuning</p></h1>
<h3 align="center">
    <p>Korean Embedding Model Fine-Tuning (with LoRA+SimCSE)</p>
</h3>

최근 언어모델의 Fine-Tuning에서 적극적으로 활용되는 LoRA는, <br>
PEFT 라이브러리의 'Lora_config'클래스를 통해 손쉽게 fine-tuning 설정이 가능합니다.<br>
하지만, 임베딩 모델을 contrastive learning의 대표적인 학습방식인 SimCSE 매서드를 통해 <br>데이터셋을 구성하고 학습하는 부분에 있어서는
Lora_config 클래스에서 구체적인 옵션 적용을 지원하지 않습니다.

이와 같은 불편함을 해소하고자, 임베딩 모델을 SimCSE 매서드를 통해 학습할때,<br>
LoRA 방식을 적용하여 Fine-Tuning 할 수 있도록 학습 코드를 구현하였습니다.

경로내에 정해진 포멧으로 학습 데이터를 구성하고 train.py 파일에 정의된 argument를 입력하여 실행하면<br>
간편하게 LoRA를 적용한 SimCSE 방식의 Embedding Model Fine-Tuning이 가능합니다.


## Quickstart

### Training Dataset setting<br>
- 학습을 시작하기 전, 해당 프로젝트의 경로에 아래 데이터 형식을 가진 <br> normal_sentences.pkl 또는 strong_dataset.pkl 파일이 셋팅되어 있어야 합니다.<br>

| 데이터 구분      | type                  | e.g.                                                                                                                                                                                      |
|------------------|-----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Normal Dataset** | List                  | ['오늘 날씨 어때?', '지금 몇 시야?', '근처 카페 추천해 줘.', '오늘 일정 확인해 줘.']                                                                                                                                 |
| **Strong Dataset** | a list of tuples      | 1) Pairs A: ['오늘 영화 추천해 줘.', '가장 인기 있는 축구선수가 누구야?', '영어를 잘하는 법은?'] <br><br> 2) Pairs B: ['오늘 같이 더운 여름에는 타이타닉이 좋을 것 같아요', '호날두와 메시가 전세계적으로 많은 인기를 누리고 있어요.', '우선 영어 단어를 충분히 숙지하는게 우선이에요.'] |
- ### train.py 실행 시, 'data_type'을 '0' 으로 설정할 경우?
  - normal dataset을 기준으로 동일문장을 한쌍으로 묶은것을 positive pair,<br> 다른문장을 한쌍으로 묶은것을 negative pair로 간주하여 <br>SimCSE 방식의 학습을 진행합니다.
- ### train.py 실행 시, 'data_type'을 '1' 로 설정할 경우?
  - strong dataset을 기준으로, 두개의 list의 동일 인덱스에 위치한 문장을 positive pair로 간주하여 학습을 진행합니다.


### Run training

```bash
python3 train.py --data_type 0 --fine-tune_model_yn 0
```

## Arguments

- ### --epochs 
  - Epoch 횟수를 설정합니다.
  - Default : 10
- ### --model_name
  - fine-tuning을 진행할 HuggingFace의 backbone model(pre-trained model)<br>이름을 입력합니다.
  - Default : "BM-K/KoSimCSE-roberta-multitask"
- ### --fine_tune_model_yn
  - 0 : backbone model을 load하여 최초 fine-tuning을 진행할 경우
  - 1 : LoRA를 통해서 이미 fine-tuning한 모델을 load해서 추가 fine-tuning할 경우
  - Default : 0
- ### --lr
  - Learning_rate 
  - Default : 1e-6
- ### --data_type
  - 0 : normal dataset을 통해 unsupervised learning 진행
  - 1 : strong dataset을 통해 supervised learning 진행
- ### --test_set_shuffle
  - test dataset을 shuffle 여부 (0.부, 1:여)
  - Default : 0
- ### --loss_temperature
  - loss를 통한 backpropagation 진행 시, weight update 반영 수준 설정
  - Default : 0.1
- ### --batch_size
  - batch_size 설정
  - Default : 64
- ### --train_mode
  - True : fine-tuning 진행
  - False : Wandb를 통한 hyperparameter logging 진행(model save 하지않음)
  - Default : True