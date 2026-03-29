#Adversarial Attacks

이 과제는 CNN 모델을 MNIST 와 CIFAR-10 데이터셋으로 학습시키고, 적대적 공격 기법을 적용하여 모델의 취약성을 보는 코드입니다. 


#공격 기법

Untargeted FGSM
Targeted FGSM
Untargeted PGD
Targeted PGD 


#환경 설정
Python 3 환경에서 작성되었으며 필요한 라이브러리를 설치해야 합니다.


pip install -r requirements.txt





#실행 방법 

터미널에서 아래 명령어로 파이썬 스크립트를 실행합니다.


python test.py





#코드 실행 

1. MNIST와 CIFAR-10 데이터를 자동으로 다운로드합니다.
2. 각 데이터셋에 대해 SimpleCNN 모델을 2 Epoch 동안 학습합니다.
3. 학습된 모델을 대상으로 다음 4가지 공격을 수행합니다.
-Untargeted FGSM
-Targeted FGSM
-Untargeted PGD
-Targeted PGD



#결과

공격의 성공률을 확인합니다.
results폴더에 공격으로 생성된 이미지 샘플이 저장됩니다.



#Acknowledge
과제 구현 과정에서 OpenAI의 도움을 받았음 밝힙니다.
