# Learning Keras

> 수업 내용 중 정리 필요한 내용 추가로 정리하고 머리속에 넣자



## 파이썬 IDE 구축하기 Visual Studio Code

Visual Studio Code를 설치하고, 이하 **5개**(`python` , `python for vscode` , `python Extension pack`, `python extended`,  `python indent`)를 설치하자. 왼쪽 하단에 Python의 버전을 확인할 수 있다.



## 유용한 단축키들

- 출력 : `ctrl` + `F5`
- 주석 달기(#) : `ctrl` + `/`



## Keras 를 활용하여 딥러닝의 기본 모델 구축하기

1. *데이터를 불러온다.*
2. *모델을 구성한다.*
3. *데이터를 학습시킨다.(compile과정과 fitting)*
4. *평가(evaluate)와 예측(predict)을 한다.*



```python
#1. 데이터
import numpy as np
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])
```

> 데이터를 numpy 배열로 만들어준다.

- `print(x.shape, y.shape)`을 입력하게 되면, `(10, ), (10, )` 이 출력된다.



```python
#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(16, input_dim = 1))
model.add(Dense(32))
model.add(Dense(1))
```

> Sequential은 내가 코딩한 순서대로 모델을 구성하는 것이고, 함수형 모델도 존재한다. 위의 모델은 input layer 포함 4-layers를 가지고 있다. dim=1 input 데이터의 shape에 따른다.



```python
#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x, y, epochs=100, batch_size=1)
```

> 모델을 훈련시킬 때, 우리가 원하는 대로 튜닝해야할 하이퍼 파라미터들이 많이 있다.

- `Lose Function`, `Optimizer`, `epoch`, `batch_size` 에 대한 내용은 TIL 레포지토리의 [Deep Learning]으로 가서 보기

- *간략하게 여기서 정리하면,* 

  lose Function은 우리가 사용하는 raw - data를 봤을 때 그 데이터가 나타내는 특징과 멀어지는 것은 손실 또는 에러라고 가정하고, 이 손실을 최대한 줄여주는 곳으로 특징(선)을 뽑아내기 위해 최적화 모듈(Optimizer)을 사용한다. 

- metrics는 우리가 어떤 기준으로 볼 것이냐에 대한 내용이다. (metrics=['mse']이면 mse 기준으로, metrics=['acc']이면 정확도 기준으로.)

- ```shell
  mse와 mae의 차이 ....
  1. mse == 실제값과 예측값 차이의 제곱
  2. rmse ==mse 결과값의 루트를 사용해서 값을 낮춰줌 
  3. mae == 차이의 절대값
  4. rmae == mae 결과값의 루트를 사용해서 값을 낮춰줌
  ```



```python
#4. 평가예측
loss, mse = model.evaluate(x, y, batch_size=1)
print('mse: ', mse)

x_prd = np.array([11,12,13])
result = model.predict(x_prd, batch_size=1)
print(result)
```

> evaluate과 predict 를 사용해서 평가와 예측을 했다.



```python
model.summary()

-----------------------------------------------------------------[output]

Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
==========================================
dense_1 (Dense)              (None, 5)                 10
_________________________________________________________________
dense_2 (Dense)              (None, 2)                 12
_________________________________________________________________
dense_3 (Dense)              (None, 3)                 9
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 4
==========================================
Total params: 35
Trainable params: 35
Non-trainable params: 0
_________________________________________________________________

```

**Output shape:**

dense_1 에서 Output shape를 살펴보면 `(None, 5)` 이라고 나오는데 우선, Output shape는 행렬 데이터이다. `(None, 5)`는 행이 없는 5개의 1차원 배열 데이터를 뜻한다.



![](data:image/png;base64,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)

- **Param # :**

  `y = Wx +b` 라는 선형 함수에서 W(weight)와 b(bias) 두 개의 파라미터를 가지고 있다.

  - 딥러닝 모델은 `y = Wx +b` 의 모델로 학습을 하는데, x와 y는 사람이 직접 정형 데이터를 Input하는 파라미터 이다. 여기서 컴퓨터는 우리가 Input한 모델에 맞춰서 w(가중치)와 b(편향)을 계산한다.
  - 상단의 이미지 처럼 편향은 각 층마다 하나씩 존재하고 있다고 상상하자. (예를 들면 Input data는 1개이고 다음 Dense1은 5개의 노드를 가지고 있다. Input data에 편향이라는 하나의 노드가 같이 학습된다.)

- **`y = Wx +b`  :**

![](https://blobscdn.gitbook.com/v0/b/gitbook-28427.appspot.com/o/assets%2F-LbBOSivlH5hwcpk3QX6%2F-Lbzu6Iu9wUpJCeiXXB3%2F-LbzuhhVovWCQBOVAAqC%2F53402.png?alt=media&token=38d4eeb8-0d0d-4e3f-83d5-bdfcb76d6191)

> 직선과의 거리 분포가 가장 적은 곳에 선을 긋는다.

x와 y는 내가 입력한 데이터에 따라 달라지며, w와 b는 Hidden layer 에서 학습하면서 조정된다. 



![](http://cfile210.uf.daum.net/R400x0/995837405A537CE524917C)

하지만 위와 같이 선형 모양의 데이터 분포를 보이지 않는 데이터들도 많을 것이다. 이럴 때 컴퓨터는 어떻게 선형 함수로 학습하는 것인가?



<img src="https://i.ytimg.com/vi/MgnjjN26C54/maxresdefault.jpg" style="zoom:50%;" />

> 2차 함수, 3차 함수 형태 혹은 더 복잡한 형태의 데이터 이더라도 값의 미분을 통해서 선형적 특징을 찾아내 분류 한다.

