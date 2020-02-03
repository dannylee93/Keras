# Learning Keras

> 수업 내용 중 정리 필요한 내용 추가로 정리하고 머리속에 넣자



1. **RNN**

   > RNN(Recurrent Neural Network)에 대한 내용은 내 TIL 레포지토리를 참고하자.

2. **Keras 모델로 연습하는 LSTM**

   데이터 구조 맞추기,  LSTM 모델 Ensemble, 하이퍼 파라미터 참고



## RNN

RNN의 핵심은 이전 퍼셉트론의 Output을 기억하는 Hidden State를 가지고 있다는 것이다.
책에서는 **메모리 셀**이라고 한다. 사람이 이전에 기억해놨던 단어를 바탕으로 새로운 단어를 이해하는 것 처럼 진행 과정마다 계속 반복하기 때문에 '순환'이라는 의미가 붙는다.



#### RNN의 데이터 구조

RNN과 Keras 모델을 통해서 코드를 구성할 때, 가장 중요한 것이 데이터의 텐서 형태와 내가 만든 모델로 진입할때 Input Shape와 Output layer 의 shape를 맞춰주는 것이다.  

Shape를 잘 맞춰줄 수 있다는 것은, RNN 모델의 알고리즘을 이해하고 있을 뿐 아니라 텐서 형태에 대한 이해도 있다는 것이므로 데이터의 구조를 잘 다룰 수 있어야 하는 것이 제일 중요하다 볼 수 있겠다.

| 100  | 110  |  90  |  30  |  70  | '''  | ??(예측) |
| :--: | :--: | :--: | :--: | :--: | :--: | :------: |
|  X   |  Y   |      |      |      |      |          |
|      |  X   |  Y   |      |      |      |          |
|  X   |  X   |  X   |  X   |      |      |          |
|      |      |      |      |  Y   |      |          |

> 1, 2행은 각각 1일치 X데이터로 Y 데이터를 예측한다는 뜻이고, 3,4행은 4일치 데이터를 쌓아두고 다음 일 수의 1일치 Y데이터를 예측한다는 뜻이다. (y값은 내가 만들어가는 값)



```shell
    x       y
[1,2,3,4,5] 5
[2,3,4,5,6] 6
[3,4,5,6,7] 7       =>  X      Y     
[4,5,6,7,8] 8          (5,5)  (5,)
[5,6,7,8,9] 9
```

> 위의 예는, x 라는 5일치 데이터를 가지고 y 라는 데이터 1개를 예측한다는 예시이다.



이제까지는 Input Shape가 `(O, )` 라는 Column 만 있었는데 이제부터 `(O, O)`의 데이터가 나온다.

```shell
Input shape = < >(열, 몇개씩 자를지)

예를 들어, 
    x       y
[1/2/3/4/5] 5     1개씩 잘라서 y에 대입한다.
[2,3/4,5/6] 6     2개씩 잘라서 y에 대입한다.
[3,4,5/6,7] 7     3개씩 잘라서 y에 대입한다.
[4,5,6,7/8] 8     4개씩 잘라서 y에 대입한다.
[5,6,7,8,9] 9
```



#### View Codes

```python
# Keras15_LSTM1

#1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7]])     # (5,3)
y = array([4,5,6,7,8])    
# (5, ) == [] 하나만 넣으면 1D array이다. 내가 생각했던 (1,5)는 2D array

# Data reshape
print(x.shape[0], x.shape[1])
x = x.reshape(x.shape[0], x.shape[1], 1)

#2. 모델 구성
model = Sequential()
model.add(LSTM(10, activation='relu', input_shape=(3,1))) # (열, 몇개씩 자를지)
```

> 일반 DNN 이였다면 Input Shape가 (3, )로 지정했지만, RNN 모델에서는 몇개씩 자를지 구분해줘야 한다. 그래서 Reshape로 (5,3) 에서 (5,3,1) 로 변경했다.



## LSTM 모델 앙상블

> 예를 들어(6,1) 모델로 하나로 할 수 있는데, 왜 (3,1),(3,1) 두개로 나눔?



- 1개로 하려면 열을 합쳐야 한다.(번거로워진다)
  - ex) csv 파일 두개 받았는데 모델 2개 만들고 Ensemble 하면 파일 자체를 바꾸지 않아도 된다.

* Ensemble 하려면 Input shape가 다 맞아야 한다.

  ```shell
  # 예를 들어 두개의 데이터가 있을 때,
   X1     Y1     :     x2        y2
  (13,3) (13,)   :   (11, 3)   (11, )  
  
  >> 위 데이터를 모델로 Ensemble할 때, 양쪽이 동일한 shape가 아니면 ERROR
  ```

  



## 파라미터 참고

#### Verbose

```python
#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x, y, epochs=240, batch_size=1, verbose=3)
```

> verbose는 훈련과정이 프린트 되는 과정에서 원하는 것만 볼 수 있게 한다.

- verbose=1 (디폴트)
- verbose=0 은 fitting 과정 보이지 않는다.
- verbose=2 실행과정 안보이고, 값만 보인다.
- verbose=3 은 진행중인 epoch 순서만 보인다(예: Epoch 240/240 )
- 3 이상은 값이 같다고 한다.



#### Early Stopping

```python
#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')

model.fit(x, y, epochs=240, batch_size=1, verbose=1, callbacks=[early_stopping])
```

> Fit 과정에서 callback 파라미터로 적용한다.

- `Patience=` : 값이 변화 없는 구간에 얼마나 참을거냐는 말(위의 예는 30번 훈련하는 동안 변화 거의 없으면 훈련 종료)
- `monitor=` : 관찰 할 대상
- `mode=` : 
  - monitor의 카테고리에 따라 자동 설정
  - `=min`, `=max` 또한 설정 할 수 있다.
- Early Stopping은 학습률,보폭과 연관 있어서 같이 잘 조정해야한다. 
- 과적합 구간이 판단이 잘 안되는 단점이 있음

