# Learning Keras

> 수업 내용 중 정리 필요한 내용 추가로 정리하고 머리속에 넣자



1. *다중 LSTM*
2. *Scikit-learn 라이브러리를 통한 Data Scaling*
3. *참고(활성화 함수 사용)*



## 다중 LSTM 

Keras의 LSTM 모델에서는 `return_sequences=True` 를 통해서 LSTM 층이 바로 벡터화 진행 되는 것을 중단하고, LSTM 층을 더 붙일 수 있다.

```python
# 다중 LSTM 모델 만들어보면서 Data shape의 변화 확인해보기

from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM, Reshape

# 1. 데이터 준비
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], 
           [5,6,7], [6,7,8], [7,8,9], [8,9,10],
           [9,10,11], [10,11,12], [20,30,40], [30,40,50],
           [40,50,60]])                                        # (13,3)
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])               # (13, )

print("x :", x)
x = x.reshape(x.shape[0], x.shape[1], 1)
print("x.reshape :", x)

#2. 모델 구성
model = Sequential()

model.add(LSTM(10, activation='relu', return_sequences=True, input_shape=(3,1))) # return_sequence 를 통해 (None,3,10)으로 반환

model.add(LSTM(2, activation='relu', return_sequences=True))
# return_sequence 를 통해 (None,3,2)으로 반환
model.add(LSTM(10, activation='relu', return_sequences=False))
# (None,10)으로 반환
```

> 위에 처럼 계속 다중 LSTM으로 엮게 되면 좋은 결과를 얻을 수 있을까? 
>
> 답은 `없다` 이다. 앞에서 만들었던 LSTM의 "기억"이 다음 Layer에서 같은 "기억"이 될 수 있다고 장담할 수 없다.



딥러닝 모델의 텐서의 흐름을 명확히 이해하고 있다면, Reshape 메소드로도 Return Sequence를 구현할 수 있다.

```python
model.add(Reshape((10,1)))
```



## scikit-learn으로 Data Scaling

Data Scaling은 다양한 범위의 데이터를 비교 분석하기 쉽게 만들어 준다. 데이터가 가진 Feature의 scale이 많이 차이 나는 경우, 내가 원하는 만큼의 성능을 얻지 못할 수 있다. 

<img src="https://www.sporbiz.co.kr/news/photo/201802/195433_165642_5910.jpg" style="zoom:50%;" />

> 위의 이미지 처럼 실제 Feature는 다양한 범주로 존재한다. 우리는 Scaling을 통해 하나의 통일된 규칙성을 만들어 주는 것.



<img src="https://t1.daumcdn.net/cfile/tistory/997D23455C63973C2D" style="zoom:67%;" />



#### (1) Normalization 정규화

일반적으로 값을 **0에서 1사이** 의 범위로 재 조정하는 것을 의미한다. 

![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSK0JyXHT7G2ma5Fsznc2Zjpe0dtDIT1Q-TjQJQfHYOd0d3eqwSUA&s)

> 각 값(x)에 최소 값을 빼고, (최대 - 최소) 로 나눈다. 



#### (2) Standardization 표준화

일반적으로 데이터의 **평균을 0**으로 하고 **표준편차를 1**로 조정하는 것을 의미한다.

![](https://mblogthumb-phinf.pstatic.net/MjAxODA3MzFfNDIg/MDAxNTMzMDIwOTUwMjk0.rDioAfP5eatJ8SPoUOJFMZfUQtprCC99gvKFkWU6k3Yg.gDW93QnA2VR0XyYI8KFZD2bytVEhlEsDT-A8wHk9DS0g.PNG.angryking/2304E84656B1B53A07.png?type=w800)

> 각 값(x)에 평균(mean)을 빼주고, 표준편차로 나눈다.

두 모델 전부 이상치가 있으면 변환된 값이 매우 달라질 수 있다는 단점이 존재한다.

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/74/Normal_Distribution_PDF.svg/1280px-Normal_Distribution_PDF.svg.png" style="zoom:67%;" />

> 파란색 선은 Feature를 뽑기 가장 좋고 황색 선은 가장 난해한 선이다.



예를 들어, 아래와 같이 스케일링 된 데이터가 있다고 하자.

|      |  X   |      |  Y   |  -   |      |  X   |      |  Y   |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
|  1   |  2   |  3   |  4   |  -   | 0.1  | 0.2  | 0.3  |  4   |
|  2   |  3   |  4   |  5   |  -   | 0.2  | 0.3  | 0.4  |  5   |
|  3   |  4   |  5   |  6   |  -   | 0.3  | 0.4  | 0.5  |  6   |
|  4   |  5   |  6   |  7   |  -   | 0.4  | 0.5  | 0.6  |  7   |

시험지에 문제가 바뀌었다고 하더라도 예측해야 될 답은 같은 것 처럼, x를 전처리 해도 매칭 되는 쌍은 바뀌지 않는다. (그래도 Y값 까지 변환해주는 것이 더 나은 방법)



#### View Codes

```python
# scikit-learn의 Scaler 메소드를 활용하여 데이터 스케일링

# 1. 데이터
from numpy import array
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7],
          [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12],
          [20000,30000,40000], [30000,40000,50000], 
          [40000,50000,60000], [100,200,300]])
y = array([4,5,6,7,8,9,10,11,12,13,50000,60000,70000,400])

from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x)               # scikit-learn에는 fit의 종류가 많다.
x = scaler.transform(x)
```

> sckit-learn 라이브러리를 통해 쉽게 표준화, 정규화 스케일링을 구현할 수 있다.



```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
```

Scikit - learn 라이브러리로 스케일링 적용할 때,  **Train data 에 `fit` 한 scaler**를 가지고 나머지 데이터들을 **`Transform`** 해야 한다. (Train data의 logic 적용하기 위함)



## 참고

#### Activation Function활성화 함수 

대체로 효율적인 함수는 ReLU 라고 한다.  그 외에도 Sigmoid, Elu, tanh도 있다. 내가 하고자 하는 모델에 맞춰서 선정하면 된다.(현재 하고 있는 RNN 모델은 Tanh를 사용)

- 이진 분류 : 마지막 레이어에 `sigmoid`

- 다중 분류 : 마지막 레이어에 `softmax`

- Linear(선형 함수)는 Default 값으로 있다.



