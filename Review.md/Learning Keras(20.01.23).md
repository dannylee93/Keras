# Learning Keras

> 수업 내용 중 정리 필요한 내용 추가로 정리하고 머리속에 넣자



1. *데이터 검증하기(Validation)*
2. *데이터 나누기(Split)*
3. *평가지표 만들고 활용해보기*
4. *MLP(Multi Layer Perceptron)에서 행렬*





## Data Validation

> 훈련하는 동안 처음 본 데이터에 대한 모델의 정확도를 측정하기 위해서는 검증 세트를 만들어야 한다.



1. 모든 데이터가 세팅되어 있을 때 : fitting 할 때는 Train set에 투입 하고 평가 할 때  Test Set투입한다.
2. **validation_data=(x_test, y_test)** 을 파라미터에 투입(fitting은 머신이 한다는 전제 / 훈련 과정에서 검증)



#### View Codes

```python
 model.fit(x_train, y_train, epochs=100, batch_size=1,
           validation_data=(x_val, y_val))
```



*데이터를 각 용도에 맞춰 정제되어 있는 상태로 받지 않았을 때는 어떻게 할까*



## Data Split

> 데이터를 잘 모델링 하는 것 만큼 중요한게 전처리이며, 활용하고자 하는 목적에 맞게 데이터를 나누는 과정이 필요하다.



1. 직접 리스트를 **슬라이싱(Slicing)** 해서 데이터를 나눈다.
2. scikit learn의 함수를 활용해서 데이터를 나눈다.



#### View Codes

```python
# Data split하기
x_train = x[:60]
x_test = x[60:80]
x_val = x[80:]

y_train = y[:60]
y_test = y[60:80]
y_val = y[80:]
```



```python
# Data split하기
from sklearn.model_selection import train_test_split

x_train, x_test , y_train, y_test = train_test_split(x, y,
                                                     test_size=0.4,
                                                     shuffle = False)
x_test, x_val , y_test, y_val = train_test_split(x_test, y_test,
                                                     test_size=0.5,
                                                     shuffle = False)
```

> 데이터를 Train set과 Test set으로 먼저 나눈다음 그 안에서 또 validation set을 만들었다.



## 평가지표 Evaluation Index

학습된 모델이 실제로 무언가 학습했는지 평가하는 방법 중 첫 번째 도구는 Loss Function을 보고 적절한지 판단하는 것이다.



모델과 관련된 지표(metric)를 살펴본다. 이 지표(metric)은 예측한 레이블과 정답 레이블을 비교하는 도구이며, 회귀문제의 경우 두가지 일반적 지표가 있다. (R2, RMSE)



#### R2

+1과 0 사이의 값인 두 변수 간의 상관관계를 측정하는 방법이다. +1은 완전 상관관계를 나타내고, 0은 상관관계가 없음을 나타낸다.

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/77/Okuns_law_quarterly_differences.svg/1200px-Okuns_law_quarterly_differences.svg.png" alt="수학적 정의" style="zoom:67%;" />



#### RMSE

RMSE는 예측값과 정답값과의 평균적인 차이를 측정한 것이다. RMSE값은 상대적으로 높고 R2와 달리 발견하지 못한 오류를 진단해준다.



#### View Codes

```python
# RMSE 만들기
from sklearn.metrics import mean_squared_error
y_predict = model.predict(x_test, batch_size=1)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test
    , y_predict))
print("RMSE :", RMSE(y_test, y_predict))
```

>scikit-learn에서 mse 지표를 import하고 그 데이터에 루트를 덧붙인다.



```python
# R2 구하기
from sklearn.metrics import r2_score

r2_y_predict  = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)
```

> scikit-learn에서 R2를 불러온다.





## MLP 에서 행렬

> Multi Layer Perceptron에서 행렬은 Input data로 진입시키기 전, 꼭 Reshape을 통해 정제 해줘야 한다.



#### 데이터를 Reshape 해줘야 하는 이유?

<img src="http://cfile208.uf.daum.net/R400x0/215FBF385791B0A00EDB1A" style="zoom:67%;" />

보기와 같은 성적이 있다고 할 때, 각 개인은 행에 있는 자기 이름을 보고 열을 보지만, 전체적인 데이터를 보는 입장에서는 각 열 (국,영,수,,, 등등)로 구분된 데이터가 의미가 있다. 딥러닝에서도 마찬가지로 각 열에 맞춘 데이터로 Input data에 입장시켜야 된다.

예를들어,

````shell
([1,2,3,4,5,6,7,8,9,10],
 [11,12,13,14,15,16,17,18,19,20])
````

이라는 (2, 10)의 데이터가 있다고 치자.  사실 우리는 위의 '국영수 성적 현황표'처럼 각 열 데이터에 맞는 데이터 수 만큼 보고 싶은것이다. 그렇기 때문에 우리는 Reshape 과정을 거쳐서

```shell
[[ 1 11]
 [ 2 12]
 [ 3 13]
 [ 4 14]
 [ 5 15]
 [ 6 16]
 [ 7 17]
 [ 8 18]
 [ 9 19]
 [10 20]]
```

(10,2) 형식의 데이터로 진입을 시켜야 된다는 것이다.

#### View Codes

```python
#1. 데이터
import numpy as np
x = np.array([range(1, 101), range(101, 201), range(101, 201)])  #(3,100)
y = np.array([range(1, 101)]) #(1, 100)
y2 = np.array(range(1, 101)) #(100, )

print(x.shape, y.shape, y2.shape)


# Data reshape
x = np.transpose(x)
y = np.transpose(y)
```

> 데이터의 shape에 맞게 2차원의 배열을 transpose를 사용해서 변형 해준다.



```python
# numpy array 형태를 통해 행렬 이해하기
  
[1,2,3]                              # (1,3)    2차원 텐서
[[1,2,3,], [1,2,3]]                  # (2,3)    2차원 텐서
[[[1,2,3], [1,2,3]]]                 # (1,2,3)  3차원 행렬
[[[1,2],[1,2], [[1,2,][1,2]]]        # (2,2,3)  3차원 행렬
```

> 행렬 모양의 예시이다.