# Learning Keras

> 수업 내용 중 정리 필요한 내용 추가로 정리하고 머리속에 넣자



1. *Functional API (함수형 모델)*
2. *Model Ensemble*
3. *scikit-learn 에서 Data split*



## Functional API

> Keras 에서는 모델을 정의하는 방법이 두가지가 있다. 하나는 Sequential 모델이고, 다음 하나가 바로 함수형 API 모델이다. 



Sequential 모델에서는 첫번째 layer에서 **Input shape** 를 선언 해준 후, 순차적으로 **Dense** 층을 **add** 했다. 

```python
#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(64, input_shape=(3, )))
model.add(Dense(32))
#.....이후 생략
```



#### 함수형 API에서는 어떤 점이 다를까?



`함수형` 이라는 말처럼 마치 함수에서 x와 y라는 독립변수와 종속변수를 통해서 함수를 정의했던 모습같이, 처음 **Input layer의 모습을 먼저 정의**하고 내가 원하는 Hidden Layer를 **객체에 계속 저장**한 후 **마지막에 모델을 선언**한다.

~~두가지 이상의 모델을 붙여서 만들려면 시퀀셜모델로 불가~~

![](https://tykimos.github.io/Keras/warehouse/2017-2-22_Integrating_Keras_and_TensorFlow_2.png)

```python
from keras.models import Model
from keras.layers import Dense, Input

input_tensor = Input(shape=(3, ))

hiddenlayers = Dense(32)(input_tensor)
hiddenlayers = Dense(16)(hiddenlayers)
hiddenlayers = Dense(8)(hiddenlayers)

output_tensor = Dense(1)(hiddenlayers)   
# Hidden Layer의 이름을 각각 부여하지 않고 동일한 이름으로 해도 가능

model = Model(inputs=input_tensor, outputs=output_tensor)
```

위 코드의 모델을 `Summary` 하면 다음과 같다.

```shell
# Summary
```





## Model Ensemble

> 함수형 API는 여러 모델을 앙상블 할 수 있다. 각각의 모델로 나누어 학습시키는 중 모델을 Merge 하여 최종 모델을 앙상블 하는 방법이다.



예를 들어, (3, 1) shape의 데이터를 2개로 나누어 각각의 모델로 나누어 학습시킬 수 있다. 그런데 차라리 (6, 1)의 shape인 하나의 모델로 학습시키지 않고 다르게 해줄까?



**데이터 특성에 따라, 각각의 가중치를 비교해야하거나 또는 그렇게 하고싶을 때 나눌 수 있다.**

```python
input_tensor_1 = Input(shape=(3, ))
hiddenlayers_1 = Dense(32)(input_tensor_1)
hiddenlayers_1 = Dense(16)(hiddenlayers_1)
hiddenlayers_1 = Dense(8)(hiddenlayers_1)
output_tensor_1 = Dense(1)(hiddenlayers_1)

input_tensor_2 = Input(shape=(3, ))
hiddenlayers_2 = Dense(16)(input_tensor_2)
hiddenlayers_2 = Dense(8)(hiddenlayers_2)
hiddenlayers_2 = Dense(8)(hiddenlayers_2)
output_tensor_2 = Dense(1)(hiddenlayers_2) 
# 내가 원하는 모델 쪽에 더 많은 가중치 두는것도 가능하다.

from keras.layers.merge import concatenate  
# 모델을 사슬처럼 엮는 메소드

merged_model = concatenate([output_tensor_1, output_tensor_2])
# concatenate에서 디폴트 값중에 하나인 axis=-1를 변경하면 파라미터의 개수로 확인 가능

middle_1 = Dense(4)(merged_model)
middle_2 = Dense(7)(middle_1)
output = Dense(1)(middle_2)

model = Model(inputs=[input_tensor_1,input_tensor_2], outputs=output) 
# 파라미터 개수가 2개 이상일 때, 리스트('[]') 사용하게 된다.
```

> Keras의 merge 함수에서 concatenate라는 메소드를 이용하면 편리하게 진행할 수 있다.



위 코드의 모델을 `Summary` 하면 다음과 같다.

```shell
# Summary
```



Concatenate 메소드로 병합 이후에 모델을 나눌 수도 있다.

```python
merged_model = concatenate([output_tensor_1, output_tensor_2])

middle_1 = Dense(4)(merged_model)
middle_2 = Dense(7)(middle_1)
middle_3 = Dense(1)(middle_2)  # merge된 마지막 layer

output_tensor_3 = Dense(8)(middle_3)        # 첫 번째 아웃풋 모델
output_tensor_3 = Dense(3)(output_tensor_3)

output_tensor_4 = Dense(8)(middle_3)        # 두 번째 아웃풋 모델
output_tensor_4 = Dense(3)(output_tensor_4)

output_tensor_5 = Dense(8)(middle_3)        # 세 번째 아웃풋 모델
output_tensor_5 = Dense(3)(output_tensor_5)
```

위 코드의 모델을 `Summary` 하면 다음과 같다.

```shell
# Summary
```



## Scikit-learn 에서 Data split

> 우리는 Scikit-learn 에서 model_selection 의 train_test_split 메소드를 통해 `train & test` 세트로 데이터를 나눴다.



<img src="https://t1.daumcdn.net/cfile/tistory/9951E5445AAE1BE025" style="zoom:67%;" />

x와 y 두 가지에 대해서 나누는 건 알겠는데, **그 이상의 변수들**이 있으면 어떻게 나눌까?



**답**은 **데이터 변수가 있는 만큼** 인자를 두면 된다.



아래 예시는 두 개의 x 와 하나의 y 총 3가지 데이터를 **6 : 2 : 2(train : test : val)** 비율로 나눈 것이다.

```python
x1_train, x1_test , x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y,
                                                     test_size=0.4,
                                                     random_state=0,
                                                     shuffle = False)
x1_test, x1_val , x2_test, x2_val, y_test, y_val = train_test_split(x1_test, x2_test, y_test,
                                                     test_size=0.5,
                                                     random_state=0,
                                                     shuffle = False)
```

> 여기서 shuffle=False는 데이터를 섞지 않고 나누겠다는 의미이다.



주의할 점, Shuffle할 때 데이터들의 x와 y는 쌍으로 같이 바뀐다.

```python
x = [1,2,3,4,5]
y = [11,12,13,14,15]

# 위의 데이터가 있으면 아래는 Shuffle 후의 데이터 예시이다.

x = 1,5,3     :  2   :  4
y = 11,15,13  :  12  :  14
```

지도 학습 중 회귀 모델의 예시를 통해 이해해보자. 우선 **회귀(Regression)**란 타깃 변수의 연속된 출력을 예측하는 것이다.

*예를 들어* 

**{원펀치 : 3강냉이, 투펀치 : 5강냉이}** 데이터가 있다. 이 데이터를 가지고 학습시킨 다음 `key`에 대한 `value`를 예측한다고 가정했을 때, 타겟 `Test` 데이터는 **{쓰리펀치: 7강냉이}** 이고, 나의 학습 모델이 **{쓰리펀치:8강냉이}** 로 예측했다면 1강냉이 차이가 정확도 차이이며, 이를 **예측** 이라 하는 것이다.



그렇기 때문에 쌍으로 바뀌지 않으면 데이터의 상관관계를 없애는 것이나 마찬가지라 할 수 있다.