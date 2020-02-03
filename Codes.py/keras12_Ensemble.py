# Funtional API Model (Ensemble)
# 다르게 해주는 이유: 각각의 가중치를 비교할 수 있고, 또는 다른 가중치를 두게끔 하고 싶을 때.


#1. 데이터
import numpy as np
x1 = np.array([range(1, 101), range(101, 201), range(201, 301)])  #(3,100)
x2 = np.array([range(1001, 1101), range(1101, 1201), range(1201, 1301)])  #(3,100)
y = np.array([range(101, 201)]) #(1, 100)

print(x1.shape, x2.shape, y.shape)


# Data reshape
x1 = np.transpose(x1)
x2 = np.transpose(x2)
y = np.transpose(y)


# Data split하기
from sklearn.model_selection import train_test_split

x1_train, x1_test , x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y,
                                                     test_size=0.4,
                                                     random_state=0,
                                                     shuffle = False)
x1_test, x1_val , x2_test, x2_val, y_test, y_val = train_test_split(x1_test, x2_test, y_test,
                                                     test_size=0.5,
                                                     random_state=0,
                                                     shuffle = False)



#2. 모델 구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input

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


model.summary()


#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit([x1_train, x2_train], [y_train], epochs=100, batch_size=10,
           validation_data=([x1_val,x2_val], y_val))


#4. 평가예측
loss, mse = model.evaluate([x1_test, x2_test], [y_test], batch_size=10)
print('mse: ', mse)

# 평가지표 / RMSE 만들기
from sklearn.metrics import mean_squared_error
y_predict = model.predict([x1_test, x2_test], batch_size=1)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE :", RMSE(y_test, y_predict))

# 평가지표 / R2 만들기
from sklearn.metrics import r2_score

r2_y_predict  = r2_score(y_test, y_predict)
print("R2 : ", r2_y_predict)