# Funtional API Model (Ensemble)
# 다르게 해주는 이유: 각각의 가중치를 비교할 수 있고, 또는 다른 가중치를 두게끔 하고 싶을 때.


#1. 데이터
import numpy as np
x1 = np.array([range(1, 101), range(101, 201), range(201, 301)])  #(3,100)
x2 = np.array([range(1001, 1101), range(1101, 1201), range(1201, 1301)])  #(3,100)


y1 = np.array([range(1, 101), range(101, 201), range(201, 301)])
y2 = np.array([range(1001, 1101), range(1101, 1201), range(1201, 1301)])
y3 = np.array([range(301, 401), range(401, 501), range(501, 601)])

print(x1.shape, x2.shape, y1.shape, y2.shape, y3.shape)


# Data reshape
x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.transpose(y1)
y2 = np.transpose(y2)
y3 = np.transpose(y3)


# Data split하기
from sklearn.model_selection import train_test_split

x1_train, x1_test, x2_train, x2_test, y1_train, y1_test, y2_train, y2_test, y3_train, y3_test  = train_test_split(x1, 
                                                                                                                  x2, 
                                                                                                                  y1, 
                                                                                                                  y2, 
                                                                                                                  y3, 
                                                                                                                  test_size=0.4, 
                                                                                                                  random_state=0, 
                                                                                                                  shuffle = False)
x1_test, x1_val, x2_test, x2_val, y1_test, y1_val, y2_test, y2_val, y3_test, y3_val = train_test_split(x1_test, 
                                                                                                       x2_test, 
                                                                                                       y1_test,
                                                                                                       y2_test, 
                                                                                                       y3_test, 
                                                                                                       test_size=0.5, 
                                                                                                       random_state=0, 
                                                                                                       shuffle = False)

# Train & test split을 본인만의 함수로 정의해서 나중에 import 하면서 사용할 수 있음.

#2. 모델 구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input

# model = Sequential()
input_tensor_1 = Input(shape=(3, ))
hiddenlayers_1 = Dense(32)(input_tensor_1)
hiddenlayers_1 = Dense(16)(hiddenlayers_1)
hiddenlayers_1 = Dense(8)(hiddenlayers_1)
output_tensor_1 = Dense(1)(hiddenlayers_1)

input_tensor_2 = Input(shape=(3, ))
hiddenlayers_2 = Dense(16)(input_tensor_2)
hiddenlayers_2 = Dense(8)(hiddenlayers_2)
hiddenlayers_2 = Dense(8)(hiddenlayers_2)
output_tensor_2 = Dense(1)(hiddenlayers_2) # 내가 원하는 모델 쪽에 더 많은 가중치 두는것도 가능하다.

from keras.layers.merge import concatenate  # 모델을 사슬처럼 엮는 메소드

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

model = Model(inputs=[input_tensor_1,input_tensor_2], 
              outputs=[output_tensor_3, output_tensor_4, output_tensor_5]) # 앙상블 형식의 모델 사용할 때 리스트('[]') 사용하게 된다.


model.summary()


#3. 훈련
model.compile(loss='mae', optimizer='adam', metrics=['acc'])
model.fit([x1_train, x2_train], [y1_train,y2_train,y3_train], epochs=100, batch_size=10,
           validation_data=([x1_val,x2_val], [y1_val,y2_val,y3_val]))


#4. 평가예측
# 변수를 1개 / MSE 개수별로

results_acc = model.evaluate([x1_test, x2_test], [y1_test,y2_test,y3_test], batch_size=10)
print('results_acc: ', results_acc)


x1_prd = np.array([[201,202,203], [204,205,206],[204,205,206]])
x2_prd = np.array([[201,202,203], [204,205,206],[204,205,206]])

x1_prd = np.transpose(x1_prd)
x2_prd = np.transpose(x2_prd)

results_prd = model.predict([x1_prd,x2_prd], batch_size=1)
print(results_prd)

print("----------------------------------------------")
print("----------------------------------------------")
# 평가지표 / RMSE 만들기
from sklearn.metrics import mean_squared_error
y_predict = model.predict([x1_test, x2_test], batch_size=1)

# print(y_predict)
# print(y1_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
    
RMSE_mean = (RMSE(y1_test, y_predict[0]) + RMSE(y2_test, y_predict[1]) + RMSE(y3_test, y_predict[2])) / 3


print("RMSE_mean :", RMSE_mean )

print("----------------------------------------------")
print("----------------------------------------------")

# 평가지표 / R2 만들기
from sklearn.metrics import r2_score

r2_1  = r2_score(y1_test, y_predict[0])
r2_2  = r2_score(y2_test, y_predict[1])
r2_3  = r2_score(y3_test, y_predict[2])
r2_mean  = (r2_1 + r2_2 + r2_3) / 3

print("R2 : ", r2_mean)

