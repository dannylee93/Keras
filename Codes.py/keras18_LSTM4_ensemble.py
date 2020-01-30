# 1. 데이터 준비
from numpy import array

x1 = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], 
           [5,6,7], [6,7,8], [7,8,9], [8,9,10],
           [9,10,11], [10,11,12], [20,30,40], [30,40,50],
           [40,50,60]])                                        # (13,3)
y1 = array([4,5,6,7,8,9,10,11,12,13,50,60,70])                 # (13, )

x2 = array([[10,20,30], [20,30,40], [30,40,50], [40,50,60], 
           [50,60,70], [60,70,80], [70,80,90], [80,90,100],
           [90,100,110], [100,110,120], [2,3,4], [3,4,5],
           [4,5,6]])                                           # (13,3)
y2 = array([40,50,60,70,80,90,100,110,120,130,5,6,7])          # (13, )

x1 = x1.reshape(x1.shape[0], x1.shape[1], 1)
x2 = x2.reshape(x2.shape[0], x2.shape[1], 1)    # x1, x2 둘 다 (13,3,1) 모델로 변경 == 3행 1열 이 13개 있는 3차원 모델


#2. 모델 구성
from keras.models import Model
from keras import layers    # Input, LSTM, Dense는 layer에서 온다는 걸 기억하기 위해 일부러 layers 로 불러왔다.

input_1 = layers.Input(shape=(3,1))
input_tensor_1 = layers.LSTM(32, activation='relu')(input_1)   # x1에 투입할 모델
hidden_layer_1 = layers.Dense(16, activation='relu')(input_tensor_1)
hidden_layer_1 = layers.Dense(8, activation='relu')(hidden_layer_1)
output_tensor_1 = layers.Dense(1, activation='relu')(hidden_layer_1)

input_2 = layers.Input(shape=(3,1))
input_tensor_2 = layers.LSTM(32, activation='relu')(input_2)    # x2에 투입할 모델
hidden_layer_2 = layers.Dense(16, activation='relu')(input_tensor_2)
hidden_layer_2 = layers.Dense(8, activation='relu')(hidden_layer_2)
output_tensor_2 = layers.Dense(1, activation='relu')(hidden_layer_2)

from keras.layers.merge import concatenate , Add

merged_model = concatenate(inputs=[output_tensor_1, output_tensor_2])  # 두 개 이상은 []로 묶기
# merged_model = Add()([output_tensor_1, output_tensor_2])  # concatenate 대신 Add를 사용해도 가능하다.

output_tensor_3 = layers.Dense(8)(merged_model)        # 첫 번째 아웃풋 모델
output_tensor_3 = layers.Dense(1)(output_tensor_3)

output_tensor_4 = layers.Dense(8)(merged_model)        # 두 번째 아웃풋 모델
output_tensor_4 = layers.Dense(1)(output_tensor_4)

model = Model(inputs=[input_1, input_2],
              outputs=[output_tensor_3, output_tensor_4])

model.summary()


#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit([x1,x2], [y1,y2], epochs=240, batch_size=1, verbose=1)


#4. 평가예측
evaluation = model.evaluate([x1,x2], [y1,y2], batch_size=1)

print("evaluation:{}".format(evaluation))


x1_predict = array([[6.5,7.5,8.5], [50,60,70], [70,80,90],[100,110,120]])
x2_predict = array([[6.5,7.5,8.5], [50,60,70], [70,80,90],[100,110,120]])

x1_predict = x1_predict.reshape(4,3,1)
x2_predict = x2_predict.reshape(4,3,1)
p = model.predict([x1_predict, x1_predict], batch_size=1)

print("predict: {}".format(p))
