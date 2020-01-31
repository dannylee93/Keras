# Scaler와 LSTM 모델링을 통해 값 확인

from numpy import array ,transpose
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1. 데이터
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

print(x)

x = x.reshape(x.shape[0], x.shape[1], 1)

#2. 모델 구성
from keras.models import Model
from keras.layers import Dense, LSTM, Input

# model = Sequential()
input_tensor = Input(shape=(3, 1))
hiddenlayers = LSTM(32)(input_tensor)
hiddenlayers = Dense(16)(hiddenlayers)
hiddenlayers = Dense(8)(hiddenlayers)

output_tensor = Dense(1)(hiddenlayers)   # Hidden Layer의 이름을 각각 부여하지 않고 동일한 이름으로 해도 가능

model = Model(inputs=input_tensor, outputs=output_tensor)

model.summary()


#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x, y, epochs=100, batch_size=1)

#4. 평가예측
loss, metrics = model.evaluate(x, y, batch_size=10)
print('loss: {}, metrics: {} '.format(loss, metrics))

x_predict = array([[201,202,203], [204,205,206], [207,208,209]])
x_predict = x_predict.reshape(x_predict.shape[0], x_predict.shape[1], 1)
results = model.predict(x_predict, batch_size=1)
print(results)