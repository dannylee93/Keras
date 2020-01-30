# 1. 데이터 준비
from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], 
           [5,6,7], [6,7,8], [7,8,9], [8,9,10],
           [9,10,11], [10,11,12], [20,30,40], [30,40,50],
           [40,50,60]])                                        # (13,3)
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])               # (13, )

print("x_shape :", x.shape)

x = x.reshape(x.shape[0], x.shape[1], 1)


#2. 모델 구성
model = Sequential()

model.add(LSTM(32, activation='relu', input_shape=(3,1))) # (열, 몇개씩 자를지)
model.add(Dense(16, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='relu'))

model.summary()


#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')

model.fit(x, y, epochs=240, batch_size=1, verbose=1, callbacks=[early_stopping])


#4. 평가예측
loss, mae = model.evaluate(x, y, batch_size=1)
print("eval_loss : {} eval_mae : {}".format(loss, mae))



x_predict = array([[6.5,7.5,8.5], [50,60,70], [70,80,90],[100,110,120]])
x_predict = x_predict.reshape(4,3,1)
p = model.predict(x_predict, batch_size=1)

print(p)