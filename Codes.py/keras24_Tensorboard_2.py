#1. 데이터
import numpy as np
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

print(x.shape, y.shape)

#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(16, input_shape=(1, )))
model.add(Dense(32))
model.add(Dense(1))

model.summary()

# 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

from keras.callbacks import EarlyStopping, TensorBoard
tb_hist = TensorBoard(log_dir='./graph',histogram_freq=0, write_graph=True, write_images=True)

model.fit(x, y, epochs=100, batch_size=1, callbacks=[tb_hist])


# 평가예측
loss, mse = model.evaluate(x, y, batch_size=1)
print('mse: ', mse)

x_prd = np.array([11,12,13])
results = model.predict(x_prd, batch_size=1)
print(results)