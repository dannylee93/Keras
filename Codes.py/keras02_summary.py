#1. 데이터
import numpy as np
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# print(x.shape, y.shape)

#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(5, input_dim = 1))
model.add(Dense(2))
model.add(Dense(3))
model.add(Dense(1))

model.summary()

'''
# 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x, y, epochs=100, batch_size=1)
# model.fit(x, y, epochs=100)

# 평가예측
loss, mse = model.evaluate(x, y, batch_size=1)
print('mse: ', mse)

x_prd = np.array([11,12,13])
resultsA = model.predict(x_prd, batch_size=1)
print(resultsA)
'''