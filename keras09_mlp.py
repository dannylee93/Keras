# Multi Layer Perceptron

#1. 데이터
import numpy as np
x = np.array([range(1, 101), range(101, 201)])
y = np.array([range(1, 101), range(101, 201)])
# 2차원의 데이터(2, 100)가 형성된다.

# Data reshape
x_reshaped = np.reshape(x, (100,2))
y_reshaped = np.reshape(y, (100,2))

print(x_reshaped.shape)

# x = np.transpose(x)
# y = np.transpose(y)


# Data split하기
from sklearn.model_selection import train_test_split

x_train, x_test , y_train, y_test = train_test_split(x_reshaped, y_reshaped,
                                                     test_size=0.4,
                                                     shuffle = False)
x_test, x_val , y_test, y_val = train_test_split(x_test, y_test,
                                                     test_size=0.5,
                                                     shuffle = False)


#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

# model.add(Dense(5, input_dim = 2))
model.add(Dense(64, input_shape=(2, )))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(2))  # 2차원 데이터이기 때문에 아웃풋 노드를 2로 바꿔야한다.

model.summary()

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=10,
           validation_data=(x_val, y_val))
# model.fit(x, y, epochs=100)

#4. 평가예측
loss, mse = model.evaluate(x_test, y_test, batch_size=10)
print('mse: ', mse)

x_prd = np.array([[201,202,203], [204,205,206]])
x_prd = np.transpose(x_prd)
results = model.predict(x_prd, batch_size=1)
print(results)

# RMSE 만들기
from sklearn.metrics import mean_squared_error
y_predict = model.predict(x_test, batch_size=1)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE :", RMSE(y_test, y_predict))