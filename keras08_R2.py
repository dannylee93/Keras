#1. 데이터
import numpy as np
x = np.array(range(1, 101))
y = np.array(range(1, 101))

# Data split하기
from sklearn.model_selection import train_test_split

x_train, x_test , y_train, y_test = train_test_split(x, y,
                                                     test_size=0.4,
                                                     shuffle = False)
x_test, x_val , y_test, y_val = train_test_split(x_test, y_test,
                                                     test_size=0.5,
                                                     shuffle = False)


#2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

# model.add(Dense(5, input_dim = 1))
model.add(Dense(5, input_shape=(1, )))
model.add(Dense(2))
model.add(Dense(3))
model.add(Dense(1))

model.summary()

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=1,
           validation_data=(x_val, y_val))
# model.fit(x, y, epochs=100)

#4. 평가예측
loss, mse = model.evaluate(x_test, y_test, batch_size=1)
print('mse: ', mse)

x_prd = np.array([101,102,103])
results = model.predict(x_prd, batch_size=1)
print(results)

# RMSE 만들기
from sklearn.metrics import mean_squared_error
y_predict = model.predict(x_test, batch_size=1)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test
    , y_predict))
print("RMSE :", RMSE(y_test, y_predict))