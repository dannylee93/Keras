# Funtional API Model

#1. 데이터
import numpy as np
x = np.array([range(1, 101), range(101, 201), range(101, 201)])  #(3,100)
y = np.array([range(1, 101)]) #(1, 100)
y2 = np.array(range(1, 101)) #(100, )

print(x.shape, y.shape, y2.shape)


# Data reshape
x = np.transpose(x)
y = np.transpose(y)


# Data split하기
from sklearn.model_selection import train_test_split

x_train, x_test , y_train, y_test = train_test_split(x, y,
                                                     test_size=0.4,
                                                     shuffle = False)
x_test, x_val , y_test, y_val = train_test_split(x_test, y_test,
                                                     test_size=0.5,
                                                     shuffle = False)


#2. 모델 구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input

# model = Sequential()
input_tensor = Input(shape=(3, ))

# Dense_1 = Dense(32)(input_tensor)
# Dense_2 = Dense(16)(Dense_1)
# Dense_3 = Dense(8)(Dense_2)

# output_tensor = Dense(1)(Dense_3)

hiddenlayers = Dense(32)(input_tensor)
hiddenlayers = Dense(16)(hiddenlayers)
hiddenlayers = Dense(8)(hiddenlayers)

output_tensor = Dense(1)(hiddenlayers)   # Hidden Layer의 이름을 각각 부여하지 않고 동일한 이름으로 해도 가능

model = Model(inputs=input_tensor, outputs=output_tensor)

model.summary()

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
model.fit(x_train, y_train, epochs=100, batch_size=10,
           validation_data=(x_val, y_val))
# model.fit(x, y, epochs=100)

#4. 평가예측
loss, mse = model.evaluate(x_test, y_test, batch_size=10)
print('mse: ', mse)

x_prd = np.array([[201,202,203], [204,205,206],[207,208,209]]) # 1개 더 늘림
x_prd = np.transpose(x_prd)
results = model.predict(x_prd, batch_size=1)
print(results)

# RMSE 만들기
from sklearn.metrics import mean_squared_error
y_predict = model.predict(x_test, batch_size=1)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE :", RMSE(y_test, y_predict))