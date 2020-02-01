# Load Model & Customize

#1. 데이터
import numpy as np
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

print(x.shape, y.shape)

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

#2. 모델구성 


from keras.layers import Dense
from keras.models import load_model, Sequential

model = load_model("./save/savetest01.h5")
model.add(Dense(2, name="dense_a"))
model.add(Dense(3, name="dense_b"))
model.add(Dense(1, name="dense_c"))
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