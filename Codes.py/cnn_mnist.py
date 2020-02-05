# 1. 데이터 불러오기
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
import numpy as np

# MNIST 예제에서 알아서 데이터셋 분리 해놓음
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. 데이터 형태 변경

x_train = x_train.reshape(x_train.shape[0],28,28,1).astype('float32')/255
x_test = x_test.reshape(x_test.shape[0],28,28,1).astype('float32')/255
# MNIST 예제는 충분히 잘 정제되어 있는 데이터이기 때문에 astype으로 바로 변경

print(type(x_train))

# 3. y값 수치화 변경
from keras.utils import to_categorical

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 4. 모델링(CNN)
model = Sequential()
model.add(Conv2D(128, (2,2), input_shape=(28,28,1), activation='relu' ,padding='valid')) 
model.add(Conv2D(64, (2,2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.summary()

#5. 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=10, validation_split=0.2,verbose=1)

# 6. 평가예측
evaluation = model.evaluate(x_test, y_test, batch_size=1)
print("evaluation:{}".format(evaluation))