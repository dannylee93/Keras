# 모델링
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
from keras.initializers import he_normal
from keras.optimizers import adam

num_classes = 19

model=Sequential()

model.add(Dense(64,input_shape=(20, ))) # input dimension
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.1))

model.add(Dense(128))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.1))

model.add(Dense(128, kernel_initializer=, kernel_regularizer=l2(0.01), bias_regularizer= l2(0.01), bias_initializer=))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.1))

model.add(Dense(256))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.1))

model.add(Dense(128))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.1))

model.add(Dense(128))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.4))

model.add(Dense(num_classes, activation='softmax'))

model.summary()

# 옵티마이저 파라미터 조정
optimizer_adam = adam(lr=0.0008, clipvalue=1.0, decay=1e-8)

# 모델 훈련
model.compile(loss='categorical_crossentropy', optimizer= optimizer_adam, metrics=['accuracy'])

from keras.callbacks import LearningRateScheduler
import math

# 학습률 조정
def step_decay(epoch):
    initial_lrate = 0.001  # 최초 학습률
    drop = 0.5            # 학습할 때마다 수정되는 양
    epochs_drop = 20.0    # 학습속도를 변경하는 빈도
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))  # 최초 학습률 * 감소량
    return lrate

callback = LearningRateScheduler(step_decay)

hist = model.fit(x_train, y_train, batch_size=38, epochs=100, verbose=2,validation_split=0.25)
loss, acc = model.evaluate(x_test,y_test,batch_size=64)
print('Loss:',loss,'Accuracy:',acc)



