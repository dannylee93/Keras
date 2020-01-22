import numpy as np
import keras
import tensorflow as tf
from sklearn import model_selection
from keras import models
from keras import layers
from keras import optimizers
from keras.utils import to_categorical



# 분류 모델(로지스틱)
data3 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
target3 = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(16, input_dim=1, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='softmax'))
    model.compile(optimizer='rmsprop',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = build_model()
model.fit(data3, target3, epochs=2, batch_size=1)

result = model.evaluate(test_data, test_label)
model.predict(np.array([11, 12, 13]))