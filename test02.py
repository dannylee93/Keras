import numpy as np
import keras
import tensorflow as tf
from sklearn import model_selection
from keras import models
from keras import layers
from keras import optimizers

# 분류 모델(로지스틱)
data2 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
target2 = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

train_data, test_data, train_label, test_label = model_selection.train_test_split(data2, target2,
                                                                                 test_size=0.3,
                                                                                 random_state=0)

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(16, input_dim=1, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

model = build_model()
model.fit(train_data, train_label, epochs=2, batch_size=1)

result = model.evaluate(test_data, test_label)
model.predict(np.array([11, 12, 13]))