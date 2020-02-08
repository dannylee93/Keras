# Learning Keras

> 수업 내용 중 정리 필요한 내용 추가로 정리하고 머리속에 넣자



1. *CNN codes in Keras*
2. *CNN with MNIST*
3. *Representative examples like CIFAR-10*
4. *Visualization by using Matplotlib*



## CNN Codes in Keras

> CNN 알고리즘 이론이 케라스라는 라이브러리를 통해서 어떻게 구현되는지 핵심적인 코드를 통해 알아보자



CNN(Convolutional Neural Network) 라는 심층 합성곱 신경망의 알고리즘에 대한 이론적인 내용은 아래 주소에 정리했다. 

*[Understanding CNN Algorithm](https://github.com/dannylee93/TIL/blob/master/Deep_Learning/딥러닝(Deep Learning)_08(CNN_Algorithm).md) (나의 Github/ TIL 저장소)*



### View Codes

복잡한 CNN의 알고리즘이 케라스 라이브러리를 통해서 얼마나 간결하게 표현될 수 있는지 확인하자.

```python
# CNN 모델링 기초
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten 

# 2D 이미지 데이터의 대표적 예
model = Sequential()

model.add(Conv2D(7, (2,2), input_shape=(5,5,1),strides=(2) ,padding='valid'))  # 5X5의 흑백 사이즈를 2X2 피쳐맵 사이즈로 보면서 32번 보겠다

model.add(MaxPooling2D(pool_size=(2,2)))  
# 내가 지정한 Pooling size 만큼 stride가 디폴트로 지정된다. 

model.add(Flatten()) #Convnet의 아웃풋 >> 일반 덴스층에 바로 붙지못한다.

model.add(Dense(1))  # Fully-Connected layer로 진입

model.summary()
```

> 시퀀셜 모델로 표현한 Convnet의 코드

- 위의 단 몇 줄만으로도 CNN의 복잡한 알고리즘을 구현해 냈다. 물론 파라미터 조정을 세부적으로 했을 때는 조금 더 까다로워 지지만 전체적인 맥락은 위와 같다.
- **꼭 기억해야할 파라미터**
  - `Kernel_size= `: Input의 이미지 **데이터를 어떤 사이즈로** 볼 것인가 정하는 파라미터
  - `strides=` : 커널(kernel)을 **얼마나 이동시키면서** 볼 것인지 정하는 파라미터
  - `padding=` : 
    - **valid** ==  kernel size와 stride로 feature map 을 뽑아낸 만큼 데이터 사이즈가 줄어든다.
    - **same** == kernel size와 stride로 feature map 을 뽑아 내더라도 전체적인 **사이즈 변하지 않게** 테두리에 패딩 넣어준다.
  - `MaxPooling` : Convolutional layer 를 지나고 나온 feature map 의 데이터 중 특징이 강한 (max 값을 가진 픽셀)만 본다.
  - `Flatten` : Convolutional layer를 통해 저수준의 특징들을 뽑았다면, Fully-Connected layer로 진입시킬 때 **Reshape** 한다고 생각하면 좋다.





## CNN with MNIST

> MNIST는 이미지 예제의 대표적 샘플인 손글씨 데이터로, 28*28 픽셀 사이즈의 데이터로 60,000개의 훈련 데이터와 10,000개의 훈련 데이터로 구성되어 있다.

<img src="https://www.researchgate.net/profile/Steven_Young11/publication/306056875/figure/fig1/AS:393921575309346@1470929630835/Example-images-from-the-MNIST-dataset.png" alt="Mnist 이미지 검색결과" style="zoom:50%;" />



### View Codes

MNIST 데이터셋의 데이터 전처리와 모델링에서 중요한 핵심 파라미터를 알아보자

```python
# 1. MNIST 모델 불러오기
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.callbacks import EarlyStopping
import numpy as np

# MNIST 예제에서 알아서 데이터셋 분리 해놓음
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

> 머신러닝 패키지에서는 대부분 유명한 데이터셋을 내장해놨다.



불러온 MNIST 예제 중  x 데이터를 정제해준다.

```python
# 2. 데이터 형태 변경
x_train = x_train.reshape(x_train.shape[0],28,28,1).astype('float32')/255
x_test = x_test.reshape(x_test.shape[0],28,28,1).astype('float32')/255
```

위와 같이 변경한 이유는,

MNIST는 흑백(channel==1)의 이미지 데이터이기 때문에 최소 0과 최대 255사이의 데이터로 구성되어 있는 픽셀 데이터이다. 

```shell
array([[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,  84, 185, 159, 151,  60,  36,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0, 222, 254, 254, 254, 254, 241, 198,
        198, 198, 198, 198, 198, 198, 198, 170,  52,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,  67, 114,  72, 114, 163, 227, 254,
        225, 254, 254, 254, 250, 229, 254, 254, 140,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  17,  66,
         14,  67,  67,  67,  59,  21, 236, 254, 106,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,  83, 253, 209,  18,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,  22, 233, 255,  83,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0, 129, 254, 238,  44,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,  59, 249, 254,  62,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0, 133, 254, 187,   5,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   9, 205, 248,  58,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0, 126, 254, 182,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,  75, 251, 240,  57,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         19, 221, 254, 166,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   3,
        203, 254, 219,  35,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  38,
        254, 254,  77,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  31, 224,
        254, 115,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 133, 254,
        254,  52,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  61, 242, 254,
        254,  52,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 121, 254, 254,
        219,  40,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 121, 254, 207,
         18,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0]], dtype=uint8)
```

1. 가장먼저 (60000, 28, 28)의 이미지를 **채널 수**를 가진 데이터로 인식시켜주기 위해  Reshape(28,28,1) 로 변경한다.
2. 데이터를 **부동소수점**을 가진 데이터 형태('float32')로 변경한다.
3. 255로 나누어준 이유는, **Minmax-Scaling** (정규화)에서 각 분자 분모의 min 값은 생략하고 간략화 하여 계산한 것이다.



x데이터를 정제한 다음, y데이터를 정제해준다. 처음 y데이터는 MNIST의 예제대로 0~9의 손글씨 숫자를 분류하는 데이터 이다.

```python
set(y_test)  # 0부터 9까지 총 10개의 라벨을 분류하는 문제.
-----------------------------------------------(실행)
{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
```

0부터 9까지의 데이터를 각각 `One-hot Encoding` 해준다.

```python
# 2. 데이터 형태 변경(y값(label)의 값을 수치형으로 변경)
from keras.utils import to_categorical

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
```

위와 같이 전처리한 y데이터의 형태는 (60000, ), (10000, )에서 (60000, 10), (10000, 10)의 데이터 형태로 변경된다.

```shell
# 예를 들어,
[5] == [[0,0,0,0,1,0,0,0,0,0], 
```



모델링한 결과는 다음과 같다.

```python
# 3. 모델링
model = Sequential()
model.add(Conv2D(128, (2,2), input_shape=(28,28,1) ,padding='valid')) 
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.summary()

# 4. 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=10, batch_size=10, validation_split=0.2,verbose=2)
```

> 다중분류 문제에서 활성화 함수로 'softmax' , 손실함수로 'categorical_crossentropy', 평가지표로 정확도를 선택했다.



## Representative examples

> CNN 알고리즘을 배우는데 사용되는 대표적인 예제로 MNIST 뿐만 아니라 CIFAR-10과 Fashion MNIST 데이터도 있다.



#### (1) CIFAR - 10

머신러닝에 사용되는 대표적인 예시 중 하나로, 10가지의 클래스 분류 문제이다.(비행기, 자동차, 새, 고양이, 사슴, 개, 개구리, 말, 배 및 트럭)

각 클래스마다 6,000개의 이미지가 있으며, (60000, 32, 32, 3) 데이터의 형태로 구성되어있다.

```python
# 1. 데이터 불러오기
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 2. 데이터 형태 변경
x_train = x_train.reshape(x_train.shape[0],32,32,3).astype('float32')/255
x_test = x_test.reshape(x_test.shape[0],32,32,3).astype('float32')/255
```

> 주의할 점은 코드 중, Reshape 부분이다. 이하 모델링은 MNIST와 동일하게 했다.



#### (2) Fashion MNIST

Fashion-MNIST는 기존의 MNIST데이터셋(10개 카테고리의 손으로 쓴 숫자)을 대신해 사용할 수 있다. 그 이유는 MNIST와 동일한 이미지 크기(28x28)이고, 동일한 학습 셋 (60,000), 테스트 셋(10,000)으로 나눌 수 있기 때문이다. 

<img src="https://raw.githubusercontent.com/KerasKorea/KEKOxTutorial/master/media/10_1.png" style="zoom:50%;" />

> 케라스 코리아의 김태영님 블로그를 참고하면 자세한 정보를 얻을 수 있다.



## Visualization

앞서 살펴본 MNIST 예제 모델링을 하면서 내 모델이 얼마나 성능을 내고 있는지 확인해 볼 필요가 있다. Keras와 Scikit-learn 에서 모델링 할 때 `fit`을 하면 아래와 같은 결과를 볼 수 있을거다.

```powershell
Epoch 1/10
 - 7s - loss: 0.4280 - acc: 0.8763 - val_loss: 0.2848 - val_acc: 0.9175
Epoch 2/10
 - 6s - loss: 0.2876 - acc: 0.9175 - val_loss: 0.2623 - val_acc: 0.9259
Epoch 3/10
 - 6s - loss: 0.2647 - acc: 0.9230 - val_loss: 0.2419 - val_acc: 0.9318
 ......
```

> fit을 실행하면, 아래와 같이 각 Epoch마다의 값이 history로 출력된다.

```python
# 6. history 만들기(fit을 하면 loss, metrics, history가 생성된다)
print(hist.history.keys())
---------------------------------------------------------(실행)
dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])
```



**Matplotlib** 라이브러리로 그래프화 시키자

```python
# 7. 그래프 그리기 
import matplotlib.pyplot as plt

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])

plt.title('model loss, accuracy')
plt.ylabel('loss, acc')
plt.xlabel('epoch')
plt.legend(['train loss', 'test loss', 'train acc', 'test acc'])
plt.show()
```

아래와 같은 그래프를 볼 수 있다.

![](data:image/png;base64,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%0AcwwAAAAASUVORK5CYII=%0A)



## References

- 케라스 창시자에게 배우는 딥러닝
- [케라스 코리아의 김태영님 Github](https://tykimos.github.io/)

