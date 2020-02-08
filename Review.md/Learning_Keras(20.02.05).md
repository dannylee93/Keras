# Learning Keras

> 수업 내용 중 정리 필요한 내용 추가로 정리하고 머리속에 넣자



1. *CNN codes in Keras*
2. *CNN with MNIST*
3. *Representative examples like CIFAR-10*
4. *Visualization by using Matplotlib*



## CNN Codes in Keras

> CNN 알고리즘 이론이 케라스라는 라이브러리를 통해서 어떻게 구현되는지 핵심적인 코드를 통해 알아보자



CNN(Convolutional Neural Network) 라는 심층 합성곱 신경망의 알고리즘에 대한 이론적인 내용은 아래 주소에 정리했다. 

[Understanding CNN Algorithm](https://github.com/dannylee93/TIL/blob/master/Deep_Learning/딥러닝(Deep Learning)_08(CNN_Algorithm).md)   **Github/ TIL** Repository



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



<p align="center"><img src="https://www.researchgate.net/profile/Steven_Young11/publication/306056875/figure/fig1/AS:393921575309346@1470929630835/Example-images-from-the-MNIST-dataset.png" alt="Mnist 이미지 검색결과" style="zoom:50%;" /><img src="" /></p>


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





## References

- 케라스 창시자에게 배우는 딥러닝
- [케라스 코리아의 김태영님 Github](https://tykimos.github.io/)

