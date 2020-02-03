# Learning Keras

> 수업 내용 중 정리 필요한 내용 추가로 정리하고 머리속에 넣자



1. *Tensorboard 사용하기*
2. *Save model & Load model*
3. *univarate & multiple* & Parallel



## Tensorboard

> 텐서플로우와 함께 제공되는 브라우저 기반의 시각화 도구. 텐서플로우를 백엔드로 설정했을 때만 케라스 라이브러리에서 사용할 수 있다.



Keras 의 `Callback` 은 fit 메소드가 호출될 때, 전달되는 객체이다. 훈련하는 동안 여러 지점에서 콜백을 호출한다.



#### Callback 객체의 사례

1. *모델 체크포인트 저장: 어떤 특정 지점에서 현재 가중치 저장*
2. *Early Stopping*
3. *하이퍼 파라미터 값 동적 조정*
4. *훈련과 검증 지표를 로그에 기록하거나 시각화*



#### Keras 에서 Tensorboard 사용하기

1. 로그 기록을 저장할 폴더 생성한다. (편의를 위해 가급적 현재 작업중인 폴더 내에서 만들기)

2. callbacks로 메소드를 불러오고 객체를 정의한다.

   ```python
   # 훈련의 fit 계에서 callback 함수에 투입한다.
   model.compile(loss='mse', optimizer='adam', metrics=['mse'])
   
   from keras.callbacks import TensorBoard
   tb_hist = TensorBoard(log_dir='./graph',histogram_freq=0, write_graph=True, write_images=True)
   
   model.fit(x, y, epochs=100, batch_size=1, callbacks=[tb_hist])
   ```

   - `histogram_freq` : 에서 `=1` 은 1에포크마다 활성화 출력 히스토그램 기록한다.
   - 

1. 커맨드 창(Cmd)을 연다.

2. 내가 만든 로그파일 위치로 가서 아래 명령어를 수행한다

   ```powershell
   tensorboard --logdir=./graph
   ```
5. 크롬에서 http://localhost:6006/ 으로 들어가면 완성 
* 6006은 텐서보드의 주소명

6. 가장 최신 로그파일만 적용된다.
**안되면 커맨드 다시확인
7. 순환신경망의 경우에는 fit 함수를 여러번 호출하기 때문에 제대로 된 학습 상태를 볼 수 없다. (새로운 Class를 커스터 마이징 해야한다 / https://tykimos.github.io/2017/07/09/Training_Monitoring/ 참고하기)



## Save Model & Load Model

> 다른 사람의 모델을 가져와 커스터마이징 하고 싶을 때는 어떻게 진행 할까



1. 모델을 만들거나 다른 곳에서 가져온다

   ```python
   # 모델 구성
   from keras.models import Sequential
   from keras.layers import Dense
   
   model = Sequential()
   
   model.add(Dense(5, input_shape=(1, )))
   model.add(Dense(2))
   model.add(Dense(3))
   model.add(Dense(1))
   
   model.save('./save/savetest01.h5')
   print('저장 완료')
   ```

2. 만들었다면, `Save` 메소드로 내가 원하는 위치에 저장

3. 다른 파일에서 save 되어 있는 모델을 models의 `load_model` 메소드 사용

   * 불러온 후 층 늘릴수도 있다.  또한 층에 `name` 인자를 변경해야 한다.

   ```python
   #2. 모델구성 
   from keras.layers import Dense
   from keras.models import load_model, Sequential
   
   model = load_model("./save/savetest01.h5")
   model.add(Dense(2, name="dense_a"))
   model.add(Dense(3, name="dense_b"))
   model.add(Dense(1, name="dense_c"))
   model.summary()
   ```



## univarate & multiple & Parallel

#### 실습해보기

1. 함수 분석(split squence)
2. DNN 모델 만들기
3. 자료형 loss
4. `[[90,95], [100,105],[110,115]] `Predict 하기