# keras28_univarate2 & DNN

# 1. 데이터
from numpy import array, transpose
                # 10, 4
def split_sequence(seqence, n_steps):
    x,y = list(), list()
    for i in range(len(seqence)):     # 10
        end_ix = i +n_steps           # 0 + 4 = 4 /// 0 + 4
        if end_ix > len(seqence)-1:   # 4 > 10-1 ??
            break
            
        seq_x, seq_y = seqence[i:end_ix], seqence[end_ix]   # x=0,1,2,3 / y=4 
        x.append(seq_x)                                      
        y.append(seq_y)
    return array(x), array(y)


dataset = [10,20,30,40,50,60,70,80,90,100]
n_steps = 3
x, y = split_sequence(dataset, n_steps)

for i in range(len(x)):
    print(x[i], y[i])
    
print(x.shape, y.shape)    # (7,3), (7, )


# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense



model = Sequential()
model.add(Dense(10, input_shape=(3, )))
model.add(Dense(2))
model.add(Dense(3))
model.add(Dense(1))

model.summary()

# 3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x, y, epochs=100, batch_size=1)

# 4. 평가예측
loss, mae = model.evaluate(x, y, batch_size=1)
print("Loss : {}, mae : {}".format(loss, mae))