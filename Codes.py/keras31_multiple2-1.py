# keras31_multiple2-1(LSTM)
# MNIST 예제 같은 데이터에 CNN만 쓸 수 있을까?

#############################################################################
# 1. 데이터
from numpy import array
                # 10, 4
def split_sequence2(seqence, n_steps):
    x,y = list(), list()
    for i in range(len(seqence)):     # 10
        end_ix = i +n_steps           # 0 + 4 = 4 /// 0 + 4
        if end_ix > len(seqence):   # 4 > 10 ??
            break
            
        seq_x, seq_y = seqence[i:end_ix, :-1], seqence[end_ix-1, -1]   # x=0,1,2,3 / y=4 
        x.append(seq_x)                                      
        y.append(seq_y)
    return array(x), array(y)
##############################################################################

in_seq1 = array([10,20,30,40,50,60,70,80,90,100])
in_seq2 = array([15,25,35,45,55,65,75,85,95,105])

out_seq = array([in_seq1[i] + in_seq2[i] for i in range(len(in_seq1))])   # 선행된 두 개 를 더한 새로운 객체

print(in_seq1.shape)  #(10,)
print(out_seq.shape)  #(10,)

in_seq1 = in_seq1.reshape(len(in_seq1), 1)
in_seq2 = in_seq2.reshape(len(in_seq2), 1)
out_seq = out_seq.reshape(len(out_seq), 1)

print(in_seq1.shape)  #(10, 1)
print(in_seq2.shape)  #(10, 1)
print(out_seq.shape)  #(10, 1)


from numpy import hstack   # 배열 결합
dataset = hstack((in_seq1, in_seq2, out_seq))
print(dataset)             # (10, 3)
##############################################################################

n_steps = 3
x, y = split_sequence2(dataset, n_steps)

for i in range(len(x)):
    print(x[i], y[i])
    
print(x.shape, y.shape)   # (8, 3, 2) (8,)
##############################################################################

# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense, LSTM

model = Sequential()

model.add(LSTM(10, activation='relu', input_shape=(3,2)))
model.add(Dense(5))
model.add(Dense(1))

model.summary()

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from keras.callbacks import TensorBoard
# tb_hist = TensorBoard(log_dir='./graph', histogram_freq=0, write_images=True, write_graph=True)
# , callbacks=[tb_hist]
model.fit(x, y, epochs=100, batch_size=1)

predict = array([[90,95], [100,105],[110,115]])
predict = predict.reshape(1,3,2)
results = model.predict(predict, batch_size=1)

print(results)