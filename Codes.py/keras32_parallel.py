# keras32_평행(parallel) / DNN

#############################################################################
# 1. 데이터
from numpy import array
                # 10, 4
def split_sequence2(seqence, n_steps):
    x,y = list(), list()
    for i in range(len(seqence)):     # 10
        end_ix = i +n_steps           # 0 + 4 = 4 /// 0 + 4
        if end_ix > len(seqence)-1:   # 4 > 10 ??
            break
            
        seq_x, seq_y = seqence[i:end_ix, : ], seqence[end_ix-1, :]   # x=0,1,2,3 / y=4 
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
    
print(x.shape, y.shape)   # (7, 3, 3) (7,3)   x와 y의 shape를 어떻게 맞출까
##############################################################################

# 2. 모델 구성
from keras.models import Sequential
from keras.layers import Dense, LSTM, Reshape

x = x.reshape(7,9)

model = Sequential()

model.add(Dense(5, input_shape=(9, )))
model.add(Dense(5))
model.add(Dense(3))

model.summary()

#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from keras.callbacks import TensorBoard
model.fit(x, y, epochs=100, batch_size=1)

predict = array([[90,95,185], [100,105,205],[110,115,225]])
predict = predict.reshape(1,9)
results = model.predict(predict, batch_size=1)

print(results)