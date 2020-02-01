# keras27_univarate (x,y 만들어주는 함수 정의)

from numpy import array
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


dataset = [0,1,2,3,4,5,6,7,8,9]

'''

0,1,2,3 / 4
1,2,3,4 / 5
'
'
'
5,6,7,8 / 9

'''

n_steps = 3
x, y = split_sequence(dataset, n_steps)
print(x)
print(y)