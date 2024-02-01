import numpy as np
V = [0.5,0.5,0.5,0.5,0.5]
alpha = 0.15
disc = 1
sum = [0]*5
for i in range(100):
    pos = 2  # Reset position to the center at the beginning of each episode
    T = True
    while  T == True:
        old_pos = pos
        R=0
        if np.random.rand()<0.5:
            if pos>0:
                pos-=1
                vpos = V[pos]
            else:
                vpos = 0
                T = False
        else:
            if pos<4:
                pos+=1
                vpos = V[pos]
            else:
                R=1
                vpos = 0
                T = False
        #states.append(old_pos)
        #rewards.append(R)    
        #V[old_pos] += V[old_pos] + alpha*(R + disc * vpos - V[old_pos])
        sum[old_pos] += + alpha*(R + disc * vpos - V[old_pos])
    for i in range(len(sum)):
        V[i] = sum[i];

print (V)
