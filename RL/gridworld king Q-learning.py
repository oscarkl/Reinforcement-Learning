import numpy as np
import random
rows = 7
cols = 10
actions = [
    #(0, 0),
    (0, 1),
    (0, -1),
    (1, 0),
    (1, 1),
    (1, -1),
    (-1, 0),
    (-1, 1),
    (-1, -1),
]
size = 10**6
S = []
S[:] = [0, 0] * size 
Q = np.zeros((rows, cols,len(actions)), dtype=float)
pi = np.ones((rows, cols), dtype=int)
pos = [3, 0]
eps = 0.1
alpha = 0.5
disc = 1
R = -1
wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
for r in range(rows):
    for c in range(cols):
        pi[r, c] = np.random.randint(len(actions))
Goal = [3,7]
def move():
    t = 0
    S[t] = [3, 0]
    
    while True:
        s = S[t]
        #print(s)
        pia = pi[s[0], s[1]]
        act = pia
        #act = np.argmax(pi[s[0], s[1]])
        
        if np.random.rand() <= eps:
            act=random.randrange(0,len(actions))
        acttt = actions[act] 

        if s[0] - wind[s[1]] < 0:
            next_s = [0, s[1]]
        else:
            next_s = [s[0] - wind[s[1]], s[1]]
        
          #pos + =
        
        #print(next_s[0], actions[act][0])
        if next_s[0] + actions[act][0]<0 or next_s[1] + actions[act][1]<0 or next_s[0] + actions[act][0]>=rows or next_s[1] + actions[act][1]>=cols:
            acttt = [0,0]
            
        next_s = [next_s[0] + acttt[0], s[1] + acttt[1]]
        next_a = pi[next_s[0], next_s[1]]
        if np.random.rand() <= eps:
            next_a=random.randrange(0,len(actions))

        if(s == Goal):
           Q[s[0], s[1], act] += alpha*(1 + disc*0 - Q[s[0], s[1], act])
           print(t)
           break 

        Q[s[0], s[1], act] += alpha*(R + disc*Q[next_s[0], next_s[1], next_a] - Q[s[0], s[1], act])
        pi[s[0], s[1]] = np.argmax(Q[s[0], s[1]])      

        t+=1
        S[t] = next_s

        if(t==100000):
            print(t)
            break

def run(episodes = 10000):
    for i in range(episodes):
        move()
#move()
run()
#print(Q)