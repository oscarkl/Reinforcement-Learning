import numpy as np
import matplotlib.pyplot as plt
data_str = """
0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 3
0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 3
0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 3
0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 3
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 3
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 3
1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0
1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0
1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0
1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0
1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0
1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0
1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0
1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0
0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0
0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0
0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0
0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0
0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0
0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0
0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0
0 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0
0 0 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0
0 0 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0
0 0 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0
0 0 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0
0 0 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0
0 0 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0
0 0 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0
0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0
0 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0
0 0 0 2 2 2 2 2 2 0 0 0 0 0 0 0 0
"""


track_array = [list(map(int, line.split())) for line in data_str.strip().split("\n")]
track_array = np.array(track_array)
#print(track_array)
# Display the resulting 2D array
#for row in track_array:
    #print(row)

eps = 0.1
disc = 1

actions = [
    (0, 0),
    (0, 1),
    (0, -1),
    (1, 0),
    (1, 1),
    (1, -1),
    (-1, 0),
    (-1, 1),
    (-1, -1),
]
rows = 32
cols = 17
vel_len = 5
act_len=9

Q = np.random.rand(rows, cols, vel_len, vel_len, act_len) * 400 - 500
C = np.zeros((rows, cols, vel_len, vel_len, act_len),dtype=float)
π = np.ones((rows, cols, vel_len, vel_len), dtype=int)
#print (Q[0][0][0][0][0])
for r in range(rows):
    for c in range(cols):
        for h in range(vel_len):
            for v in range(vel_len):
                π[r, c, h, v] = np.argmax(Q[r, c, h, v, :])

size = 10**6
S = []
R=[0] * size
S[:] = [(0, 0), (0, 0)] * size 
# A array with Int64
A = np.empty(size, dtype=np.int64)

# B array with Float64
B = np.empty(size, dtype=np.float64)

border = np.argwhere(track_array==0)
track = np.argwhere(track_array==1)
start = np.argwhere(track_array==2)
finish = np.argwhere(track_array==3)

actions_as_lists = [list(action) for action in actions]
actions_array = np.array(actions_as_lists)

def get_valid_actions(x, y, max_limit=4):
    valid_actions = []
    for i in range(len(actions)):
        new_x = x + actions[i][0]
        new_y = y + actions[i][1]

        # Check if the new position is within bounds
        if 0 <= new_x <= max_limit and 0 <= new_y <= max_limit:
            if new_x!=0 or new_y!=0:
                valid_actions.append(i)

    return valid_actions
reward_fin = 0;
reward_move = -1
reward_out = -100


def make_trajectory(eps, noise = True):
    t=1
    R[t]=0
    velocity = np.array([0, 0], dtype=np.int16)
    random_start = np.random.choice(len(start))
    S[t] = (start[random_start], velocity)
    while True:
        s = S[t]
        r=0

        acts = get_valid_actions(s[1][0], s[1][1], 4)
        
        πa = π[s[0][0], s[0][1], s[1][0], s[1][1]]
        πa_valid = πa in acts
        #print(πa_valid)
        if np.random.rand() >= eps:
            if πa_valid:
                a = πa
                b = 1 - eps + eps / len(acts)
            else:
                
                a=np.random.choice(acts)
                
                b = eps / len(acts)
        else:
            a=np.random.choice(acts)
            b = (eps if πa_valid else 1) / len(acts)
        #if noise == True and np.random.rand() < 0.1:
            #a = 0
            #b = 0.1
        
        A[t]=a
        B[t]=b
        act = actions[a]
        
        vel = (s[1][0] + act[0], s[1][1] + act[1])
        #print(vel, act, acts)
        
        next_s = ((s[0][0] + vel[1]*-1, s[0][1] + vel[0]), vel)
        #print(s)
        finisha = np.array(finish)
        bordera = np.array(border)
        tracka = np.array(track)
        finish_tuples = [tuple(x) for x in finisha]
        border_tuples = [tuple(x) for x in bordera]
        track_tuples = [tuple(x) for x in tracka]
        if next_s[0] in finish_tuples:
            #print(S[t])
            #R[t]+=100
            return t
        

            
        elif next_s[0] in border_tuples:
            random_start = np.random.choice(len(start))
            vel=(0,0)
            s = (start[random_start], vel)
            r-=100

            #print(border)
           # print("border")
        elif next_s[0] in track_tuples:
            #rint(track)
            #print("track")
            s=next_s
            
        else:
            r-=100
            random_start = np.random.choice(len(start))
            vel=(0,0)
            s = (start[random_start], vel)
            
            #print("out")
            #break
        r-=1
        t += 1
        S[t] = s
        R[t]=r
        #if(t%100==0):
            #return t
        #print(next_s)
        #path = if

def run_episode(T):
    Gt=0
    W = 1
    #print(T)
    for t in range (T, 1, -1):
        s = S[t]
        sa = (s[0][0], s[0][1], s[1][0], s[1][1], A[t])
        Gt=disc*Gt -1
        C[sa]=C[sa]+W 
        #print(Q[sa])
        Q[sa] += W * np.abs(Gt - Q[sa]) / C[sa]
        
        acts = get_valid_actions(s[1][0], s[1][1], 4)
        #print(s)
        #print(acts)
        π[s[0][0], s[0][1], s[1][0], s[1][1]] = np.argmax(Q[s[0][0], s[0][1], s[1][0], s[1][1]][acts])
        if A[t] != π[s[0][0], s[0][1], s[1][0], s[1][1]]:
            return t
        W = W/B[t]
    return 0

    
#print(run_episode())

def output_trajectories():
    T = make_trajectory(0.0, noise=False)
    for t in range (1, T):
        print(S[t], actions[A[t]])
    #print("S: ", S[T])
    #print("A: ", A[T-1])
    #print(π)
    print("R: ", -1 * T)

def main():
    episode_num = 10**4
    rewards = []

    for i in range(1, episode_num):
        T = make_trajectory(eps)
        t = run_episode(T)
        print(f"episode {i}: {T}, {t}, {T-t}")
        rewards.append(T)
        #if i % 1 == 0:
            #output_trajectories()
            #T = make_trajectory(0.0)
            

    output_trajectories()
    plt.semilogy(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Rewards Over Episodes')
    plt.show()

main()