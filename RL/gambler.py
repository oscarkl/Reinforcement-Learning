import numpy as np 
import matplotlib.pyplot as plt

def improve(ph = 0.4, theta = 0.00000000000001):
    V=[0]*100
    pi=[0]*100
    for i in range(0, 100):
        V[i] = np.random.random() * 1000
    V[0] = 0
    while True:
        delta = 0
        for s in range (1, 100):
            old_v=V[s]
            v=[0]*51
            for a in range (1, min(s, 100-s)+1):
                v[a]=0
                if a + s < 100:
                    v[a] += ph * (0 + V[s + a])
                    v[a] += (1 - ph) * (0 + V[s - a])
                if a + s == 100:
                    v[a] += ph
                    v[a] += (1 - ph) * (0 + V[s - a])
            op_a = np.argmax(v)
            pi[s] = op_a
            V[s]=v[op_a]
            delta = max(delta, abs(old_v-V[s]))
        if delta < theta:
            break
    return [V[1:100], pi[1:100]]

if __name__ == "__main__":
    [V1, pi1] = improve(ph=0.4)
    [V2, pi2] = improve(ph=0.25) 
    [V3, pi3] = improve(ph=0.55)
    S = np.linspace(1, 99, num=99, endpoint=True)
    plt.figure()
    plt.plot(S, V1)
    plt.plot(S, V2)
    plt.plot(S, V3)
    plt.show()
    plt.figure()
    plt.bar(S, pi1)
    plt.show()
    plt.figure()
    plt.bar(S, pi2)
    plt.show()
    plt.figure()
    plt.bar(S, pi3)
    plt.show()