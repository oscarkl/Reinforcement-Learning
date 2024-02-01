import numpy as np
from tabulate import tabulate

class State:
    def __init__ (self, id):
        self.val =0
        self.id = id

        self.nextS = [self.move("up"), self.move("left"), self.move("right"), self.move("down")]

    def move(self, dir):
        if dir=="up":
            if self.id < 4:
                return self.id
            if self.id == 4:
                #self.id=self.id-4
                return 0
            else:
                return self.id - 4

        if dir=="left":
            if self.id % 4 == 0:
                return self.id
            if self.id == 1:
                #self.id=self.id-1
                return 0
            else:
                return self.id - 1

        if dir=="right":
            if self.id % 4 == 3:
                return self.id
            
            if self.id == 14:
                #self.id=self.id+1
                return 0
            else:
                return self.id + 1

        if dir=="down":
            if self.id > 11:
                return self.id

            if self.id == 11:
                #self.id=self.id+4
                return 0
            else:
                return self.id + 4

    def evaluate(self, S):
        #optimal action
        #'''
        V = -10
        for i in range(0, 4):
            V =max(V, S[self.nextS[i]].val)
        self.val = -1 + V
        
        '''
        #random action
        V = 0 
        for i in range(0, 4):
            V +=S[self.nextS[i]].val
        self.val = -1 + 0.25 * V
        #'''
    
def improve(k = 100):
    V  = []
    S_T = State(0)
    S = {0: S_T}
    for j in range(1, 15):
        S[j] = State(j)
    for loops in range(k):
        n = np.random.random()
        if n > 0.5:
            for k in range(1, 15):
                S[k].evaluate(S)
        else:
            for j in range(14, 0, -1):
                S[j].evaluate(S)
    for t in range(0,16):
        if t == 0 or t == 15:
            V.append("0")
        else:
            V.append(S[t].val)
    draw(V)

def draw(valueArray):
    for i in range(4):
        print("----------------------")
        print("| {:.0f} | {:.0f} | {:.0f} | {:.0f} |".format(float(valueArray[i * 4]), float(valueArray[i * 4 + 1]), float(valueArray[i * 4 + 2]), float(valueArray[i * 4 + 3])))


improve(10)