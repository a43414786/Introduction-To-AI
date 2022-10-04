from copy import copy
import json

class Problem:
    def __init__(self, input):
        self.input = input
        self.numTasks = len(input)
    def cost(self, ans):
        totalTime = 0
        for task, agent in enumerate(ans):
            totalTime += self.input[task][agent]
        return totalTime
###########################
class BF :
    def __init__(self,input):
        self.input = input
        self.size = len(input)
        self.minima = 0
        self.ans = list(range(self.size))
        self.arr = list(range(self.size))
        for i in range(self.size):
            self.minima += self.input[i][i]

    def permutation(self,size):
        if size == 1:
            temp = 0
            for i in range(self.size):
                temp += self.input[i][self.arr[i]]
            if self.minima > temp:
                self.minima = temp
                self.ans = copy(self.arr)
                

        for i in range(size):
            self.permutation(size-1)
            if(size & 1):
                self.arr[0],self.arr[size - 1] = self.arr[size - 1],self.arr[0]
            else:
                self.arr[i],self.arr[size - 1] = self.arr[size - 1],self.arr[i]

###########################
if __name__ == '__main__':

    

    with open('input.json','r') as f:
        data = json.load(f)
        for i,key in enumerate(data):
            input = data[key]
            # print(input)
            bf_solver = BF(input)
            bf_solver.permutation(len(input))
            assignment = bf_solver.ans # ⽤演算法得出的答案
            solver = Problem(input)
            print(i)
            print('Assignment:', assignment) # print 出分配結果
            print('Cost:', solver.cost(assignment)) # print 出 cost 是多少
            
    
        