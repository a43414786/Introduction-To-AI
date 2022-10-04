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
class GA :
    def __init__(self,input):
        self.input = input
        self.size = len(input)
        self.pop_size = 0               #Population size
        self.chromosomes = 0            #Chromosomes num
        self.selected_chromosomes = []  #Selected chromosomes
        self.best_chromosomes = []      #Best chromosome
        self.numOfGene = self.size      #Number of genes in chromosome
        self.fitness = []               #Fitness for each chromosome
        


###########################
if __name__ == '__main__':

    

    with open('input.json','r') as f:
        data = json.load(f)
        for i,key in enumerate(data):
            input = data[key]
            # print(input)
            bf_solver = GA(input)
            assignment = bf_solver.ans # ⽤演算法得出的答案
            solver = Problem(input)
            print(i)
            print('Assignment:', assignment) # print 出分配結果
            print('Cost:', solver.cost(assignment)) # print 出 cost 是多少
            
    
        