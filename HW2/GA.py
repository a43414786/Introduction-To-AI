import json
from random import shuffle
import random

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
    def __init__(self,input,pop_size,iterations = 100,crossover_policy = 'uniform',mutate_probability = 0.1):
        self.input = input
        self.size = len(input)
        self.pop_size = pop_size        #Population size
        self.chromosomes = []           #Chromosomes
        self.selected_chromosomes = []  #Selected chromosomes
        self.best_chromosome = []       #Best chromosome
        self.best_fitness = 0
        self.best_iterations = 0
        self.fitness = []               #Fitness for each chromosome
        
        self.crossover_policy = crossover_policy
        self.mutate_probability = mutate_probability

        self.initialize()
        self.process(iterations)
    def process(self,iterations):
        
        for i in range(iterations):
            self.select()
            self.crossover()
            self.mutate()
            self.CalFitness()
            min_idx = self.argmin()
            if(self.best_fitness > self.fitness[min_idx]):
                self.best_fitness = self.fitness[min_idx]
                self.best_chromosome = self.copy(self.chromosomes[min_idx])
                self.best_iterations = i
    
    def normal(self,maxnum):
        """Generate a value between 1-100 in a normal distribution"""
        count = 100
        values =  sum([random.randint(0, maxnum) for _ in range(count)])
        return round(values/count)

    def argmin(self):
        amin = 0
        min = self.fitness[0]
        for i in range(len(self.fitness)):
            if self.fitness[i] < min:
                min = self.fitness[i]
                amin = i
        return amin
    def copy(self,arr):
        temp = []
        for i in arr:
            temp.append(i)
        return temp

    def sort(self):
        x = sorted(range(self.pop_size),key = lambda x: self.fitness[x])
        ctemp = []
        ftemp = []
        for i in x:
            ctemp.append(self.chromosomes[i])
            ftemp.append(self.fitness[i])
        self.chromosomes = ctemp
        self.fitness = ftemp

    def initialize(self):
        for _ in range(self.pop_size):
            temp = list(range(self.size))
            shuffle(temp)
            self.chromosomes.append(temp)
        self.CalFitness()
        min_idx = self.argmin()
        self.best_fitness = self.fitness[min_idx]
        self.best_chromosome = self.copy(self.chromosomes[min_idx])

    def select(self):
        self.selected_chromosomes = []
        self.sort()
        gfg = [random.expovariate(self.pop_size/3) for _ in range(self.pop_size)]
        for i in gfg:
            if(int(i) >= self.pop_size):
                self.selected_chromosomes.append(self.chromosomes[self.pop_size - 1])    
            else:
                self.selected_chromosomes.append(self.chromosomes[int(i)])
    
    def crossover_two(self,chromosome1,chromosome2,bp):
        temp1 = []
        temp2 = []
        for i in range(bp):
            temp1.append(chromosome1[i])
            temp2.append(chromosome2[i])
            chromosome1[i],chromosome2[i] = chromosome2[i],chromosome1[i]

        temp3 = []

        for i in temp1:
            for j in temp2:
                if(i == j):
                    temp3.append(i)

        for i in temp3:
            temp1.remove(i)
            temp2.remove(i)
        
        for i in range(len(temp1)):
            for j in range(bp,self.size):
                if(temp2[i] == chromosome1[j]):
                    chromosome1[j] = temp1[i]
                if(temp1[i] == chromosome2[j]):
                    chromosome2[j] = temp2[i]
        return chromosome1, chromosome2

    def crossover(self):
        if(self.crossover_policy == 'uniform'):
            for i in range(0,self.pop_size,2) :
                bp = random.randint(0, self.size - 1)
                self.chromosomes[i],self.chromosomes[i + 1] = self.crossover_two(self.chromosomes[i],self.chromosomes[i + 1],bp)
        elif(self.crossover_policy == 'fixed'):
            for i in range(0,self.pop_size,2) :
                bp = self.size // 2
                self.chromosomes[i],self.chromosomes[i + 1] = self.crossover_two(self.chromosomes[i],self.chromosomes[i + 1],bp)
        elif(self.crossover_policy == 'normal'):
            for i in range(0,self.pop_size,2) :
                bp = int(self.normal(self.size - 1))
                if bp < 0:bp = 0
                elif bp >= self.size:bp = self.size - 1
                self.chromosomes[i],self.chromosomes[i + 1] = self.crossover_two(self.chromosomes[i],self.chromosomes[i + 1],bp)  
        
    def mutate(self):
        for i in self.chromosomes:
            if(random.randint(0,int(1/self.mutate_probability)) == 0):
                i.reverse()
        return 
    def CalFitness(self):
        self.fitness = []
        for chromosome in self.chromosomes:
            temp = 0
            for i in range(self.size):
                temp += input[i][chromosome[i]]
            self.fitness.append(temp)
###########################
if __name__ == '__main__':
    with open('input.json','r') as f:
        data = json.load(f)
        for i,key in enumerate(data):
            input = data[key]
            ga_solver = GA(input,30,100,mutate_probability=0.01)
            assignment = ga_solver.best_chromosome # ⽤演算法得出的答案
            solver = Problem(input)
            print(f'題號:{i}')
            print('Assignment:', assignment) # print 出分配結果
            print('Cost:', solver.cost(assignment)) # print 出 cost 是多少
    