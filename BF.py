class Program:

    def __init__(self,input):
        self.input = input
        self.numTasks = len(input)
        self.minima = 0
        for i in range(self.numTasks):
            self.minima += self.input[i][i]
        self.a = []
        for i in range(self.numTasks):
            self.a.append(i)

    def cost(self,ans):
        totalTime = 0
        for task,agent in enumerate(ans):
            totalTime += self.input[task][agent]
        return totalTime
    
    def permutation(self,size):
        if size == 1:
            temp = 0
            for i in range(self.numTasks):
                temp += self.input[i][self.a[i]]
            if self.minima > temp:
                self.minima = temp

        for i in range(size):
            self.permutation(size-1);

            if(size & 1):
                self.a[0],self.a[size - 1] = self.a[size - 1],self.a[0]
            else:
                self.a[i],self.a[size - 1] = self.a[size - 1],self.a[i]

    
input =[[10,20,23,4],
        [15,13,6,25],
        [2,22,35,34],
        [12,3,14,17]]

solver = Program(input)
solver.permutation(len(input));
print(solver.minima)
