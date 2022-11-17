import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
class Linear_Regression:
    def __init__(self,data):
        self.x = data['X'][:10]
        self.y = data['y'][:10]
        self.dim = np.size(self.x[0])
        self.size = len(self.x)
        self.w = np.ones(self.dim + 1)
        self.lr = 0.0000001

        self.distance = self.combine()
        self.distance = sorted(self.distance, key = lambda s: s[0])

        self.x = self.sep_x()
        
        for _ in range(1000):
            self.update()

        x = np.array(range(12))

        fig = plt.figure(figsize=(10,10))
        ax = fig.gca(projection='3d')
        X, Y, Z = axes3d.get_test_data(0.05)
        print(X)
        print(Y)
        print(Z)
        
        ax.plot_surface(X, Y, Z, cmap='seismic')
        plt.show()
        # print(self.w)
        # print(self.size)
        # print(self.dim)
        # print(self.x)
        # print(rst)

    def combine(self):
        if(self.dim == 1):
            rst = []
            for i in range(self.size):
                rst.append([self.x[i],self.y[i]])
            return np.array(rst)
        

    def sep_x(self):
        if(self.dim == 1):return np.reshape(self.x,(1,self.size))
        rst = np.zeros((self.dim,self.size))
        for i in range(self.size):
            for j in range(self.dim):
                rst[j,i] = self.x[i,j]
        return rst

    def update(self):
        y_h = self.y - self.h(self.x)
        self.w[0] += self.lr * np.sum(y_h)
        for i in range(1,self.dim + 1):
            self.w[i] += self.lr * np.sum(y_h * self.x[i - 1])
        print(self.w)
        

    def h(self,x):
        rst = self.w[0]
        for i in range(1,self.dim + 1):
            rst += self.w[i] * x[i - 1]
        return rst


data = np.load('data1.npz')
regressor = Linear_Regression(data)
# x = data['X']
# y = data['y']
# plt.scatter(x,y)
# plt.show()