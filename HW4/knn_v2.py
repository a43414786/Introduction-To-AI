import numpy as np
import matplotlib.pyplot as plt

class KNN:
    def __init__(self,k):
        self.k = k
    def fit(self,train_x,train_y):
        self.X = np.copy(train_x)
        try:
            self.X.shape[1]
        except:
            self.X.resize((self.X.shape[0],1))
        self.Y = np.copy(train_y)
        self.Y.resize((self.Y.shape[0],1))
        index = np.sum(np.sqrt(self.X),axis = 1).argsort()
        self.X = self.X[index]
        self.Y = self.Y[index]
        
        self.dim = self.X.shape[1]
        self.lr = 0.0001
        self.b = 0
        self.w = np.zeros((self.X.shape[1],1))
        for i in range(1000):
            print(self.b,self.w)
            self.update()

    def one_D(self):
        
        pass

    def update(self):
        h = np.matmul(self.X,self.w) + self.b
        y_h = self.Y - h
        self.b += self.lr * np.sum(y_h)
        for i in range(self.w.shape[0]):
            self.w[i,0] += self.lr * np.sum(y_h * self.X[:,i])
        return



    def findKNN(self,x):
        X = np.array(x)
        rst = []
        for x in X:
            dist = np.sum(np.sqrt(x - self.X) ** 2,axis = 1)
            index = dist.argsort()
            rst.append(self.X[index[:self.k]])
        return np.array(rst)

    # def train(self,train):
if __name__ == '__main__':
    data = np.load('data1.npz')
    train_x = data['X']
    train_y = data['y']
    # print(train_x)
    # print(train_y)
    knn = KNN(k=5)
    knn.fit(train_x,train_y)
    # x = train_x
    # y = knn.predict(train_x)
    # fig = plt.figure(10,10)
    
    # x = x.squeeze()
    # print(x.shape)
    # print(y.shape)
    # plt.figure()

    # plt.scatter(x,y)
    # plt.show()

        