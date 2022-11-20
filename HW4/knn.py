import numpy as np
import matplotlib.pyplot as plt

class KNN():
    def __init__(self,k):
        self.k = k

    def fit(self,train_X,train_Y):
        self.X = np.array(train_X)
        self.Y = np.array(train_Y)
        self.size = self.Y.shape[0]
        try:
            self.dim = self.X.shape[1]
        except:
            self.dim = 1
            self.X.resize((self.size,1))
        self.Y.resize((self.size,1))
        
    def predict(self,test_x):
        rst = []
        for x in test_x:
            dist = np.sqrt(np.sum((x - self.X) ** 2,axis=1))
            index = dist.argsort()
            rst.append(np.mean(self.Y[index[:self.k]]))
        return np.array(rst)
    def plot_2D(self):
        plot_x = np.linspace(0,10,100)
        plot_y = self.predict(plot_x)
        fig = plt.figure(figsize=(10,10))
        plt.scatter(self.X,self.Y)
        plt.plot(plot_x,plot_y,'r')
        plt.show()
    def plot_3D(self):
        fig = plt.figure(figsize=(10,8))
        ax = plt.subplot(projection='3d')
        ax.scatter(self.X[:,0],self.X[:,1],self.Y.T)
        x = np.linspace(-3, 3, 30)
        y = np.linspace(-3, 3, 30)
        X, Y = np.meshgrid(x, y)
        test_X = np.concatenate([X.reshape((900,1)),Y.reshape((900,1))],axis=1)
        Z = self.predict(test_X)
        Z = Z.reshape((30,30))
        ax.plot_surface(X,Y,Z,cmap='viridis')
        plt.show()   
        


if __name__ == '__main__':
    data = np.load('data1.npz')
    train_x = data['X']
    train_y = data['y']

    knn = KNN(k=50)
    accuracy = []
    for _ in range(20):
        index = [i for i in range(1000)]
        index = np.random.choice(index,1000,replace=False)
        knn.fit(train_x[index[:900]], train_y[index[:900]])
        predict = knn.predict(train_x[index[900:]])
        accuracy.append(np.average(np.abs(train_y[index[900:]] - predict)))
    print(np.average(accuracy))

    if(knn.dim == 1):
        knn.plot_2D()
    elif(knn.dim == 2):
        knn.plot_3D()