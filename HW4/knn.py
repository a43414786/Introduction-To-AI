import numpy as np
import matplotlib.pyplot as plt

class KNN():
    def __init__(self,k):
        self.k = k

    def fit(self,train_X,train_Y,lr = 0.0001):
        self.X = np.array(train_X)
        self.Y = np.array(train_Y)
        self.size = self.Y.shape[0]
        try:
            self.dim = self.X.shape[1]
        except:
            self.dim = 1
            self.X.resize((self.size,1))
        self.Y.resize((self.size,1))
        self.lr = lr
        
    def predict(self,test_x):
        rst = []
        try:
            test_x.shape[1]
        except:
            test_x.resize((test_x.shape[0],1))
        for x in test_x:
            dist = np.sqrt(np.sum((x - self.X) ** 2,axis=1))
            index = dist.argsort()
            w = np.ones((1,self.dim + 1))
            point = np.concatenate([np.ones((self.k,1)),self.X[index[:self.k]]],axis=1)
            for _ in range(1000):
                w = self.update(w,point,self.Y[index[:self.k]])
            x = np.resize(np.concatenate([[1],x],axis=0),(1,self.dim + 1))
            rst.append(np.matmul(x,w.T).squeeze())
            
        return np.array(rst)

    def update(self,w,x,y):
        h = np.matmul(x,w.T)
        y_h = y - h
        w += np.resize(self.lr * np.sum(y_h * x,axis=0),(1,self.dim + 1))
        return w
    def plot_2D(self):
        plot_x = np.linspace(-2,12,100)
        plot_y = self.predict(plot_x)
        fig = plt.figure(figsize=(10,10))
        plt.scatter(self.X,self.Y)
        plt.plot(plot_x,plot_y,'r')
        plt.show()
    def plot_3D(self):
        fig = plt.figure(figsize=(10,8))
        ax = plt.subplot(projection='3d')
        ax.scatter(self.X[:,0],self.X[:,1],self.Y.T)
        x = np.linspace(-4, 4, 30)
        y = np.linspace(-4, 4, 30)
        X, Y = np.meshgrid(x, y)
        test_X = np.concatenate([X.reshape((900,1)),Y.reshape((900,1))],axis=1)
        Z = self.predict(test_X)
        Z = Z.reshape((30,30))
        ax.plot_surface(X,Y,Z,cmap='viridis')
        plt.show()   
        
def accuracy(k,train_x,train_y):
    knn = KNN(k=k)
    size = train_y.shape[0]
    sep_point = int(size*0.9)
    acc = []
    for _ in range(20):
        index = [i for i in range(size)]
        index = np.random.choice(index,size,replace=False)
        knn.fit(train_x[index[:sep_point]], train_y[index[:sep_point]])
        predict = knn.predict(train_x[index[sep_point:]])
        acc.append(np.average(np.abs(train_y[index[sep_point:]] - predict)))
    print(np.average(acc))

if __name__ == '__main__':
    data = np.load('data2.npz')
    train_x = data['X']
    train_y = data['y']

    knn = KNN(k=5)
    accuracy(5,train_x,train_y)
    knn.fit(train_x, train_y)
    if(knn.dim == 1):
        knn.plot_2D()
    elif(knn.dim == 2):
        knn.plot_3D()