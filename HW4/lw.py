import numpy as np
import matplotlib.pyplot as plt
class LW:
    def __init__(self):
        pass
    
    def fit(self,train_x,train_y,band_width=0.08,node_num=100):
        self.X = np.array(train_x)
        self.Y = np.array(train_y)
        self.size = self.Y.shape[0]
        try:
            self.dim = self.X.shape[1]
        except:
            self.dim = 1
            self.X.resize((self.size,1))
        self.Y.resize((self.size,1))
        self.band_width = band_width
        self.node_num = node_num
        self.d = -2 * self.band_width ** 2


    def weight(self,point,X): 
        w = np.mat(np.eye(self.size)) 
        for i in range(self.size): 
            temp = X[i] - point 
            w[i, i] = np.exp(np.matmul(temp,temp.T)/self.d) 
        return w

    def predict(self,test_X): 
        X = np.append(self.X, np.ones(self.size).reshape(self.size,1), axis=1)
        theta = []
        pred = []
        for x in test_X:
            try:
                point = np.concatenate([x,np.ones((1))],axis=0)
            except:
                point = np.array([x,1])
            w = self.weight(point,X) 
            temp = np.linalg.pinv(X.T*(w * X))*(X.T*(w * self.Y)) 
            theta.append(temp)
            pred.append(np.matmul(point, temp)) 
        return np.array(theta), np.array(pred)

    def plot_2D(self):
        test_X = np.linspace(0, 10, self.node_num)
        test_X.resize((test_X.shape[0],1))
        theta,Y = self.predict(test_X)
        test_X = test_X.squeeze()
        Y = Y.squeeze()
        plt.figure(figsize=(10,10))
        plt.scatter(self.X, self.Y)
        plt.plot(test_X, Y, 'r')
        plt.show()
    def plot_3D(self):
        fig = plt.figure(figsize=(10,8))
        ax = plt.subplot(projection='3d')
        ax.scatter(self.X[:,0],self.X[:,1],self.Y.T,c='r')
        x = np.linspace(-3, 3, self.node_num)
        y = np.linspace(-3, 3, self.node_num)
        X, Y = np.meshgrid(x, y)
        test_X = np.concatenate([X.reshape((self.node_num ** 2,1)),Y.reshape((self.node_num ** 2,1))],axis=1)
        theta, Z = self.predict(test_X) 
        Z = Z.reshape((self.node_num,self.node_num))
        ax.plot_surface(X,Y,Z,cmap='viridis')
        plt.show()   
def accuracy(node_num,band_width):
    acc = []
    lw = LW()
    for _ in range(20):
        index = [i for i in range(1000)]
        index = np.random.choice(index,1000,replace=False)
        lw.fit(train_x[index[:900]], train_y[index[:900]],band_width=band_width,node_num=node_num)
        predict = lw.predict(train_x[index[900:]])[1]
        acc.append(np.average(np.abs(train_y[index[900:]] - predict.squeeze())))
    print(np.average(acc))
if __name__ == '__main__':
    data = np.load('data2.npz')
    train_x = data['X']
    train_y = data['y']

    lw = LW()
    node_num = 50
    band_width = 0.1
    accuracy(node_num,band_width)

    lw.fit(train_x,train_y,band_width=band_width,node_num=node_num)
    if(lw.dim == 1):
        lw.plot_2D()
    elif(lw.dim == 2):
        lw.plot_3D()