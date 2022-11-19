import numpy as np
import matplotlib.pyplot as plt

class KNN():
    def __init__(self,train_x,train_y,k=5):
        self.X,self.Y = self.preprocess(train_x,train_y)
        self.dim = self.X.shape[0]
        self.k = k
        self.w = np.zeros(self.dim + 1)
        self.lr = 0.0001
        if(self.dim == 1):
            self.global_w = self.train(self.X,self.Y,k)
        
    
    def plot(self):
        if(self.dim == 1):
            counter = 0
            w_counter = 0
            self.X_interval = []
            self.plot_y = []
            while(counter < self.X.shape[1]):
                self.X_interval.append(self.X[0][counter])
                temp = self.internal_predict(self.X[0][counter],self.global_w[w_counter])
                self.plot_y.append(temp)
                if(counter + self.k - 1 < self.X.shape[1]):
                    self.X_interval.append(self.X[0][counter + self.k - 1])
                    temp = self.internal_predict(self.X[0][counter + self.k - 1],self.global_w[w_counter])
                else:
                    self.X_interval.append(self.X[0][self.X.shape[1] - 1])
                    temp = self.internal_predict(self.X[0][self.X.shape[1] - 1],self.global_w[w_counter])
                self.plot_y.append(temp)
                w_counter += 1
                counter += (self.k - 1)
            fig = plt.figure(figsize=(10,10))
            plt.scatter(self.X,self.Y)
            plt.plot(self.X_interval,self.plot_y,'r',lw=2)
            plt.show()
    def train(self,train_x:np.ndarray,train_y:np.ndarray,k:int):
        counter = 0
        w = []
        x = []
        y = []
        while(counter < train_x.shape[1]):
            for i in range(counter,counter + k):
                if(i < train_x.shape[1]):
                    x.append(train_x[0][i])
                    y.append(train_y[i])
            for _ in range(10000):
                self.update(x,y)
            w.append(np.copy(self.w))
            x = []
            y = []
            counter += (k - 1)
        return np.array(w)

    def preprocess(self,train_x:np.ndarray,train_y:np.ndarray):
        try:
            train_x.shape[1]
        except:
            train_x.resize((train_x.shape[0],1))
        index = np.sqrt(np.sum(train_x ** 2,axis = 1)).argsort()
        train_x = train_x[index]
        train_y = train_y[index]
        return train_x,train_y
    def predict(self,x):
        X = np.array(x)
        rst = []
        for x in X:
            dist = np.sqrt(np.sum((x - self.X) ** 2,axis = 1))
            index = dist.argsort()
            rst.append(np.mean(self.Y[index[:self.k]]))
        return np.array(rst)
    def internal_predict(self,x,w):
        sum = w[0]
        try:
            for i in range(len(x)):
                sum += w[i + 1] * x[i]
        except:
            sum += w[1] * x
        return sum
    def update(self,x,y):
        h = self.h(x)
        self.w[0] += self.lr * np.sum(y - h)
        for i in range(self.w.size - 1):
            self.w[i + 1] += self.lr * np.sum((y - h) * x[i])
        return

    def h(self,x):
        rst = self.w[0]
        for i in range(1,self.w.size):
            rst += self.w[i] * x[i - 1]
        return rst
    


if __name__ == '__main__':
    train_data = np.load("data1.npz")
    train_x = train_data['X']
    train_y = train_data['y']
    
    knn = KNN(train_x[:900],train_y[:900],k=20)
    predict = knn.predict(train_x[900:1000])
    label = train_y[900:1000]
    knn.plot()
    print(np.average(np.abs(predict-label)))