import numpy as np
import matplotlib.pyplot as plt

class KNN():
    def __init__(self,train_x,train_y,k=5):
        train_x,train_y = self.sort(train_x,train_y)
        return
        self.x = self.preprocess_x(train_x)
        self.y = np.array(train_y)
        self.w = np.zeros(self.x.shape[0] + 1)
        self.lr = 0.00001


        # for _ in range(10000):
        #     self.update(self.x,self.y)
        
        # self.plot()    
    def plot(self):
        fig = plt.figure(figsize=(10,10))
        plt.scatter(self.x,self.y)
        x = [i for i in range(int(max(self.x[0])) + 3)]
        y = [self.predict(i) for i in x]
        plt.plot(x,y)
        plt.show()
    def sort(self,x,y):
        

    def predict(self,x):
        sum = self.w[0]
        try:
            for i in range(len(x)):
                sum += self.w[i + 1] * x[i]
        except:
            sum += self.w[1] * x
        return sum
    def update(self,x,y):
        y_h = y - self.h(x)
        self.w[0] += self.lr * np.sum(y_h)
        for i in range(self.w.size - 1):
            self.w[i + 1] += self.lr * np.sum(x[i] * y_h)
        return

    def h(self,x):
        rst = self.w[0]
        for i in range(1,self.w.size):
            # print(i)
            rst += self.w[i] * x[i - 1]
        return rst
    
    def preprocess_x(self,train_x)->np.ndarray:
        train_x = np.array(train_x)
        rst = []
        if(len(train_x.shape) == 1):
            train_x.resize((train_x.shape[0],1))
        for i in range(train_x.shape[1]):
            rst.append(train_x[:,i])
        return np.array(rst)


if __name__ == '__main__':
    train_data = np.load("data1.npz")
    train_x = train_data['X']
    train_y = train_data['y']
    
    knn = KNN(train_x,train_y,k=5)