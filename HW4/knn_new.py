import numpy as np
import matplotlib.pyplot as plt

class KNN():
    def __init__(self,train_x,train_y,k=5):
        self.x,self.y = self.sort(train_x,train_y)
        self.x.resize(self.x.shape[1],self.x.shape[0])
        # self.y.reshape(self.y.shape[1],self.y.shape[0])
        # print(self.x.shape)
        # print(self.y.shape)
        
        self.w = np.zeros(self.x.shape[0] + 1)
        self.lr = 0.00001
        for _ in range(10000):
            # print(self.w)
            self.update(self.x,self.y)
        
        self.plot()    
    def plot(self):
        fig = plt.figure(figsize=(10,10))
        plt.scatter(self.x,self.y)
        x = [i for i in range(int(max(self.x[0])) + 3)]
        y = [self.predict(i) for i in x]
        plt.plot(x,y)
        plt.show()
    def sort(self,train_x:np.ndarray,train_y:np.ndarray):
        train_x = np.array(train_x)
        temp = np.ndarray
        if(len(train_x.shape) == 1):
            train_x.resize((train_x.shape[0],1))
            train_y.resize((train_y.shape[0],1))
            temp = np.concatenate([train_x,train_y],axis = 1)
            temp = np.array(sorted(temp, key = lambda s: s[0]))
            train_x = np.resize(temp[:,0],(train_x.shape[0],1))
            train_y = temp[:,1]
            # train_y = np.resize(temp[:,1],(train_y.shape[0],1))
        else:
            temp_x = train_x[:,0] * train_x[:,0]
            for i in range(1,train_x.shape[1]):
                temp_x += train_x[:,i] * train_x[:,i]
            temp_x = temp_x ** 0.5
            temp_x.resize((temp_x.shape[0],1))
            train_y.resize((train_y.shape[0],1))
            temp = np.concatenate([temp_x,train_x,train_y],axis = 1)
            temp = np.array(sorted(temp, key = lambda s: s[0]))
            train_x = temp[:,1:train_x.shape[1] + 1]
            train_y = temp[:,train_x.shape[1] + 1]
            # train_y = np.resize(temp[:,train_x.shape[1] + 1],(train_y.shape[0],1))
            
        return train_x,train_y
        
        # for i in range(train_x.shape[1]):
        #     rst.append(train_x[:,i])
        # return np.array(rst)


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
    


if __name__ == '__main__':
    train_data = np.load("data1.npz")
    train_x = train_data['X']
    train_y = train_data['y']
    
    knn = KNN(train_x,train_y,k=5)