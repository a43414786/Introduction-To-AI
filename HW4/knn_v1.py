import numpy as np
import matplotlib.pyplot as plt

class KNN():
    def __init__(self,train_x:np.ndarray,train_y:np.ndarray,k=5):
        self.size = train_y.size
        self.dim = train_x.shape[1]
        self.w = np.ones((self.dim,1))
        print(self.w)
        self.b = 10.
        self.x,self.y = self.preprocess(train_x,train_y)
        print(self.x)
        print(self.y)
        
        

    def preprocess(self,train_x:np.ndarray,train_y:np.ndarray):
        temp = np.concatenate([train_x,train_y],axis = 1)
        temp = np.array(sorted(temp, key = lambda s: s[0]))
        x = np.array(temp[:,0])
        y = np.array(temp[:,1])
        x.resize((self.size,self.dim))
        y.resize((self.size,1))
        return x,y
        # print(x)
        # print(y)
        
        
    #     self.x,self.y = self.sort(train_x,train_y)
    #     self.x = self.sep_x()
    #     self.w = np.zeros(self.x.shape[0] + 1)
    #     self.lr = 0.0001
    #     # for _ in range(10000):
    #     #     self.update(self.x,self.y)
    #         # print(self.w)
    #     self.global_w = self.train(self.x,self.y,k)
    #     self.x_interval = []
    #     self.plot_y = []
    #     self.get_plot_xy(self.global_w,k)
        
    #     self.plot()    
    
    # def plot(self):
    #     if(self.x.shape[0] == 1):
    #         fig = plt.figure(figsize=(10,10))
    #         plt.scatter(self.x,self.y)
    #         plt.plot(self.x_interval,self.plot_y,'r',lw=2)
    #         plt.show()
    #     elif(self.x.shape[0] == 2):
    #         return
    #     else:
    #         return

    # def get_plot_xy(self,w,k):
    #     if(self.x.shape[0] == 1):
    #         counter = 0
    #         w_counter = 0
    #         self.x_interval = []
    #         self.plot_y = []
    #         while(counter < self.x.shape[1]):
    #             self.x_interval.append(self.x[0][counter])
    #             temp = self.internal_predict(self.x[0][counter],w[w_counter])
    #             self.plot_y.append(temp)
    #             if(counter + k - 1 < self.x.shape[1]):
    #                 self.x_interval.append(self.x[0][counter + k - 1])
    #                 temp = self.internal_predict(self.x[0][counter + k - 1],w[w_counter])
    #             else:
    #                 self.x_interval.append(self.x[0][self.x.shape[1] - 1])
    #                 temp = self.internal_predict(self.x[0][self.x.shape[1] - 1],w[w_counter])
    #             self.plot_y.append(temp)
    #             w_counter += 1
    #             counter += (k - 1)
    #     elif(self.x.shape[0] == 2):
    #         return
    #     else:
    #         return
        
    # def train(self,train_x:np.ndarray,train_y:np.ndarray,k:int):
    #     counter = 0
    #     w = []
    #     x = []
    #     y = []
    #     while(counter < train_x.shape[1]):
    #         for i in range(counter,counter + k):
    #             if(i < train_x.shape[1]):
    #                 x.append(train_x[0][i])
    #                 y.append(train_y[i])
    #         for _ in range(10000):
    #             self.update(x,y)
    #         w.append(np.copy(self.w))
    #         x = []
    #         y = []
    #         counter += (k - 1)
    #     return np.array(w)

    # def sort(self,train_x:np.ndarray,train_y:np.ndarray):
    #     train_x = np.array(train_x)
    #     temp = np.ndarray
    #     if(len(train_x.shape) == 1):
    #         train_x.resize((train_x.shape[0],1))
    #         train_y.resize((train_y.shape[0],1))
    #         temp = np.concatenate([train_x,train_y],axis = 1)
    #         temp = np.array(sorted(temp, key = lambda s: s[0]))
    #         train_x = np.resize(temp[:,0],(train_x.shape[0],1))
    #         train_y = temp[:,1]
    #     else:
    #         temp_x = train_x[:,0] * train_x[:,0]
    #         for i in range(1,train_x.shape[1]):
    #             temp_x += train_x[:,i] * train_x[:,i]
    #         temp_x = temp_x ** 0.5
    #         temp_x.resize((temp_x.shape[0],1))
    #         train_y.resize((train_y.shape[0],1))
    #         temp = np.concatenate([temp_x,train_x,train_y],axis = 1)
    #         temp = np.array(sorted(temp, key = lambda s: s[0]))
    #         train_x = temp[:,1:train_x.shape[1] + 1]
    #         train_y = temp[:,train_x.shape[1] + 1]
            
    #     return train_x,train_y
    # def sep_x(self):
    #     x = []
    #     for i in range(self.x.shape[1]):
    #         x.append(self.x[:,i])
    #     return np.array(x)
    # def predict(self,x):
    #     for i in range(len(self.global_w)):
    #         w = self.global_w[i]
    #         sum = 0
    #         if(i == 0):
    #             if(x < self.x_interval[1]):
    #                return w[0] + w[1] * x; 
    #         elif(i == len(self.global_w) - 1):
    #             if(x > self.x_interval[len(self.global_w) - 1]):
    #                return w[0] + w[1] * x
    #         else:
    #             if(self.x_interval[i * 2] < x and x < self.x_interval[i * 2 + 1]):
    #                 return w[0] + w[1] * x
    # def internal_predict(self,x,w):
    #     sum = w[0]
    #     try:
    #         for i in range(len(x)):
    #             sum += w[i + 1] * x[i]
    #     except:
    #         sum += w[1] * x
    #     return sum
    # def update(self,x,y):
    #     h = self.h(x)
    #     self.w[0] += self.lr * np.sum(y - h)
    #     for i in range(self.w.size - 1):
    #         self.w[i + 1] += self.lr * np.sum((y - h) * x[i])
    #     return

    # def h(self,x):
    #     rst = self.w[0]
    #     for i in range(1,self.w.size):
    #         rst += self.w[i] * x[i - 1]
    #     return rst
    


if __name__ == '__main__':
    train_data = np.load("data2.npz")
    train_x = train_data['X']
    try:
        train_x.shape[1]
    except:
        train_x.resize((train_x.shape[0],1))
    train_y = train_data['y']
    train_y.resize((train_y.shape[0],1))

    knn = KNN(train_x,train_y,k=100)
    # predict = knn.predict(train_x[100])
    # label = train_y[100]
    # print(predict,label)