import numpy as np
import random
import matplotlib.pyplot as plt

class LWF:
    def __init__(self):
        data = np.load("data1.npz")
        self.a1 = data['X']
        self.a2 = data['y']
        try:
            self.a1.shape[1]
        except:
            index = self.a1.argsort()
            self.a1 = self.a1[index]
            self.a2 = self.a2[index]            
            self.a1.resize((self.a1.shape[0],1))
        
        self.a1 = np.concatenate([np.ones(shape=(self.a1.shape[0],1)),self.a1],axis = 1)
        print(self.a1)
        # self.a1,self.a2 = self.genData(100,10,0.6)

        print(self.a1.shape)
        print(self.a2.shape)

        a3 = []
        for i in self.a1:
            pdf = self.predict(self.a1,self.a2,i,1)
            a3.append(pdf.tolist()[0])
        plt.figure(figsize=(10,10))
        plt.plot(self.a1[:,1],self.a2,"x")
        plt.plot(self.a1[:,1],a3,"r--")
        plt.show()
    def kernel(self,x,x0,c,a=1.0):
        diff = x - x0
        dot_product = diff * diff.T
        return a* np.exp(dot_product / (-2.0 * c ** 2))

    def get_weights(self,train,data,c = 1.0):
        x = np.mat(train)
        n_rows = x.shape[0]
        weights = np.mat(np.eye(n_rows))
        for i in range(n_rows):
            weights[i,i] = self.kernel(data,x[i],c)
        return weights

    def predict(self,train_in,train_out,data,c=1.0):
        weights = self.get_weights(train_in,data,c=c)
        x = np.mat(train_in)
        y = np.mat(train_out).T
        xt = x.T * (weights * x)
        betas = xt.I * (x.T * (weights * y))
        return data * betas

    def genData(self,numPoints,bias,variance):
        x = np.zeros(shape=(numPoints,2))
        y = np.zeros(shape=numPoints)
        for i in range(0,numPoints):
            x[i][0] = 1
            x[i][1] = i
            y[i] = bias + i * variance + random.uniform(0,1) * 20
        return x,y

if __name__ == '__main__':
    lwf = LWF()
    # print(lwf.a1)
    # print(lwf.a2)