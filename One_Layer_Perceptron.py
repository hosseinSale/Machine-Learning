# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 13:42:11 2022

@author: ICS
"""
import numpy as np
import matplotlib.pyplot as plt

# Training of a one layer sigmoid perceptron

Data = [[1.00, 0.08, 0.72, 1.0],
        [1.00, 0.10, 1.00, 0.0],
        [1.00, 0.26, 0.58, 1.0],
        [1.00, 0.35, 0.95, 0.0],
        [1.00, 0.45, 0.15, 1.0],
        [1.00, 0.60, 0.30, 1.0],
        [1.00, 0.70, 0.65, 0.0],
        [1.00, 0.92, 0.45, 0.0]]

class One_Layer_perceptron:
    
    def __init__(self,training_Data,number_input):
        self.input_num = number_input
        self.Data = training_Data
    def preprocess(self):
        y_sample = []
        x = []
        for i in range(len(self.Data)):
            y_sample.append(self.Data[i][-1])
            x.append(self.Data[i][0:self.input_num+1])
            
            
        y_sample = np.array(y_sample)
        x = np.array(x)
        w = np.zeros(self.input_num+1)
        return y_sample, x, w
    
        
    def sigmoid(self,x):
        return 1/(1 + np.exp(-x))
    
    def sigmoid_diff(self,x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def feedforward_net(self,x,w):
        z   = np.dot(w,x)
        return self.sigmoid(z)
    
    def backpropagation_net(self,x,w,y_sample):
        pred_y   = self.feedforward_net(x,w)
        error    = -pred_y + y_sample
        diff_y_z = self.sigmoid_diff(pred_y)
        gain = error * diff_y_z
        return (gain * x)
    
    def predict(self,x,y_sample,num_iter,learning_rate):
        w = 0.01 *np.random.rand(self.input_num+1)
        
        delta_w = np.zeros(self.input_num +1)
        
        for i in range(num_iter):
            for j,k in zip(x,y_sample):
                w = w + learning_rate * delta_w 
                delta_w = self.backpropagation_net(j,w,k)
                
        return w
            
            
        

x = []
w = []
y_pred = []

net = One_Layer_perceptron(Data, 2)
y_sample, x, w = net.preprocess()
w = net.predict(x, y_sample, 10000, 0.01)
x1_1 = []
x2_1 = []
x1_0 = []
x2_0 = []
for i in Data:
    if i[3] == 1.0:
        x1_1.append(i[1])
        x2_1.append(i[2])
    else:
        x1_0.append(i[1])
        x2_0.append(i[2])


print(net.feedforward_net(x[0], w))
print(w)
t = np.linspace(0, 1,20)
x2_w = (-w[1]/w[2]) * t -(w[0]/w[2])
plt.plot(t,x2_w)
plt.scatter(x1_1,x2_1,c = 'r')
plt.show()
plt.scatter(x1_0,x2_0,c = 'b')
plt.show()