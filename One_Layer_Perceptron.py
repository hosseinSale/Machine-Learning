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


def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_diff(x):
    return sigmoid(x) * (1 - sigmoid(x))

def feedforward_net(x,w):
    w_T = np.transpose(w)
    z = np.matmul(w_T,x)
    return sigmoid(z)
def backpropagation_net(pred_y, sample_y,x, w):
    error = pred_y - sample_y
    z = feedforward_net(x, w)
    diff_y_z = sigmoid_diff(z)
    gain = error * diff_y_z
    return (gain * x)

x = np.array([[1],[2],[3]])
w = np.random.rand(3,1)
pred_y = feedforward_net(x, w)
sample_y = np.array([1])
plot