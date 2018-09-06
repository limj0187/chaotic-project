# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 08:16:34 2018

@author: USER
"""

import numpy
import random

#data
x = numpy.array([1,random.randint(0,50)/100,random.randint(0,50)/100])
x = numpy.vstack((x,[0,random.randint(50,100)/100,random.randint(50,100)/100]))
for i in range(49):
    x =numpy.vstack((x,[1,random.randint(0,50)/100,random.randint(0,50)/100]))
    x =numpy.vstack((x,[0,random.randint(50,100)/100,random.randint(50,100)/100]))

#print('data: ',x.shape)

def sigmoid(x):
    return (1/(1+numpy.exp(-x)))

def sigmoid_derivative(x):
    return x*(1-x)

#colours
y= numpy.array([x[0][0]])
for i in range(1,x.shape[0]):
    y = numpy.vstack((y,x[i][0]))

#print('colours:',y.shape)

#xpos,ypos
z= numpy.array([x[0][1],x[0][2]])
for i in range(1,x.shape[0]):
    z = numpy.vstack((z,[x[i][1],x[i][2]]))

#print('xpos,ypos:',z.shape)

#weights matrices initialisation
weights1 = numpy.array([[random.random(), random.random()],[random.random(), random.random()]])
weights2 = numpy.array([[random.random()],[random.random()]])
#print(weights1,weights2)
#print(weights1.shape,weights2.shape)

for i in range(999999):
    layer0 = z
    layer1 = sigmoid(numpy.dot(layer0,weights1))
    layer2 = sigmoid(numpy.dot(layer1,weights2))
    
    layer2_error = y - layer2
    
    layer2_delta = layer2_error*sigmoid_derivative(layer2)
    
    layer1_error = numpy.dot(layer2_delta,weights2.T)
    
    layer1_delta = layer1_error*sigmoid_derivative(layer1)
    
    weights2 += numpy.dot(layer1.T,layer2_delta)
    weights1 += numpy.dot(layer0.T,layer1_delta)

print(layer2)

for i in range(layer2.shape[0]):
    layer2[i][0] = round(layer2[i][0])

print(layer2)