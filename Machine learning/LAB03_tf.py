#-*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

X = np.array([1,2,3])
Y = np.array([1,2,3])

def cost_func(W, X, Y):
    hypothesis = X * W
    return tf.reduce_mean(tf.square(hypothesis - Y))

W_values = np.linspace(-3, 5, num=15)
cost_values = []

for feed_W in W_values:
    curr_cost = cost_func(feed_W, X, Y)
    cost_values.append(curr_cost)
    print("{:6.3f}|{:10.5f}".format(feed_W, curr_cost)) #C의 print할 때 변수 출력하는 것과 똑같다.
    #여기서 6.3f 는 6칸 공백 , 소수점 3자리 출력이다.