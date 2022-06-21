#-*- coding: utf-8 -*-
import numpy as np

X = np.array([1,2,3])
Y = np.array([1,2,3])

def cost_func(W, X, Y):
    c = 0
    for i in range(len(X)):
        c += (W * X[i] - Y[i]) ** 2
    return c / len(X)

for feed_W in np.linspace(-3, 5, num=15):#시작, 끝을 지정하고 num은 15개의 구간으로 지정
    curr_cost = cost_func(feed_W, X, Y)
    print("{:6.3f}|{:10.5f}".format(feed_W, curr_cost)) #C의 print할 때 변수 출력하는 것과 똑같다.
    #여기서 6.3f 는 6칸 공백 , 소수점 3자리 출력이다.