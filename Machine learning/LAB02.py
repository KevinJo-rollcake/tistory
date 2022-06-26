#-*- coding: utf-8 -*-
#Machine Learning의 목적은 minimize Cost 
#Tensor flow
import tensorflow as tf

x_data = [1,2,3,4,5]
y_data = [1,2,3,4,5] #입력과 출력이 같은 모델, 예측이 쉽게
W = tf.Variable(2.9)
b = tf.Variable(0.5)

#Gradient Descent
learning_rate = 0.01

print('반복 횟수, W(초기 2.9, Target 1.0), b(초기 0.5, Target 0), 데이터에 대한 오차')

for i in range(100+1):
    with tf.GradientTape() as tape :
        hypothesis = W * x_data + b
        cost = tf.reduce_mean(tf.square(hypothesis - y_data))
    W_grad, b_grad = tape.gradient(cost,[W,b])
    W.assign_sub(learning_rate * W_grad) # A = A-B 꼴로 쭐여나감    
    b.assign_sub(learning_rate * b_grad)
    if i % 10 == 0 : 
        print('{:5}|{:10.4f}|{:10.4}|{:10.6f}'.format(i, W.numpy(), b.numpy(), cost))