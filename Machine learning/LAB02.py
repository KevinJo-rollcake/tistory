#Machine Learning의 목적은 minimize Cost 
#Tensor flow

x_data = [1,2,3,4,5]
y_data = [1,2,3,4,5]
#입력과 출력이 같은 모델, 예측이 쉽게

W = tf.Variable(2.9)
b = tf.Variable(0.5)

hypothesis = W * x_data + b