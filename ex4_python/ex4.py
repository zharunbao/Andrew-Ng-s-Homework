import numpy as np
import scipy.io as sio
import nnCostFunction as ncf
import predict as pd

#初始参数
num_labels = 10
m = 5000
input_layer_size  = 400
hidden_layer_size = 25
num_labels = 10
lamda = 1
loop = 500
alfa = 2

def vec(y):
    y_01 = np.zeros((num_labels, m))
    for i in range(1, num_labels):
        for j in range(m):
            if y[j] == i:
                y_01[i - 1, j] = 1
    for j in range(m):
        if y[j] == 10:
            y_01[9, j] = 1
    y_01 = np.mat(y_01).T
    return y_01


data = sio.loadmat('ex4data1.mat')
theta = sio.loadmat('ex4weights.mat')
'''theta1 = np.mat(theta['Theta1'])
theta2 = np.mat(theta['Theta2'])'''
theta1 = np.mat(np.random.uniform(-0.12,0.12,(25,401)))
theta2 = np.mat(np.random.uniform(-0.12,0.12,(10,26)))
X = np.mat(data['X'])
y = np.mat(data['y'])
y = vec(y)


for i in range(0,loop):
    cost,D1,D2 = ncf.nnCostFunction(theta1,theta2,X,y,lamda)
    theta1 = theta1 - alfa * D1
    theta2 = theta2 - alfa * D2
    print(str(cost) + ", 迭代次数：%d" % i)
np.save('theta1.npy',theta1)
np.save('theta2.npy',theta1)
pred = pd.predict(theta1,theta2,X)
correct_rate = 0
for i in range(m):
    if (y[i] == pred[i]).all():
        correct_rate += 1
print('正确率：' + str(correct_rate/m*100) + '%' )
print(cost)