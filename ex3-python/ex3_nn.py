import numpy as np
import scipy.io as sio
import predict as pd
data = sio.loadmat('ex3data1.mat')
theta = sio.loadmat('ex3weights.mat')
theta1 = theta['Theta1']
theta2 = theta['Theta2']
X = data['X']
y = data['y']
num_labels = 10
m = 5000
y_01 = np.zeros((num_labels, m))
for i in range(1,num_labels):
    for j in range(m):
        if y[j] == i:
            y_01[i-1, j] = 1
for j in range(m):
    if y[j] == 10:
        y_01[9, j] = 1
y_01 = np.mat(y_01).T
pred = pd.predict(theta1,theta2,X)
correct_rate = 0
for i in range(m):
    if (y_01[i] == pred[i]).all():
        correct_rate += 1
print('正确率：' + str(correct_rate/m*100) + '%' )



