import plotdata
import numpy as np
import Sigmoid
X,label = plotdata.file2matrix('ex2data1.txt')
X_shape = np.shape(X)
m = X_shape[0]
n = X_shape[1]
X_1 = np.zeros((m,n+1))
X_1[:,0] = np.ones(m)
X_1[:,1:3] = X
theta = np.load('theta.npy')
y = Sigmoid.sigmoid(np.dot(X_1,theta))
y_01 = np.zeros(np.shape(label))
for i in range(np.size(y)):
    if y[i] >= 0.5:
        y_01[i] = 1
    else:
       y_01[i] = 0
num = 0
for i in range(np.size(label)):
    if y_01[i] == label[i]:
        num += 1
correct_rate = num/np.size(label)
print(correct_rate)

