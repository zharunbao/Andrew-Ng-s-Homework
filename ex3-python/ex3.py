import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import costfunction as cf
import onevsall as ova
import Sigmoid
data = sio.loadmat('ex3data1.mat')
'''img = data['X'][1500].reshape((20,20)).T
plt.imshow(img)
plt.axis('off')
plt.show()
lamda = 3
theta = np.mat([[-2],[-1],[1],[2]])
X = np.arange(1,16).reshape((3,5)).T/10
X = np.hstack((np.ones((5,1)),X))
y = np.mat([[1],[0],[1],[0],[1]])
cost,grad = cf.costfunction(theta, X, y, lamda)
print(X,cost,grad)'''
X = data['X']
y = data['y']
#m = X.shape[0]
num_labels = 10
lamda = 0.1
m = 5000
y_01 = np.zeros((num_labels, m))
for i in range(num_labels):
    for j in range(m):
        if y[j] == i:
            y_01[i, j] = 1
for j in range(m):
    if y[j] == 10:
        y_01[0, j] = 1
y_01 = np.mat(y_01).T
print(y_01.shape)

'''theta = ova.onevsall(X, y, num_labels, lamda)
np.save('theta.npy',theta)
print(theta)'''
theta = np.load('theta.npy')
print(theta.shape)
X_1 = np.mat(np.hstack((np.ones((m, 1)), X)))
A = X_1 * theta.T
print(A.shape)
A = Sigmoid.sigmoid(A)
for i in range(m):
    for j in range(num_labels):
        if A[i,j] >= 0.5:
            A[i,j] = 1
        else:
            A[i,j] = 0
num = 0
for i in range(m):
    if A[i].any() ==y_01[i].any():
        num += 1
correct_rate = num/m
print(correct_rate)



