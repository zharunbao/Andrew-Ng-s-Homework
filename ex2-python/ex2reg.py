import file2matrix as f2m
import matplotlib.pyplot as plt
import numpy as np
import costfunctionreg as cfr
from sklearn.preprocessing import PolynomialFeatures
X,y = f2m.file2matrix('ex2data2.txt')
poly = PolynomialFeatures(degree=6, include_bias=False, interaction_only=False)
X_ploly = poly.fit_transform(X)
X_shape = np.shape(X_ploly)
m = X_shape[0]
n = X_shape[1]
X_1 = np.zeros((m,n+1))
X_1[:,0] = np.ones(m)
X_1[:,1:m] = X_ploly
'''theta = np.zeros((n+1,1))
lamda = 1
outloop = 1000000
alfa = 0.003
for i in range(outloop):
    cost,grad = cfr.costfunction(theta,X_1,y,lamda)
    theta = theta - alfa * grad
print(cost,grad)
np.save('theta.npy',theta)'''
theta = np.load('theta.npy')
'''fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(X[:,0],X[:,1],c = y,s=15)'''
plt.figure()
xx = np.arange(0,1.1,0.01).reshape((110,1))
yy = np.arange(0,1.1,0.01).reshape((110,1))
xy = np.hstack((xx,yy))
print(xy.shape)
f = poly.fit_transform(xy)
f1 = np.hstack((np.ones((110,1)),f))
f = np.dot(theta.reshape((1,28)),np.transpose(f1)).reshape((110,1))
print(f)
plt.contour(xx, yy, f,0,)
plt.show()
