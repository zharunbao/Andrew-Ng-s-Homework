import Sigmoid
import numpy as np
import plotdata
import CostFunction
import matplotlib.pyplot as plt
X,y = plotdata.file2matrix('ex2data1.txt')
X_shape = np.shape(X)
m = X_shape[0]
n = X_shape[1]
X_1 = np.zeros((m,n+1))
X_1[:,0] = np.ones(m)
X_1[:,1:3] = X
theta = np.zeros((n+1,1))
outloop = 3000000
alfa = 0.003
cost_list = np.zeros((int(outloop/100),2))
for i in range(outloop):
    cost,grad = CostFunction.costfunction(theta,X_1,y)
    theta = theta - alfa*grad
    if i%100 == 0:
        cost_list[int(i/100),0] = i
        cost_list[int(i/100),1] =cost
print(cost,grad,theta)
np.save('theta.npy',theta)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(cost_list[:,0],cost_list[:,1],s=10)
plt.show()
'''theta = np.load('theta.npy')
y = Sigmoid.sigmoid(np.dot([[1,45,85]],theta))
print(y)'''
