import numpy as np

def CostFunction(X,y,theta,lamda):
    X = np.mat(X)
    y = np.mat(y)
    theta = np.mat(theta)
    m = np.shape(X)[0]
    dim = np.shape(theta)[0]
    grad = np.ones(np.shape(theta))
    A_1 = X
    Z_2 = A_1 * theta
    AA = Z_2-y
    cost = 0
    for i in range(0,m):
        cost += AA[i]**2
    for i in range(1,dim):
        cost += lamda * theta[i]
    #cost = np.sum(np.multiply(Z_2-y,Z_2-y)) + lamda * np.sum(np.multiply(theta[1:dim],theta[1:dim]))
    cost = cost / (2*m)
    grad[0] = A_1[:,0].T * (Z_2 - y)
    #print(A_1)
    for i in range(1,np.shape(theta)[0]):
        grad[i] = A_1[:, i].T * (Z_2 - y) + lamda * theta[i]
        #print(A_1[:, i].T)
        #print(grad)
    grad = grad /m
    return cost,grad
