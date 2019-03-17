import numpy as np
import Sigmoid
import SigmoidGradient

def nnCostFunction(theta1,theta2,X,y,lamda):
    m = np.shape(X)[0]
    A_1 = np.hstack((np.ones((m, 1)), X))  # 加偏置
    Z_2 = A_1 * theta1.T
    A_2 = Sigmoid.sigmoid(Z_2)
    A_2 = np.hstack((np.ones((m, 1)), A_2))  # 加入偏置
    Z_3 = A_2 * theta2.T
    A_3 = Sigmoid.sigmoid(Z_3)
    log_A_3 = np.log(A_3)
    log_a = np.log(np.ones((m,1)) - A_3)
    a = np.multiply(-y,log_A_3) - np.multiply(np.ones((m,1)) - y,log_a)
    cost = np.sum(a)/m   #没有正则项的cost
    #含有正则项的cost
    cost = cost + lamda * (np.sum(np.multiply(theta1[:,1:m],theta1[:,1:m])) + np.sum(np.multiply(theta2[:,1:m],theta2[:,1:m])))/(2*m)
    D1 = np.zeros(np.shape(theta1))
    D2 = np.zeros(np.shape(theta2))
    for i in range(0,m):
        a1 = A_1[i,:].T
        z2 = Z_2[i,:].T
        z2 = np.vstack((1,z2))
        a2 = A_2[i,:].T
        z3 = Z_3[i,:].T
        a3 = A_3[i,:].T
        y1 = y[i,:].T
        delta3 = a3 - y1
        delta2 = np.multiply((theta2.T)*delta3,SigmoidGradient.SG(z2))
        D1 += delta2[1:m] * a1.T
        D2 += delta3 * a2.T
    D1 = D1 / m + lamda * theta1 /m
    D2 = D2 / m + lamda * theta2 /m
    return cost,D1,D2