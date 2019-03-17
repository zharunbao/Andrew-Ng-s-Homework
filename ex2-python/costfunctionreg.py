import numpy as np
import Sigmoid
def costfunction(theta,X,y,lamda):
    m = len(y)
    J = 0
    grad = np.zeros(np.shape(theta))
    a = np.dot((-y).reshape(1,m),np.log(Sigmoid.sigmoid(np.dot(X, theta))))
    c = (np.ones((m,1))-y).reshape(1,m)
    d = np.log(np.ones((m,1))-Sigmoid.sigmoid(np.dot(X,theta)))
    b = np.dot(c,d)
    J = (a -b)/m + lamda * np.dot(np.transpose(theta),theta)/(2*m)
    grad = np.dot((Sigmoid.sigmoid(np.dot(X, theta))-y).reshape(1,m),X)/m
    grad = grad.reshape(np.shape(theta))
    grad = grad + lamda * theta/m
    return J,grad