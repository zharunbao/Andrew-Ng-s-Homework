import numpy as np
import Sigmoid


def costfunction(theta,X,y,lamda):
    m = y.shape[0]
    a = (-y).T * np.log(Sigmoid.sigmoid(X*theta))
    c = (np.ones((m,1))-y).T
    d = np.log(np.ones((m,1))-Sigmoid.sigmoid(X*theta))
    b = c * d
    J = (a -b)/m + lamda * (theta.T * theta - theta[0]**2)/(2*m)
    grad = X.T * (Sigmoid.sigmoid(X*theta)-y)/m
    grad = grad + lamda * theta/m
    grad[0] = X[:,0] * (Sigmoid.sigmoid(X*theta)-y)/m

    return J,grad