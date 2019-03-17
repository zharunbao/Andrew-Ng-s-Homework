import numpy as np
import LinearRegCostFunction

def learningcurve(X,y,X_cv,y_cv,lamda):
    m = np.shape(X)[0]
    n = np.shape(X_cv)[0]
    X = np.mat(X)
    X = np.hstack((np.ones((m, 1)), X))
    dim = np.shape(X)[1]
    print(dim)
    y = np.mat(y)
    X_cv = np.mat(X_cv)
    X_cv = np.hstack((np.ones((n, 1)), X_cv))
    error_train =[]
    error_cv =[]
    for i in range(0,m):
        X1 = X[0:i+1,:]
        y1 = y[0:i+1]
        alfa = 0.1
        loop = 2000
        theta = np.zeros((dim,1))
        for a in range(0, loop):
            cost, grad = LinearRegCostFunction.CostFunction(X1, y1, theta, lamda)
            theta = theta - alfa * grad
            #print(grad,cost)
        AA = X1 * theta - y1
        er_train = 0
        for b in range(0, i+1):
            er_train += AA[b] ** 2
        er_train = er_train/(2*(i+1))
        # er_train = np.sum(np.multiply(X1 * theta - y1,X1 * theta - y1))/(2*(i+1))
        #er_cv = np.sum(np.multiply(X_cv * theta - y_cv,X_cv * theta - y_cv))/(2*n)
        BB = X_cv * theta - y_cv
        er_cv = 0
        for c in range(0,n):
            er_cv += BB[c] **2
        er_cv = er_cv/(2*n)
        error_train.append(er_train)
        error_cv.append(er_cv)
    return error_train,error_cv,theta