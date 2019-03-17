import numpy as np
import costfunction as cf
def onevsall(X, y, num_labels, lamda):
    m = X.shape[0]
    n = X.shape[1]
    X_1 = np.hstack((np.ones((m, 1)), X))
    theta = np.mat(np.zeros((num_labels,n+1)))
    y_01 = np.mat(np.zeros((num_labels,m)))
    for i in range(num_labels):
        for j in range(m):
            if y[j] == i:
                y_01[i,j] = 1
    for j in range(m):
        if y[j] == 10:
            y_01[0, j] = 1

    outloop = 100000
    alfa = 0.003
    for i in range(num_labels):
        for j in range(outloop):
            cost, grad = cf.costfunction(theta[i].T, X_1, y_01[i].T, lamda)
            theta[i] = theta[i] - alfa * grad.T
        print('迭代次数：%d' % i)
    return theta