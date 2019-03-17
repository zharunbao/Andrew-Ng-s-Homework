import numpy as np
def ployfeatures(X,p):
    X = np.mat(X)
    #X_p = np.ones(np.shape(X))
    X_p = X
    if p == 1:
        return X
    else:
        for i in range(0, p - 1):
            X_p = np.multiply(X_p, X[:,0])
            X = np.hstack((X, X_p))
        return X
'''A = np.mat([[1],[2],[3]])
X = ployfeatures(A,5)
print(X)'''
