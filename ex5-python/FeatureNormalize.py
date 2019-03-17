import numpy as np
from sklearn import preprocessing

def featurenormalize(X):
    mu = np.mean(X,axis=0)
    sigma = np.std(X, axis=0)
    #X_norm = preprocessing.scale(X)
    X_norm = (X - X.mean())/X.std()
    return mu,sigma,X_norm
X = np.mat([[2,3],[1,2]])
mu,sigma,X_norm = featurenormalize(X)
#print(mu,sigma,np.mean(X_norm,axis=0))

