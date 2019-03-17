import numpy as np
import Sigmoid
def predict(theta1, theta2,X):
    m = np.shape(X)[0]
    X_1 = np.hstack((np.ones((m, 1)), X))
    A = Sigmoid.sigmoid(X_1 * theta1.T)
    A_1 = np.hstack((np.ones((m, 1)),A))
    B = Sigmoid.sigmoid(A_1 * theta2.T)
    pred = np.mat(np.zeros(np.shape(B)))
    '''for i in range(m):
        for j in range(B.shape[1]):
            if B[i, j] >= 0.5:
                pred[i, j] = 1
            else:
                pred[i, j] = 0'''
    for i in range(m):
        a = np.argmax(B[i])
        pred[i, a] = 1
    return pred