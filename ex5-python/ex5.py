import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import LinearRegCostFunction
import LearningCurve
import PloyFeatures
from sklearn import preprocessing

# 初始参数
lamda = 100
'''theta = np.ones((2,1))
alfa = np.array([[0.01],[0.001]])
loop = 1000'''

# 导入数据
data = sio.loadmat('ex5data1.mat')
X = data['X']
y = data['y']
Xcv = data['Xval']
ycv = data['yval']
Xtest = data['Xtest']
ytest = data['ytest']
'''Cost = []
I = []
for i in range(0, loop):
    cost, grad = LinearRegCostFunction.CostFunction(X, y, theta, lamda)
    theta = theta - np.multiply(alfa, grad)
    Cost.append(cost)
    I.append(i)
print(cost, grad)
np.save('theta.npy', theta)'''

# 数据与回归可视化
'''theta = np.load('theta.npy')
a =theta[0]
b =theta[1]
print(a, b)
y_line = b* X +a
plt.scatter(X, y)
plt.plot(X,y_line)
plt.show()'''

'''error_train,error_cv = LearningCurve.learningcurve(X,y,Xcv,ycv,lamda)
error_train = np.reshape(error_train,(12,1))
error_cv = np.reshape(error_cv,(12,1))
x = [i for i in range(1,13)]
plt.plot(x,error_cv)
plt.plot(x,error_train)
plt.show()'''

#ploy
p = 8
X_ploy = PloyFeatures.ployfeatures(X,p)
Xcv_ploy = PloyFeatures.ployfeatures(Xcv,p)
Xtest_ploy = PloyFeatures.ployfeatures(Xtest,p)
'''mu, sigma, X_norm = FeatureNormalize.featurenormalize(X_ploy)
print(X_ploy)'''

#归一化数据
scaler = preprocessing.StandardScaler().fit(X_ploy)
X_norm_ploy = scaler.transform(X_ploy)
Xcv_norm_ploy = scaler.transform(Xcv_ploy)
Xtest_norm_ploy = scaler.transform(Xtest_ploy)
#print(np.mean(Xcv_norm_ploy,axis=0))

error_train,error_cv,theta = LearningCurve.learningcurve(X_norm_ploy,y,Xcv_norm_ploy,ycv,lamda)
error_train = np.reshape(error_train,(12,1))
error_cv = np.reshape(error_cv,(12,1))
x = [i for i in range(1,13)]
#plt.plot(x,error_cv)
plt.scatter(X,y)

xx0 = np.mat(np.linspace(-50,50,20)).T
print(type(xx0))
xx = PloyFeatures.ployfeatures(xx0,p)
xx = scaler.transform(xx)
x_line = []
y_line = []
for i in range(0,20):
    y_line.append(np.dot(xx[i,:],theta[1:]) + theta[0])
print(xx)
plt.plot(xx0,y_line)
#plt.plot(x,error_train)
plt.show()




