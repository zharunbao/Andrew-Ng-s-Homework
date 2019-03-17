import numpy as np
import matplotlib.pyplot as plt

def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #get the number of lines in the file
    returnMat = np.zeros((numberOfLines,2))        #prepare matrix to return
    labeldata = np.zeros((numberOfLines,1))#prepare labels return
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split(',')
        returnMat[index,:] = listFromLine[0:2]
        labeldata[index] = int(listFromLine[-1])
        index += 1
    return returnMat,labeldata
data1mat,labeldata = file2matrix('ex2data1.txt')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(data1mat[:,0],data1mat[:,1],c=np.array(labeldata),s=15)
plt.show()
