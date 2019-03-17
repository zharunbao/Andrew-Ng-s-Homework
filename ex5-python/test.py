import numpy as np

theta = np.ones((2, 1))
print(theta)
the = theta
the[0] = 0
the[1] = 0
print(theta)
print(the[1])