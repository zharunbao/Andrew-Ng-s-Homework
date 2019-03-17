import numpy as np
import Sigmoid

def SG(x):
    G = np.multiply(Sigmoid.sigmoid(x),1 - Sigmoid.sigmoid(x))
    return G
