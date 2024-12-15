import numpy as np

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def leaky_relu(x, alpha=0.01):
        return np.maximum(alpha * x, x)