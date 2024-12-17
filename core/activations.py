import numpy as np

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def leaky_relu(x, alpha=0.01):
        return np.maximum(alpha * x, x)

def leaky_relu_gradient(x, alpha=0.01):
    grad = np.ones_like(x)  # для x > 0 градиент = 1
    grad[x < 0] = alpha     # для x < 0 градиент = alpha
    return grad