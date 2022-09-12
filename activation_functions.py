import math
import numpy as np


def sigmoid(x):
	if x < -10**9: return 0
	if x > 10**9: return 1
	return 1 / (1 + math.e**(-20*(x - 0.5)))

def sigmoid_deriv(x):
	return x * (1 - x)

def relu(x):
	return np.maximum(x, 0)

def relu_deriv(x):
	return 0 if x < 0 else 1
