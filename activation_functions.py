def sigmoid(x):
	if x < -10**9: return 0
	if x > 10**9: return 1
	return 1 / (1 + math.e**(-x))

def sigmoid_deriv(x):
	return x * (1 - x)

def relu(x):
	return max(0, x)

def relu_deriv(x):
	return 0 if x < 0 else 1
