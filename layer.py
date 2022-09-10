import numpy as np
from .activation_functions import relu, relu_deriv


class Layer:
	def __init__(self, num_neurons, next_layer_num_neurons, activation_function=relu, activation_function_deriv=relu_deriv):
		self.num_neurons = num_neurons
		self.next_layer_neurons = next_layer_num_neurons
		self.neuron_matrix = np.zeros((num_neurons, 1))
		self.weights_matrix = np.random.rand(num_neurons, next_layer_num_neurons)
		self.biases_matrix = np.random.rand(next_layer_num_neurons)
		self.activation_function = activation_function
		self.activation_function_deriv = activation_function_deriv

	def forward_propagate(self):
		output_neurons = np.dot(self.weights_matrix.transpose(), self.neuron_matrix)
		output_neurons = np.add(output_neurons, self.biases_matrix)
		output_neurons = self.activation_function(output_neurons)
		return output_neurons

	def set_neuron_values(self, neuron_values):
		self.neuron_matrix = np.array(neuron_values)
	
	def deep_clone(self):
		layer_clone = Layer(self.num_neurons, self.next_layer_neurons, activation_function=self.activation_function, activation_function_deriv=self.activation_function_deriv)
		layer_clone.neuron_matrix = self.neuron_matrix.copy()
		layer_clone.weights_matrix = self.weights_matrix.copy()
		layer_clone.biases_matrix = self.biases_matrix.copy()
		return layer_clone
