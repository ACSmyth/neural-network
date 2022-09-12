import random
import numpy as np
from .activation_functions import relu, relu_deriv


class Layer:
	def __init__(self, num_neurons, next_layer_num_neurons, activation_function=relu, activation_function_deriv=relu_deriv):
		self.num_neurons = num_neurons
		self.next_layer_neurons = next_layer_num_neurons
		self.neuron_matrix = np.zeros((num_neurons, 1))
		self.weights_matrix = (np.random.rand(num_neurons, next_layer_num_neurons) - 0.5) / 10
		self.biases_matrix = (np.random.rand(next_layer_num_neurons) - 0.5) / 10
		self.activation_function = activation_function
		self.activation_function_deriv = activation_function_deriv

	def forward_propagate(self):
		output_neurons = np.dot(self.weights_matrix.transpose(), self.neuron_matrix)
		output_neurons = np.add(output_neurons, self.biases_matrix)
		output_neurons = self.activation_function(output_neurons)
		return output_neurons

	def set_neuron_values(self, neuron_values):
		self.neuron_matrix = np.array(neuron_values)
	
	def merge(self, other_layer):
		self_cloned = self.deep_clone()
		
		for i in range(self_cloned.weights_matrix.shape[0]):
			for j in range(self_cloned.weights_matrix.shape[1]):
				if random.random() < 0.5:
					self_cloned.weights_matrix[i,j] = other_layer.weights_matrix[i,j]
		
		for i in range(len(self_cloned.biases_matrix)):
			if random.random() < 0.5:
				self_cloned.biases_matrix[i] = other_layer.biases_matrix[i]

		return self_cloned
	
	def mutate(self):
		weights_mutation_matrix = (np.random.rand(*self.weights_matrix.shape) - 0.5) / (5)
		biases_mutation_matrix = (np.random.rand(len(self.biases_matrix)) - 0.5) / (5)

		self.weights_matrix = np.add(self.weights_matrix, weights_mutation_matrix)
		self.biases_matrix = np.add(self.biases_matrix, biases_mutation_matrix)
	
	def deep_clone(self):
		layer_clone = Layer(self.num_neurons, self.next_layer_neurons, activation_function=self.activation_function, activation_function_deriv=self.activation_function_deriv)
		layer_clone.neuron_matrix = self.neuron_matrix.copy()
		layer_clone.weights_matrix = self.weights_matrix.copy()
		layer_clone.biases_matrix = self.biases_matrix.copy()
		return layer_clone
		
	def __repr__(self):
		return str(id(self)) + '\n' + ''.join(f'\t{var.title()}: {getattr(self, var)}\n' for var in vars(self))
