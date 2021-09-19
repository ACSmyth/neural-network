from .neuron import Neuron


class Layer:
	def __init__(self, num_neurons, next_layer_num_neurons):
		self.neurons = [Neuron(next_layer_num_neurons) for q in range(num_neurons)]

	def make_weight_changes(self):
		for n in self.neurons:
			n.make_weight_changes()
