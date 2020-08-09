import random
import math

def sigmoid(x):
	return 1 / (1 + math.e**(-x))


class Neuron:
	def __init__(self, num_weights):
		self.val = 0.0
		self.weights = [(0.8 * random.random()) - 0.4 for q in range(num_weights)]

	def set_val(self, inp):
		self.val = inp

	def clear(self):
		self.val = 0.0


class Layer:
	def __init__(self, num_neurons, next_layer_num_neurons):
		self.neurons = [Neuron(next_layer_num_neurons) for q in range(num_neurons)]


class NeuralNetwork:
	# dimensions is a list of ints
	def __init__(self, dimensions):
		self.dimensions = dimensions
		self.layers = []
		for i in range(len(dimensions)):
			if i < len(dimensions)-1:
				self.layers.append(Layer(dimensions[i], dimensions[i+1]))
			else:
				self.layers.append(Layer(dimensions[i], 0))

	# inp is a list of floats
	def set_input(self, inp):
		inp_layer = self.layers[0]
		for q in range(len(inp)):
			inp_layer.neurons[q].set_val(inp[q])

	def clear_neuron_vals(self):
		for layer in self.layers:
			for neuron in layer.neurons:
				neuron.clear()

	def forward_propagate(self):
		# clear existing neuron values
		self.clear_neuron_vals()
		# propagate
		for q in range(len(self.layers)-1):
			prev_layer = self.layers[q]
			cur_layer = self.layers[q+1]
			for i in range(len(cur_layer.neurons)):
				cur_neuron = cur_layer.neurons[i]
				for prev_neuron in prev_layer.neurons:
					cur_neuron.val += prev_neuron.val * prev_neuron.weights[i]
				cur_neuron.val = sigmoid(cur_neuron.val)
		return self.layers[len(self.layers)-1].neurons

	def merge(self, other_net):
		def merge_layers(l1, l2, ret_layer):
			for i in range(len(l1.neurons)):
				for j in range(len(l1.neurons[i].weights)):
					# randomly pick weights to keep
					# could also randomly pick entire neurons and keep all their weights
					if random.random() < 0.5:
						ret_layer.neurons[i].weights[j] = l1.neurons[i].weights[j]
					else:
						ret_layer.neurons[i].weights[j] = l2.neurons[i].weights[j]


		ret = NeuralNetwork(self.dimensions)
		# choose randomly from one or another
		# OR, to make more similar to crossover, could have "streaks" where if one
		# side randomly is selected then the next 5-10 nodes are from it
		for i in range(len(self.layers)):
			merge_layers(self.layers[i], other_net.layers[i], ret[i])
		return ret

	def mutate(self):
		mut_rate = 4.0
		def mutate_layer(l):
			for i in range(len(l.neurons)):
				for j in range(len(l.neurons[i].weights)):
					mutation = mut_rate * (0.5 - random.random()) * 0.8
					l.neurons[i].weights[j] += mutation

		for l in self.layers:
			mutate_layer(l)


net = NeuralNetwork([4, 12, 1])
res = net.forward_propagate()
print([e.val for e in net.forward_propagate()])
net.mutate()
print([e.val for e in net.forward_propagate()])


