import random
import math


def sigmoid(x):
	if x < 10**-10: return 0
	if x > 10**10: return 1
	return 1 / (1 + math.e**(-x))

def relu(x):
	return max(0, x)

class Neuron:
	def __init__(self, num_weights):
		self.val = 0.0
		self.weights = [((0.8 * random.random()) - 0.4) for q in range(num_weights)]
		self.queued_weight_changes = [0 for q in range(num_weights)]
		self.num_trials = [0 for q in range(num_weights)]

	def set_val(self, inp):
		self.val = inp

	def clear(self):
		self.val = 0.0

	def queue_weight_changes(self, i, change):
		self.queued_weight_changes[i] = ((self.queued_weight_changes[i] * self.num_trials[i]) + change)\
										/ (self.num_trials[i] + 1)
		self.num_trials[i] += 1

	def make_weight_changes(self):
		#print(self.queued_weight_changes)
		for q in range(len(self.weights)):
			self.weights[q] += self.queued_weight_changes[q]
		self.queued_weight_changes = [0 for q in range(len(self.weights))]
		self.num_trials = [0 for q in range(len(self.weights))]


class Layer:
	def __init__(self, num_neurons, next_layer_num_neurons):
		self.neurons = [Neuron(next_layer_num_neurons) for q in range(num_neurons)]

	def make_weight_changes(self):
		for n in self.neurons:
			n.make_weight_changes()


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

	def clear_neuron_vals(self):
		for layer in self.layers:
			for neuron in layer.neurons:
				neuron.clear()

	def forward_propagate(self, input):
		if self.dimensions[0] != len(input):
			raise Exception('Input must be length ' + str(self.dimensions[0]))
		# clear existing neuron values
		self.clear_neuron_vals()

		# set input layer
		inp_layer = self.layers[0]
		for q in range(len(input)):
			inp_layer.neurons[q].set_val(input[q])

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

	def back_propagate_queue_weight_changes(self, correct_output):
		learning_rate = 0.1

		def back_propagate_layer(layer, prev_layer, errors):
			prev_neuron_errors = [0 for q in range(len(prev_layer.neurons))]
			for i in range(len(layer.neurons)):
				neuron = layer.neurons[i]
				error = errors[i]
				for z in range(len(prev_layer.neurons)):
					prev_neuron = prev_layer.neurons[z]
					change = learning_rate * error * prev_neuron.val
					#print('change: ' + str(change))
					#print('error: ' + str(error))
					#print('prevneuronval: ' + str(prev_neuron.val))
					prev_neuron.queue_weight_changes(i, change)
					prev_neuron_errors[z] += change
			prev_neuron_errors = [val / len(layer.neurons) for val in prev_neuron_errors]
			return prev_neuron_errors

		new_errors = [correct_output[q] - self.layers[len(self.layers)-1].neurons[q].val\
						for q in range(len(correct_output))]
		for i in range(len(self.layers)-1, 0, -1):
			#print(new_errors)
			new_errors = back_propagate_layer(self.layers[i], self.layers[i-1], new_errors)


	# update weights after backpropagating
	def make_weight_changes(self):
		for l in self.layers:
			l.make_weight_changes()


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
			merge_layers(self.layers[i], other_net.layers[i], ret.layers[i])
		return ret

	def mutate(self):
		mut_rate = 0.5
		def mutate_layer(l):
			for i in range(len(l.neurons)):
				for j in range(len(l.neurons[i].weights)):
					mutation = mut_rate * (0.5 - random.random()) * 0.8
					l.neurons[i].weights[j] += mutation

		for l in self.layers:
			mutate_layer(l)

	def deep_clone(self):
		ret = NeuralNetwork(self.dimensions)
		for i in range(len(self.layers)):
			l = self.layers[i]
			for j in range(len(l.neurons)):
				for k in range(len(l.neurons[j].weights)):
					ret.layers[i].neurons[j].weights[k] = self.layers[i].neurons[j].weights[k]
		return ret

class GeneticAlgorithm:
	# dimensions: list of ints
	# population_size: int
	# input_func: game state -> neural network input array
	# run_game_func: agent1, input_func -> final game state
	# fitness_func: final game state -> fitness of agent
	def __init__(self, dimensions, population_size, rounds_per_agent, input_func, run_game_func, fitness_func):
		self.dimensions = dimensions
		self.population_size = population_size
		self.rounds_per_agent = rounds_per_agent
		self.input_func = input_func
		self.run_game_func = run_game_func
		self.fitness_func = fitness_func
		self.gen_count = 0
		self.prev_gen = None
		self.population = []
		self.best_agent = None

	def run_generation(self):
		if self.gen_count == 0:
			# generate initial population pool
			for i in range(self.population_size):
				# [net, avg fitness, num trials]
				self.population.append([NeuralNetwork(self.dimensions), 0, 0])

		# reset old fitness - although reset most in prev generation
		for agent in self.population:
			agent[1], agent[2] = 0, 0

		num_best_agents = self.population_size // 20
		best_agents = self.population[:num_best_agents]
		cloned_best_agents = [[agent[0].deep_clone(), agent[1], agent[2]] for agent in best_agents]

		for agent in self.population:
			net = agent[0]

			avg_fitness = 0
			# everyone plays by themselves
			for i in range(self.rounds_per_agent):
				final_game_state = self.run_game_func(net, self.input_func)
				fitness = self.fitness_func(final_game_state)
				avg_fitness += fitness
			avg_fitness /= self.rounds_per_agent
			agent[1] = avg_fitness
			agent[2] = self.rounds_per_agent

		# select / merge
		self.population.sort(key=lambda p: p[1], reverse=True) # sort by fitness, descending order
		self.best_agent = self.population[0]

		# randomly pick from pop -> best more likely
		def ran_idx():
			return int(4 * (-math.log(-random.random() + 1) + 0.1))

		num_random_to_add = self.population_size // 5
		new_population = []
		for i in range(len(self.population) - num_best_agents - num_random_to_add):
			i1 = ran_idx()
			i2 = ran_idx()
			if i1 >= len(self.population):
				i1 = 0
			if i2 >= len(self.population):
				i2 = 0
			merged = self.population[i1][0].merge(self.population[i2][0])
			new_population.append([merged, 0, 0])

		# mutate
		for agent in new_population:
			agent[0].mutate()

		# add completely random ones
		for q in range(num_random_to_add):
			new_population.append([NeuralNetwork(self.dimensions), 0, 0])

		# add best agents from prev generation unmutated
		new_population.extend(cloned_best_agents)

		#self.prev_gen = [[ag[0].deep_clone(), ag[1], ag[2]] for ag in self.population]
		self.population.clear()
		self.population.extend(new_population)

		self.gen_count += 1

	def get_best_agent(self):
		return self.best_agent

	#def get_prev_gen(self):
		#return self.prev_gen


class AdversarialGeneticAlgorithm:
	# dimensions: list of ints
	# population_size: int
	# input_func: game state -> neural network input array
	# run_game_func: agent1, agent2, input_func -> final game state
	# fitness_func: final game state -> two element list of fitness of both agents
	def __init__(self, dimensions, population_size, num_opponents, rounds_per_opponent, input_func, run_game_func, fitness_func):
		self.dimensions = dimensions
		self.population_size = population_size
		self.num_opponents = num_opponents
		self.rounds_per_opponent = rounds_per_opponent
		self.input_func = input_func
		self.run_game_func = run_game_func
		self.fitness_func = fitness_func
		self.gen_count = 0
		self.population = []
		self.best_agent = None

	def run_generation(self):
		if self.gen_count == 0:
			# generate initial population pool
			for i in range(self.population_size):
				# [net, avg fitness, num trials]
				self.population.append([NeuralNetwork(self.dimensions), 0, 0])

		# reset old fitness - although reset most in prev generation
		for agent in self.population:
			agent[1], agent[2] = 0, 0

		best_agents = self.population[:self.num_opponents]
		cloned_best_agents = [[agent[0].deep_clone(), agent[1], agent[2]] for agent in best_agents]

		for agent in self.population:
			net = agent[0]
			# run the process / evaluate fitness
			# everyone plays vs the top agents in the pool, except self
			for opponent_agent in best_agents:
				opponent_net = opponent_agent[0]
				# don't play against self
				if net is opponent_net: continue
				avg_net_fitness = 0
				avg_opponent_fitness = 0
				for i in range(self.rounds_per_opponent):
					final_game_state = self.run_game_func(net, opponent_net, self.input_func)
					net_fitness, opponent_fitness = self.fitness_func(final_game_state)
					avg_net_fitness += net_fitness
					avg_opponent_fitness += opponent_fitness
				avg_net_fitness /= rounds_per_opponent
				avg_opponent_fitness /= rounds_per_opponent

				def update_agent_fitness(ag, next_avg):
					old_avg = ag[1]
					old_n = ag[2]

					ag[2] += self.rounds_per_opponent
					ag[1] = ((old_avg * old_n) + (next_avg * self.rounds_per_opponent)) / ag[2]

				update_agent_fitness(agent, avg_net_fitness)
				update_agent_fitness(opponent_agent, avg_opponent_fitness)

			#print(str(int(100 * (self.population.index(agent)+1) / len(self.population))) + '%')

		# select / merge
		self.population.sort(key=lambda p: p[1], reverse=True) # sort by fitness, descending order
		self.best_agent = self.population[0]

		# randomly pick from pop -> best more likely
		def ran_idx():
			return int(4 * (-math.log(-random.random() + 1) + 0.1))

		new_population = []
		for i in range(len(self.population) - self.num_opponents):
			i1 = ran_idx()
			i2 = ran_idx()
			if i1 >= len(self.population):
				i1 = 0
			if i2 >= len(self.population):
				i2 = 0
			merged = self.population[i1][0].merge(self.population[i2][0])
			new_population.append([merged, 0, 0])


		# mutate
		for agent in new_population:
			agent[0].mutate()

		# add best agents from prev generation unmutated
		new_population.extend(cloned_best_agents)

		self.population.clear()
		self.population.extend(new_population)

		self.gen_count += 1

	def get_best_agent(self):
		return self.best_agent