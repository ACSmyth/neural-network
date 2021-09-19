import math
import random
from .neural_network import NeuralNetwork


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
