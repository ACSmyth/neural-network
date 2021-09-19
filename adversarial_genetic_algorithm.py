from .neural_network import NeuralNetwork


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
