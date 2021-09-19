import random


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
