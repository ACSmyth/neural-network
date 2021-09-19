from .layer import Layer
from activation_functions import sigmoid, sigmoid_deriv

class NeuralNetwork:
  # dimensions is a list of ints
  def __init__(self, dimensions, activation_function=sigmoid, activation_function_deriv=sigmoid_deriv):
    self.dimensions = dimensions
    self.layers = []
    for i in range(len(dimensions)):
      if i < len(dimensions)-1:
        self.layers.append(Layer(dimensions[i], dimensions[i+1]))
      else:
        self.layers.append(Layer(dimensions[i], 0))
    self.activation_function = activation_function
    self.activation_function_deriv = activation_function_deriv

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
        cur_neuron.val = self.activation_function(cur_neuron.val)
    return self.layers[len(self.layers)-1].neurons

  def back_propagate_queue_weight_changes(self, correct_output, learning_rate = 0.05):
    def back_propagate_layer(layer, prev_layer, errors, layer_idx):
      prev_neuron_errors = [0 for q in range(len(prev_layer.neurons))]
      for i in range(len(layer.neurons)):
        neuron = layer.neurons[i]
        error = errors[i]
        for z in range(len(prev_layer.neurons)):
          prev_neuron = prev_layer.neurons[z]
          change = 2 * learning_rate * error * self.activation_function_deriv(neuron.val) * prev_neuron.val
          prev_neuron.queue_weight_changes(i, change)
          prev_neuron_errors[z] += (change / learning_rate) * prev_neuron.weights[i]
      return prev_neuron_errors

    new_errors = [correct_output[q] - self.layers[len(self.layers)-1].neurons[q].val\
            for q in range(len(correct_output))]
    for i in range(len(self.layers)-1, 0, -1):
      new_errors = back_propagate_layer(self.layers[i], self.layers[i-1], new_errors, i)


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
    # or could have "streaks" where if one
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
