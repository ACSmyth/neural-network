from layer import Layer
from activation_functions import sigmoid, sigmoid_deriv

class NeuralNetwork:
  # dimensions is a list of ints
  def __init__(self, dimensions, activation_function=sigmoid, activation_function_deriv=sigmoid_deriv):
    self.dimensions = dimensions
    self.layers = [
      Layer(dimensions[i], dimensions[i+1] if i < len(dimensions) - 1 else 0)
      for i in range(len(dimensions))
    ]
    self.activation_function = activation_function
    self.activation_function_deriv = activation_function_deriv

  def forward_propagate(self, input):
    self.layers[0].set_neuron_values(input)
    for i in range(len(self.layers)-1):
      output = self.layers[i].forward_propagate()
      self.layers[i+1].set_neuron_values(output)
    return output
