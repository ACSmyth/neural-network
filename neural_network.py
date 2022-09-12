from .layer import Layer
from .activation_functions import relu, relu_deriv, sigmoid


class NeuralNetwork:
  # dimensions is a list of ints
  def __init__(self, dimensions, activation_function=relu, activation_function_deriv=relu_deriv):
    self.dimensions = dimensions
    self.layers = [
      Layer(
        dimensions[i], dimensions[i+1] if i < len(dimensions) - 1 else 0,
        activation_function=activation_function,
        activation_function_deriv=activation_function_deriv
      )
      for i in range(len(dimensions))
    ]
    self.activation_function = activation_function
    self.activation_function_deriv = activation_function_deriv

  def forward_propagate(self, input):
    self.layers[0].set_neuron_values(input)
    for i in range(len(self.layers)-1):
      output = self.layers[i].forward_propagate()
      self.layers[i+1].set_neuron_values(output)
    
    # Norm only the last layer to [0,1] using sigmoid
    # print(self.layers[len(self.layers)-1].neuron_matrix.tolist())
    self.layers[len(self.layers)-1].set_neuron_values(sigmoid(self.layers[len(self.layers)-1].neuron_matrix))
    return self.layers[len(self.layers)-1].neuron_matrix.tolist()
  
  def merge(self, other_nn):
    self_cloned = self.deep_clone()
    for i in range(len(self_cloned.layers)):
      self_cloned.layers[i].merge(other_nn.layers[i])
    return self_cloned
  
  def mutate(self):
    for layer in self.layers:
      layer.mutate()

  def deep_clone(self):
    nn_clone = NeuralNetwork(
      self.dimensions,
      activation_function=self.activation_function,
      activation_function_deriv=self.activation_function_deriv,
    )
    layer_clones = [layer.deep_clone() for layer in self.layers]
    nn_clone.layers = layer_clones
    return nn_clone
  
  def __repr__(self):
    return str(id(self)) + '\n' + ''.join(f'\t{var.title()}: {getattr(self, var)}\n' for var in vars(self))

  def to_json(self):
    json = {}
    for k, v in __dict__:
      to_json_op = getattr(self, 'to_json', None)
      if callable(to_json_op):
        json[k] = v.to_json()
      else:
        json[k] = v
    return json
