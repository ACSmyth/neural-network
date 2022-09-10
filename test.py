import unittest
import numpy as np
from neural_network import NeuralNetwork
from hypothesis import given, strategies as st


class Tests(unittest.TestCase):
  @given(st.integers(min_value=0, max_value=10**5))
  def test_neural_network_multiplies(self, n):
    nn = NeuralNetwork((2,3,1))
    nn.layers[0].weights_matrix = np.zeros((2,3))
    nn.layers[0].weights_matrix[0,0] = n
    nn.layers[0].biases_matrix = np.zeros(3)
    nn.layers[1].weights_matrix = np.zeros(3)
    nn.layers[1].weights_matrix[0] = n
    nn.layers[1].biases_matrix = np.zeros(1)
    out = nn.forward_propagate([n, 0])

    return self.assertEqual(out[0], n**3)

unittest.main()
