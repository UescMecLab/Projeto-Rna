import numpy as np
from network import Network

def test_forward_backward():
  input_dim = 2
  output_dim = 2

  # index (i, j, k) == (layer, neuron, input_conection)
  input_signal = np.array([2, 4])           # [x1, x2]               --dim--> (1, 2)
  hidden_weights = np.array([[0.95, 0.96],  # neuron 11 [w111, w112] --dim--> (1, 2)
              [0.8, 0.85]])                 # neuron 12 [w121, w122] --dim--> (1, 2)
  hidden_bias = np.array([0.2 , 0.1])       # [b11, b12]             --dim--> (1, 2)  


  output_weights = np.array([[0.9, 0.8]])   # neuron 21 [w211, w212] --dim--> (1, 2)
  output_bias = np.array([0.3461])          # [b21]                  --dim--> (1, 1)

  network = Network()
  hidden_layer = network.add(input_dim, output_dim, 'tanh')
  output_layer = network.add(output_dim, 1, 'linear')

  hidden_layer.set_weights(hidden_weights)
  hidden_layer.set_bias(hidden_bias)
  output_layer.set_weights(output_weights)
  output_layer.set_bias(output_bias)

  network.forward_propagation(input_signal)
  network.print_forward_propagation()
  network.backpropagation(np.array([4]))
  network.forward_propagation(input_signal)
  network.print_forward_propagation()

def launchRandom():
  input_dim = 2
  output_dim = 2

  network = Network()
  network.add(input_dim, output_dim, 'tanh')
  network.add(output_dim, 1, 'linear')

  for _ in range(3):
    input_signal = np.random.rand(input_dim)
    network.forward_propagation(input_signal)
    network.print_forward_propagation()
    print()

     
# if __name__ == '__main__':
#   test_forward_backward()
#   launchRandom()  
