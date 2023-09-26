import numpy as np

class Neuron():
    def __init__(self, input_dim, activation_function):
        self.input_dim = input_dim
        self.weights = np.random.rand(input_dim)
        self.bias = np.array(np.random.random())
        self.activation_function = self.set_activation_function(activation_function)
        self.prime_activation = self.set_prime_activation(activation_function)

    def summing_junction(self):
        return(np.dot(self.weights, self.input) + self.bias)

    def process_output(self, input_signal):
      self.input = np.array(input_signal)
      self.vk = self.summing_junction()
      return (self.activation_function(self.vk))

    def set_activation_function(self, activation_function):
        if (activation_function == 'tanh'):
            return lambda x: np.tanh(x)
        if (activation_function == 'linear'):
            return lambda x: x

    def set_prime_activation(self, activation):
        if activation == 'tanh':
            return lambda x: 1 - np.tanh(x) ** 2
        elif activation == 'linear':
            return lambda x: 1    
    
    def set_delta_w(self, output_error, learning_rate, layer_error):
      J_w = np.array([(-self.prime_activation(self.vk) * self.input.T * layer_error)])
      J_ww = np.dot(J_w.T,  J_w)
      grad = np.dot(J_w.T, output_error)
      return(np.dot(np.linalg.inv(J_ww + np.dot(learning_rate, np.eye(self.input_dim))), grad))

    def set_delta_b(self, output_error, learning_rate, layer_error):
        J_b = np.array([(-self.prime_activation(self.vk) * layer_error)])
        J_bb = np.dot(J_b.T,  J_b)
        grad = np.dot(J_b.T, output_error)
        return(np.dot(np.linalg.inv(J_bb + np.dot(learning_rate, np.eye(1))), grad))

    def set_error_to_propag(self):
      return (-self.prime_activation(self.vk) * self.weights.T)

    def backpropagation(self, output_error, learning_rate, layer_error):
        delta_w = self.set_delta_w(output_error, learning_rate, layer_error)
        delta_b = self.set_delta_b(output_error, learning_rate, layer_error)
        error_to_propag = self.set_error_to_propag()
        print ("ΔW.T = ", delta_w.T, " | ΔB = ", delta_b.T, "| φ'(vk).W = ", error_to_propag.T)
        return (error_to_propag)

class DenseLayer():
    def __init__(self):
        self.neurons = []

    def __init__(self, input_dim, output_dim, activation_function):
        self.neurons = [Neuron(input_dim, activation_function) for _ in range(output_dim)]

    def forward_propagation(self, input_signal):
        outputs = []
        for neuron in self.neurons:
            output = neuron.process_output(input_signal)
            outputs.append(output)
        return outputs
    
    def backpropagation(self, output_error, learning_rate, layer_error):
 #       outputs = []
        for neuron in self.neurons:
            output = neuron.backpropagation(output_error, learning_rate, layer_error)
#            outputs.append(output)
        return output

    def set_weights(self, weights):
        for neuron, weight in zip(self.neurons, weights):
            neuron.weights = np.array(weight)

    def set_bias(self, bias):
        for neuron, b in zip(self.neurons, bias):
            neuron.bias = np.array(b)

 #empilhar matrizes em sequência vertical
    def get_weights(self):
        weights = [neuron.weights for neuron in self.neurons]
        return np.vstack(weights)

    def get_bias(self):
        bias = [neuron.bias for neuron in self.neurons]
        return np.vstack(bias)

class Network():
    def __init__(self):
        self.layers = []
        self.forward_outputs = []
        self.input_signal = None

    def add(self, input_dim, output_dim, activation_function):
        new_layer = DenseLayer(input_dim, output_dim, activation_function)
        self.layers.append(new_layer)
        return (new_layer)

    def add_layer(self, layer):
        self.layers.append(layer)
        return (layer)

    def forward_propagation(self, input_signal):
        self.input_signal = input_signal
        forward_outputs = []
        y = input_signal
        for layer in self.layers:
            y = layer.forward_propagation(y)
            forward_outputs.append(y)
        self.forward_outputs = np.array(forward_outputs, dtype=type(forward_outputs))
        self.y_pred = y

    def backpropagation(self, y_desired, learning_rate=0.01):
        output_error = np.array([self.y_pred - y_desired])
        layer_error = 1
        for layer in reversed(self.layers):
            layer_error = layer.backpropagation(output_error, learning_rate, layer_error)

    def print_forward_propagation(self):
        print ("--------------------Forward Propagation----------------------")
        tabs = "\t"
        print("[Input Signal] ---→ ", self.input_signal)
        for i, output in enumerate(self.forward_outputs):
            layer_name = "Output Layer" if i == len(self.forward_outputs) - 1 else f"Hidden Layer {i}"
            print(tabs, "|\n", tabs, f"↳[{layer_name}] ---→ ", output)
        tabs += "\t"


def launchExample():
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

if __name__ == '__main__':
    launchExample()