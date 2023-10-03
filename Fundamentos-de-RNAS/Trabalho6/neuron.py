import numpy as np

class Neuron():
    def __init__(self, input_dim, activation_function, index):
        self.index = index
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
    
    def set_delta_w(self, output_error, learning_rate):
      J_w = np.array([(self.prime_activation(self.vk) * self.input.T * self.layer_error)])
      J_ww = np.dot(J_w.T,  J_w)
      grad = np.dot(J_w.T, output_error)
      return(np.dot(np.linalg.inv(J_ww + np.dot(learning_rate, np.eye(self.input_dim))), grad))

    def set_delta_b(self, output_error, learning_rate):
        J_b = np.array([(self.prime_activation(self.vk) * self.layer_error)])
        J_bb = np.dot(J_b.T,  J_b)
        grad = np.dot(J_b.T, output_error)
        return(np.dot(np.linalg.inv(J_bb + np.dot(learning_rate, np.eye(1))), grad))

    def set_error_to_propag(self):
      return (self.prime_activation(self.vk) * self.weights.T)

    def backpropagation(self, output_error, learning_rate, layer_error):
        self.layer_error = layer_error[self.index] if (len(layer_error) > 1) else layer_error[0]
        self.delta_w = self.set_delta_w(output_error, learning_rate)
        self.delta_b = self.set_delta_b(output_error, learning_rate)
        self.error_to_propag = self.set_error_to_propag()
        self.weights -= self.delta_w.flatten()
        self.bias -= self.delta_b[0]
        self.print_backpropagation_parameters()
        return (self.error_to_propag)

    def print_backpropagation_parameters(self):
        print ("ΔW.T = ", self.delta_w.T, " | ΔB = ", self.delta_b.T, "| φ'(vk).W = ", self.error_to_propag.T)

