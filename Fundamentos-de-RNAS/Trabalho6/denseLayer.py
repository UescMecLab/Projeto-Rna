import numpy as np
from neuron import Neuron

class DenseLayer():
    def __init__(self):
        self.neurons = []

    def __init__(self, input_dim, output_dim, activation_function):
        self.neurons = [Neuron(input_dim, activation_function, i) for i in range(output_dim)]

    def forward_propagation(self, input_signal):
        outputs = []
        for neuron in self.neurons:
            output = neuron.process_output(input_signal)
            outputs.append(output)
        return outputs
    
    def backpropagation(self, output_error, learning_rate, layer_error):
        outputs = []
        for neuron in self.neurons:
            output = neuron.backpropagation(output_error, learning_rate, layer_error[0])
            outputs.append(output)
        return np.array(outputs)

    def set_weights(self, weights):
        for neuron, weight in zip(self.neurons, weights):
            neuron.weights = np.array(weight)

    def set_bias(self, bias):
        for neuron, b in zip(self.neurons, bias):
            neuron.bias = np.array(b)

    def get_weights(self):
        weights = [neuron.weights for neuron in self.neurons]
        return np.vstack(weights)

    def get_bias(self):
        bias = [neuron.bias for neuron in self.neurons]
        return np.vstack(bias)
