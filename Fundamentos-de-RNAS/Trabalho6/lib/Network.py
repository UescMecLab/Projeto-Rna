import numpy as np
import DenseLayer

class Network():
    def __init__(self):
        self.layers = []
        self.forward_outputs = []
        self.input_signal = None
        self.loss = self.set_loss()
        self.prime = self.set_loss_prime()

    def set_loss(self, loss="MSE"):
        if loss == "MSE":
            return lambda y_true, y_pred: np.mean(np.power(y_true - y_pred, 2))

    def set_loss_prime(self, loss="MSE"):
        if loss == "MSE":
            return lambda y_true, y_pred: 2*(y_pred - y_true) / y_true.size

    def add(self, input_dim, output_dim, activation_function):
        new_layer = DenseLayer(input_dim, output_dim, activation_function)
        self.layers.append(new_layer)
        return (new_layer)

    def forward_propagation(self, input_signal):
        self.input_signal = input_signal
        forward_outputs = []
        y = input_signal
        for layer in self.layers:
            y = layer.forward_propagation(y)
            forward_outputs.append(y)
        self.forward_outputs = forward_outputs
        self.y_pred = y

    def backpropagation(self, y_desired, learning_rate=0.01):
        output_error = np.array([self.y_pred - y_desired])
        layer_error = np.array([[1]])
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