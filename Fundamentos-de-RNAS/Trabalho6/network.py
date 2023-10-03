import numpy as np
from denseLayer import DenseLayer

class Network():
    def __init__(self):
        self.layers = []
        self.forward_outputs = []
        self.input_signal = None
        self.loss = self.set_loss()
        self.prime = self.set_loss_prime()

    # runs forward through test data
    def predict(self, x_test):
        samples = len(x_test)
        result = []
        for sample in range(samples):
            input_signal = x_test[sample]
            self.forward_propagation(input_signal)
            result.append(self.y_pred)
        return (np.array(result).flatten())

    def CheckCost(self, tol=0.001):
        if (len(self.HistCost) <= 2):
            return False
        return (abs(self.HistCost[-1] - self.HistCost[-2]) < tol)

    def train(self, x_train, y_train, epochs, learning_rate, print_epochs=True):
        samples = len(x_train)
        self.HistCost = []

        for epoch in range(epochs):
            error = 0
            for sample in range(samples):
                input_signal = x_train[sample]
                self.forward_propagation(input_signal)
                error += self.loss(y_train[sample], self.y_pred)
                self.backpropagation(y_train[sample], learning_rate)
            error /= samples
            self.HistCost.append(error)
            if print_epochs == True:
                print('epoch %d/%d   error=%f' % (epoch + 1, epochs, error))

    def set_loss(self, loss="MSE"):
        if loss == "MSE":
            return lambda y_true, y_pred: np.mean(np.power(y_true - y_pred, 2))

    def set_loss_prime(self, loss="MSE"):
        if loss == "MSE":
            return lambda y_true, y_pred: 2 * (y_pred - y_true) / y_true.size

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