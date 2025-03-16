import numpy as np

class Layer:
    def __init__(self, input_size, output_size):
        self.input = None
        self.output = None
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_der, learning_rate):
        weight_der = np.dot(output_der, self.input.T)
        derE_x = np.dot(self.weights.T, output_der)
        self.weights -= learning_rate * weight_der
        self.bias -= learning_rate * np.sum(output_der, axis=1, keepdims=True)  # Fix bias update
        return derE_x

class Activation:
    def __init__(self, activation, act_der):
        self.input = None
        self.output = None
        self.activation = activation
        self.act_der = act_der

    def forward(self, input):
        self.input = input
        return self.activation(input)

    def backward(self, output_der, learning_rate):
        return np.multiply(output_der, self.act_der(self.input))

class tanh(Activation):
    def __init__(self):
        super().__init__(lambda x: np.tanh(x), lambda x: 1 - np.tanh(x) ** 2)
        
class Sigmoid(Activation):
    def __init__(self):
        super().__init__(
            lambda x: 1 / (1 + np.exp(-x)),  # Sigmoid function
            lambda x: (1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x))))  # Derivative
        )


def mse(y_exp, y_pred):
    return np.mean(np.power(y_exp - y_pred, 2))

def mse_der(y_exp, y_pred):
    return 2 * (y_pred - y_exp) / np.size(y_exp)

class NeuralNetwork:
    def __init__(self, *args):
        if len(args) < 2:
            raise ValueError("Error: Needs at least 2 layers")
        self.network = list(args)

    def predict(self, input):
        output = input
        for layer in self.network:
            output = layer.forward(output)
        return output

    def train(self, loss, loss_der, x_train, y_train, epochs=1000, learning_rate=0.01):
        for e in range(epochs):
            error = 0
            for x, y in zip(x_train, y_train):
                output = self.predict(x)
                error += loss(y, output)
                output_der = loss_der(y, output)
                
                for layer in reversed(self.network):
                    output_der = layer.backward(output_der, learning_rate)

            error /= len(x_train)
            if e % 1 == 0:
                print(f"Epoch {e}: Error {error}")
