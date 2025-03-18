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
        self.bias -= learning_rate * np.sum(output_der, axis=1, keepdims=True)
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
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_der(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_der)

class ReLU(Activation):#leads to lots of NaNs(avoid)
    def __init__(self):
        super().__init__(
            lambda x: np.maximum(0, x),
            lambda x: (x > 0).astype(float)
        )
        
class Softmax():
    def __init__(self,one_true=False):
        self.input = None
        self.output = None
        self.one_true=one_true
    def forward(self,input):
        exp=np.exp(input-np.max(input))
        self.output=exp/np.sum(exp)
        return self.output
    def backward(self,output_der,learning_rate):
        if( not self.one_true):
            n = np.size(self.output)
            return np.dot((np.identity(n) - self.output.T) * self.output, output_der)
        else:
            return output_der
    
def mse(y_exp, y_pred):
    return np.mean(np.power(y_exp - y_pred, 2))

def mse_der(y_exp, y_pred):
    return 2 * (y_pred - y_exp) / np.size(y_exp)

def cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-12, 1.0)  # make 0 int 1e-12 so no log 0
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[1]

def cross_entropy_der(y_true, y_pred):
    return y_pred - y_true

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
