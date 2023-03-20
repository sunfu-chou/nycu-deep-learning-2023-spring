import numpy as np


class ActFcn:
    def __init__(self):
        pass

    def forward(self, x):
        pass

    def backward(self, x):
        pass


class Sigmoid(ActFcn):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def backward(self, x):
        return np.multiply(x, 1.0 - x)


class ReLU(ActFcn):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return np.maximum(x,0)

    def backward(self, x):
        z = np.copy(x)
        z[z > 0] = 1.0
        z[z <= 0] = 0.0
        return z
    
class LeakyReLU(ActFcn):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return np.maximum(x,0) + np.minimum(0.01 * x, 0)

    def backward(self, x):
        z = np.copy(x)
        z[z > 0] = 1.0
        z[z <= 0] = 0.01
        return z

class Tanh(ActFcn):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return np.tanh(x)

    def backward(self, x):
        return 1.0 - np.tanh(x) ** 2

class NoneAct(ActFcn):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

    def backward(self, x):
        return 1
