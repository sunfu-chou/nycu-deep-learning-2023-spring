import numpy as np


class Optimizer:
    def __init__(self, lr):
        self.lr = lr

    def optimize(self, w, dw, b, db):
        pass

    @classmethod
    def param(cls, lr):
        return cls(lr)

    @property
    def get_param(self):
        return (self.lr,)


class SGD(Optimizer):
    def __init__(self, lr):
        super().__init__(lr)

    def optimize(self, w, dw, b, db):
        w = w - self.lr * dw
        b = b - self.lr * db
        return w, b

    @classmethod
    def param(cls, lr):
        return cls(lr)

    @property
    def get_param(self):
        return (self.lr,)


class Momentum(Optimizer):
    def __init__(self, lr, beta):
        super().__init__(lr)
        self.beta = beta
        self.vw = None
        self.vb = None

    def optimize(self, w, dw, b, db):
        if self.vw is None:
            self.vw = np.zeros(w.shape)
        self.vw = self.beta * self.vw - self.lr * dw
        w = w + self.vw

        if self.vb is None:
            self.vb = np.zeros(b.shape)
        self.vb = self.beta * self.vb - self.lr * db
        b = b + self.vb

        return w, b

    @classmethod
    def param(cls, lr, beta):
        return cls(lr, beta)

    @property
    def get_param(self):
        return (self.lr, self.beta)


class AdaGrad(Optimizer):
    def __init__(self, lr, epsilon):
        super().__init__(lr)
        self.epsilon = epsilon
        self.n = 0.0

    def optimize(self, w, dw, b, db):
        self.n = self.n + dw**2
        w = w - self.lr * dw / np.sqrt(self.n + self.epsilon)
        return w, b

    @classmethod
    def param(cls, lr, epsilon):
        return cls(lr, epsilon)

    @property
    def get_param(self):
        return (self.lr, self.epsilon)


class Adam(Optimizer):
    def __init__(self, lr, b1, b2, epsilon):
        super().__init__(lr)
        self.b1 = b1
        self.b2 = b2
        self.mw = None
        self.vw = None
        
        self.mb = None
        self.vb = None
        self.epsilon = epsilon

    def optimize(self, w, dw, b, db):
        if self.mw is None:
            self.mw = np.zeros(w.shape)
        if self.vw is None:
            self.vw = np.zeros(w.shape)

        self.mw = self.b1 * self.mw + (1 - self.b1) * dw
        self.vw = self.b2 * self.vw + (1 - self.b2) * (dw**2)
        m_hat = self.mw / (1 - self.b1)
        v_hat = self.vw / (1 - self.b2)

        w = w - self.lr * m_hat / (np.sqrt(v_hat + self.epsilon))
        
        if self.mb is None:
            self.mb = np.zeros(b.shape)
        if self.vb is None:
            self.vb = np.zeros(b.shape)

        self.mb = self.b1 * self.mb + (1 - self.b1) * db
        self.vb = self.b2 * self.vb + (1 - self.b2) * (db**2)
        m_hat = self.mb / (1 - self.b1)
        v_hat = self.vb / (1 - self.b2)

        b = b - self.lr * m_hat / (np.sqrt(v_hat + self.epsilon))

        return w, b

    @classmethod
    def param(cls, lr, b1, b2, epsilon):
        return cls(lr, b1, b2, epsilon)

    @property
    def get_param(self):
        return (self.lr, self.b1, self.b2, self.epsilon)
