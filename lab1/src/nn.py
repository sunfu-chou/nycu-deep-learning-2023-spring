import actfcn
import optimizer
import utils
import numpy as np


class Layer:
    def __init__(
        self,
        input_dim=2,
        output_dim=2,
        have_bias=True,
        act_fcn=actfcn.ActFcn,
        optimizer=optimizer.Optimizer,
        optimizer_parameter=optimizer.Optimizer
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.have_bias = have_bias
        self.act_fcn = act_fcn()
        self.optimizer = optimizer(*optimizer_parameter.get_param)

        self.w = np.random.randn(self.input_dim, self.output_dim)
        self.b = np.random.randn(1, self.output_dim)

        # self.w *= 0.1
        # self.b *= 0.1

    def forward(self, input):
        self.intputs = input
        if self.have_bias:
            self.outputs = self.act_fcn.forward(input.dot(self.w) + self.b)
        else:
            self.outputs = self.act_fcn.forward(input.dot(self.w))
        return self.outputs

    def backward(self, dy):
        dy_new = dy * self.act_fcn.backward(self.outputs)
        dw = self.intputs.T.dot(dy_new)
        db = np.sum(dy_new, axis=0)
        self.w, self.b = self.optimizer.optimize(self.w, dw, self.b, db)
        return dy_new.dot(self.w.T)


class NN:
    def __init__(
        self,
        dims,
        have_bias=True,
        act_fcn=actfcn.ActFcn,
        optimizer=optimizer.Optimizer,
        optimizer_parameter=optimizer.Optimizer
    ):
        self.layers = (
            Layer(2, dims[0], have_bias, act_fcn, optimizer, optimizer_parameter),
            Layer(dims[0], dims[1], have_bias, act_fcn, optimizer, optimizer_parameter),
            Layer(dims[1], 1, have_bias, act_fcn, optimizer, optimizer_parameter),
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dy):
        for layer in reversed(self.layers):
            dy = layer.backward(dy)
