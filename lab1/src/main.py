#!/usr/bin/env python3
import actfcn
import optimizer
import nn
import utils
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    x, y = utils.generate_linear(100)
    
    act = actfcn.Sigmoid()
    opt = optimizer.SGD(0.01)
    
    nn = nn.NN((4, 4), False, act, opt)
    loss = utils.MSE()
    losses = []
    accuracies = []
    y_pred = []
    for epoch in range(5000):
        y_pred = nn.forward(x)
        loss_value = loss.forward(y, y_pred)
        losses.append(loss_value)
        accuracy_value = utils.accuracy(y, y_pred)
        accuracies.append(accuracy_value)

        dy = loss.backward(y, y_pred)
        nn.backward(dy)

        if epoch % 500 == 0:
            print(f"epoch: {epoch}, loss: {loss_value}, accuracy: {accuracy_value}")
    y_pred = np.around(y_pred)
    # utils.show_result(x, y, y_pred)
    # plt.xlabel('Epochs')
    # plt.ylabel('Losses')
    # plt.plot(losses)
    # plt.show()

