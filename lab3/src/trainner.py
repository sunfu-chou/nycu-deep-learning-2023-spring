import torch
import torch.nn as nn
import torch.optim as optim

import logger as logger
import bcolors


class Trainer:
    def __init__(self, model, optimizer, device, log, weight_saver):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.model.to(self.device)
        self.log = log
        self.weight_saver = weight_saver

    def train(self, loader_train, loader_test, epochs=300, hyper_params=None):
        criterion = nn.CrossEntropyLoss()
        max_testing_accuracy = 0.0
        self.hyper_params = hyper_params

        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            self.model.train()

            for i, (data, label) in enumerate(loader_train):
                data = data.to(self.device, dtype=torch.float)
                label = label.to(self.device, dtype=torch.long)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, label)
                loss.backward()
                self.optimizer.step()

                correct += torch.sum(torch.argmax(output, dim=1) == label).item()
                total_loss += loss.item()

            loss = total_loss / len(loader_train)
            training_accuracy = correct / len(loader_train.dataset)
            testing_accuracy = self.test(loader_test)
            self.log.add(epoch + 1, loss, training_accuracy, testing_accuracy)

            if testing_accuracy > max_testing_accuracy:
                max_testing_accuracy = testing_accuracy
                if testing_accuracy > 0.8:
                    print(
                        f"{bcolors.BCOLORS.OKBLUE}Epoch {epoch+1:>3d}, loss: {loss: .5f}, training accuracy: {training_accuracy*100: .5f} %, testing accuracy: {testing_accuracy*100: .5f} %{bcolors.BCOLORS.ENDC}"
                    )
                    self.weight_saver.save(
                        self.model, self.hyper_params, testing_accuracy
                    )

            if (epoch + 1) % 100 == 0:
                print(
                    f"Epoch {epoch+1:>3d}, loss: {loss: .5f}, training accuracy: {training_accuracy*100: .5f} %, testing accuracy: {testing_accuracy*100: .5f} %"
                )

    def test(self, loader_test):
        self.model.eval()
        correct = 0
        for i, (data, label) in enumerate(loader_test):
            data = data.to(self.device, dtype=torch.float)
            label = label.to(self.device, dtype=torch.long)

            output = self.model(data)

            correct += torch.sum(torch.argmax(output, dim=1) == label).item()

        return correct / len(loader_test.dataset)
