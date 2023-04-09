import os
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import torch
import yaml

import bcolors


class FileHandeler:
    def __init__(self):
        # lab project directory
        self.lab_dir_path = Path(__file__).resolve().parent.parent
        print(
            f"{bcolors.BCOLORS.OKBLUE}lab directory:\n    {self.lab_dir_path}{bcolors.BCOLORS.ENDC}"
        )

        # data directory
        self.data_dir_path = self.lab_dir_path / "data"
        if not os.path.exists(self.data_dir_path):
            raise Exception(
                f"{bcolors.BCOLORS.WARNING}data directory does not exist:\n    {self.data_dir_path}{bcolors.BCOLORS.ENDC}\n"
            )
        else:
            print(
                f"{bcolors.BCOLORS.OKBLUE}data directory:\n    {self.data_dir_path}{bcolors.BCOLORS.ENDC}"
            )
        # log directory
        self.log_dir_path = self.lab_dir_path / "log"
        self.log_path = self.log_dir_path / (Log.get_current_time() + ".csv")
        if not os.path.exists(self.log_dir_path):
            print(
                f"{bcolors.BCOLORS.WARNING}log directory does not exist, creating one...:\n{self.log_dir_path}{bcolors.BCOLORS.ENDC}"
            )
            os.makedirs(self.log_dir_path)
        else:
            print(
                f"{bcolors.BCOLORS.OKBLUE}log directory:\n    {self.log_dir_path}{bcolors.BCOLORS.ENDC}"
            )

        # meta directory
        self.meta_path = self.log_path.with_suffix(".yaml")

        # weights directory
        self.weights_dir_path = self.lab_dir_path / "weights"
        self.weights_path = self.weights_dir_path / (Log.get_current_time() + ".pt")
        if not os.path.exists(self.weights_dir_path):
            print(
                f"{bcolors.BCOLORS.OKBLUE}weights directory does not exist, creating one...:\n{self.weights_dir_path}{bcolors.BCOLORS.ENDC}"
            )
            os.makedirs(self.weights_dir_path)
        else:
            print(
                f"{bcolors.BCOLORS.OKBLUE}weights directory:\n    {self.weights_dir_path}{bcolors.BCOLORS.ENDC}"
            )

    def get(self):
        return (
            self.lab_dir_path,
            self.data_dir_path,
            self.log_dir_path,
            self.log_path,
            self.meta_path,
            self.weights_dir_path,
            self.weights_path,
        )


class Log:
    def __init__(self, filename):
        self.df = pd.DataFrame(
            columns=["epoch", "loss", "training_accuracy", "testing_accuracy"]
        )
        self.filename = filename

    def add(self, epoch, loss, training_accuracy, testing_accuracy):
        self.df.loc[len(self.df)] = [  # type: ignore
            epoch,
            loss,
            training_accuracy,
            testing_accuracy,
        ]

    def get_tail(self, n=5):
        return self.df.tail(n)

    def read(self, filename):
        self.df = pd.read_csv(filename)

    def write(self, filename=None):
        if filename is None:
            filename = self.filename
        self.df.to_csv(filename, index=False)
        print(f"{bcolors.BCOLORS.OKGREEN}Log saved to {filename}{bcolors.BCOLORS.ENDC}")

    @staticmethod
    def get_current_time():
        time_now = datetime.now()
        current_time = time_now.strftime("%Y-%m-%d-%H-%M-%S")
        return current_time


# class Meta: to save the model metadata
class Meta:
    def __init__(self, filename=None, *args, **kwargs):
        self.filename = filename
        self.__dict__.update(kwargs)

    def save(self):
        with open(self.filename, "w") as file:
            yaml.dump(self.__dict__, file)
            print(f"Model metadata saved to {self.filename}")


class WeightSaver:
    def __init__(self, filename=None):
        self.filename = filename

    def save(self, model, hyper_param, accuracy):
        print(
            f"{bcolors.BCOLORS.OKBLUE}Saving model weights to {self.filename}, with accuracy {accuracy}{bcolors.BCOLORS.ENDC}"
        )
        torch.save(model.state_dict(), self.filename)
        print(
            f"{bcolors.BCOLORS.OKBLUE}Model weights saved to {self.filename}{bcolors.BCOLORS.ENDC}"
        )
        print(
            f"{bcolors.BCOLORS.OKBLUE}Saving model metadata to {self.filename}{bcolors.BCOLORS.ENDC}"
        )
        meta = Meta(accuracy=accuracy)
        meta.__dict__.update(hyper_param.__dict__)
        meta.filename = self.filename.with_suffix(".yaml")
        meta.save()
        print(
            f"{bcolors.BCOLORS.OKBLUE}Model metadata saved to {self.filename}{bcolors.BCOLORS.ENDC}"
        )
