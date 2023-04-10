import numpy as np
import os
import torch
from torch.utils.data import DataLoader, TensorDataset


def read_bci_data():
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(absolute_path, "..", "data")
    S4b_train = np.load(os.path.join(data_dir, "S4b_train.npz"))
    X11b_train = np.load(os.path.join(data_dir, "X11b_train.npz"))
    S4b_test = np.load(os.path.join(data_dir, "S4b_test.npz"))
    X11b_test = np.load(os.path.join(data_dir, "X11b_test.npz"))

    train_data = np.concatenate((S4b_train["signal"], X11b_train["signal"]), axis=0)
    train_label = np.concatenate((S4b_train["label"], X11b_train["label"]), axis=0)
    test_data = np.concatenate((S4b_test["signal"], X11b_test["signal"]), axis=0)
    test_label = np.concatenate((S4b_test["label"], X11b_test["label"]), axis=0)

    train_label = train_label - 1
    test_label = test_label - 1
    train_data = np.transpose(np.expand_dims(train_data, axis=1), (0, 1, 3, 2))
    test_data = np.transpose(np.expand_dims(test_data, axis=1), (0, 1, 3, 2))

    mask = np.where(np.isnan(train_data))
    train_data[mask] = np.nanmean(train_data)

    mask = np.where(np.isnan(test_data))
    test_data[mask] = np.nanmean(test_data)

    return train_data, train_label, test_data, test_label


# convert to torch tensor data type
def to_tensor_loader(data, label, *args, **kwargs):
    dataset = TensorDataset(torch.from_numpy(data), torch.from_numpy(label))
    loader = DataLoader(dataset, *args, **kwargs)
    return loader
