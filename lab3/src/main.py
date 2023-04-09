import argparse


import torch
from torch.utils.data import DataLoader, TensorDataset


import dataloader
import eegnet
import deepconv
import trainner
import logger
import bcolors


# /home/user/nycu-deep-learning-2023-spring/lab3/weights/2023-04-09-23-45-23.pt
# parse arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="train")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=1080)
    parser.add_argument("--model", type=str, default="EEGNet")
    parser.add_argument("--activation", type=str, default="leaky_relu")
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-1)
    parser.add_argument("--weight", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    print(args.weight)
    file_handler = logger.FileHandeler()
    (
        lab_dir_path,
        data_dir_path,
        log_dir_path,
        log_path,
        meta_path,
        weights_dir_path,
        weights_path,
    ) = file_handler.get()

    # Torch Check GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"{bcolors.BCOLORS.OKGREEN}Using GPU{bcolors.BCOLORS.ENDC}")
    else:
        print(f"{bcolors.BCOLORS.FAIL}Using CPU{bcolors.BCOLORS.ENDC}")
        raise Exception("GPU not available")

    # Load data
    train_data, train_label, test_data, test_label = dataloader.read_bci_data()
    loader_train = dataloader.to_tensor_loader(
        train_data, train_label, batch_size=args.batch_size, shuffle=True
    )
    loader_test = dataloader.to_tensor_loader(
        test_data, test_label, batch_size=args.batch_size, shuffle=False
    )

    # Create logger
    log = logger.Log(log_path)
    weight_saver = logger.WeightSaver(weights_path)

    # Hyper parameters
    hyper_params = logger.Meta(
        meta_path,
        model=args.model,
        activation=args.activation,
        optimizer=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Create model
    if args.model.lower() == "eegnet":
        model = eegnet.EEGNet("leaky_relu")
    elif args.model.lower() == "deepconv":
        model = deepconv.DeepConvNet()
    else:
        raise Exception("Invalid model name")

    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), hyper_params.lr, weight_decay=hyper_params.weight_decay
    )
    print(f"{bcolors.BCOLORS.OKCYAN}")
    print(f"Type: {args.type}")
    print(
        f"\n\n{bcolors.BCOLORS.OKCYAN}{hyper_params.model}: Using {hyper_params.activation} as activation function"
    )
    print(
        f"{hyper_params.model}: Using {hyper_params.optimizer} as optimizer with learning rate {hyper_params.lr} and weight decay {hyper_params.weight_decay}"
    )
    print(f"{hyper_params.model}: Training {args.epochs} epochs, data loaded in {args.batch_size}")

    print(f"\n{bcolors.BCOLORS.ENDC}")

    # Create trainer
    trainer = trainner.Trainer(model, optimizer, device, log, weight_saver)
    if args.type == "train":
        trainer.train(loader_train, loader_test, epochs=args.epochs, hyper_params=hyper_params)
    elif args.type == "test":
        model.load_state_dict(torch.load(args.weight))
        model.load_state_dict(torch.load(args.weight))
        testing_accuracy = trainer.test(loader_test)
        print(
            f"{bcolors.BCOLORS.WARNING}testing accuracy: {testing_accuracy*100: .5f} %{bcolors.BCOLORS.ENDC}\n"
        )

    hyper_params.save()
    log.write()


if __name__ == "__main__":
    main()
