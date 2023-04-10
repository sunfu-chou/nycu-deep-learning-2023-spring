# read csv and plot
import pandas as pd
import matplotlib.pyplot as plt
import glob
import yaml

import logger

activation_order = {"relu": 0, "leaky_relu": 1, "elu": 2}

model = "EEGNet"
model = "DeepConv"

window = 50

figure_name = f"loss_comparasion_{model.lower()}.png"


def read_yaml(file) -> dict:
    print(file)
    with open(file) as f:
        try:
            return yaml.load(f, Loader=yaml.Loader)
        except yaml.YAMLError as exc:
            print(exc)
            return {}


def main():
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

    meta_files = glob.glob(str(log_dir_path / "*.yaml"))
    meta_list = [read_yaml(file) for file in meta_files]
    meta_list = sorted(meta_list, key=lambda x: activation_order[x["activation"]])
    meta_list = sorted(meta_list, key=lambda x: x["weight_decay"])
    csv_files = [
        meta_list[i]["filename"].with_suffix(".csv") for i in range(len(meta_list))
    ]
    df_list = [pd.read_csv(file) for file in csv_files]

    for i, df in enumerate(df_list):
        if meta_list[i]["model"].lower() == model.lower():
            df.columns = df.columns.str.strip()
            df["loss"] = df["loss"].rolling(window=window).mean()
            df["epoch"] = df["epoch"].rolling(window=window).mean()

            plt.rcParams["figure.figsize"] = (10, 10)

            plt.plot(
                df["epoch"],
                df["loss"],
                label=meta_list[i]["activation"] + f" ({meta_list[i]['weight_decay']})",
            )

            plt.title(f"Loss comparasion ({model})", fontsize=28)
            plt.figtext(
                0.5,
                0.015,
                "with rolling average windows = " + str(window),
                ha="center",
                va="center",
                fontsize=12,
            )

            plt.xlabel("epoch", fontsize=18)
            # plt.ylim(0, 1)
            plt.legend(loc="upper right")
    # plt.show()
    plt.savefig(lab_dir_path / "figures" / figure_name)
    print("figure saved to", lab_dir_path / "figures" / figure_name)


if __name__ == "__main__":
    main()
