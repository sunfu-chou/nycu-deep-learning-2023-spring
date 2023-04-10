import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    lab_dir_path = Path(__file__).resolve().parent.parent
    log_dir_path = lab_dir_path / "log"
    log_name = "2048-2023-04-05-01-40-03.log"
    log_name = "2048-2023-04-05-01-40-07.log"
    log_name = "2048-2023-04-05-01-45-04.log"
    # log_name = "2048-2023-04-05-01-45-04.log"
    figures_dir_path = lab_dir_path / "figures"
    figure_name = "2048-3.png"

    if not os.path.exists(figures_dir_path):
        os.makedirs(figures_dir_path)

    df = pd.read_csv(log_dir_path / log_name)
    df.columns = df.columns.str.strip()

    plt.rcParams["figure.figsize"] = (10, 10)
    plt.plot(df["episode"], df["mean"], label="mean")
    plt.plot(df["episode"], df["max"], label="max")
    plt.title("2048", fontsize=28)
    plt.xlabel("episode", fontsize=18)
    plt.ylabel("score", fontsize=18)
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig(figures_dir_path / figure_name)


if __name__ == "__main__":
    main()
