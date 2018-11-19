import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def main(args):
    alpha = np.load(os.path.join(args.LOG_DIR_PATH, "flatness_vs_gen_p1_inter", "alpha.npy"))
    train_loss = np.load(os.path.join(args.LOG_DIR_PATH, "flatness_vs_gen_p1_inter", "train_loss.npy"))
    train_accuracy = np.load(os.path.join(args.LOG_DIR_PATH, "flatness_vs_gen_p1_inter", "train_acc.npy"))
    valid_loss = np.load(os.path.join(args.LOG_DIR_PATH, "flatness_vs_gen_p1_inter", "valid_loss.npy"))
    valid_accuracy = np.load(os.path.join(args.LOG_DIR_PATH, "flatness_vs_gen_p1_inter", "valid_acc.npy"))

    ax1 = plt.figure()
    plt.yscale("log")
    plt.xlabel("alpha")
    plt.ylabel("Cross_entropy", color="blue")
    plt.plot(alpha, train_loss, c="blue", label="train", linestyle="-", alpha=0.5)
    plt.plot(alpha, valid_loss, c="blue", label="valid", linestyle="--", alpha=0.5)
    plt.legend()

    ax2 = plt.gca().twinx()
    plt.xlabel("alpha")
    plt.ylabel("Accuracy", color="red")
    plt.plot(alpha, train_accuracy, c="red", label="train", linestyle="-", alpha=0.5)
    plt.plot(alpha, valid_accuracy, c="red", label="valid", linestyle="--", alpha=0.5)
    plt.legend()
    plt.savefig(os.path.join(args.SAVE_IMAGE_DIR_PATH, "Flatness_vs_gen_p1_inter.png"))


if __name__ == "__main__":
    PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
    LOG_DIR_PATH = os.path.join(PROJECT_DIR_PATH, "logs")
    SAVE_IMAGE_DIR_PATH = os.path.join(PROJECT_DIR_PATH, "image")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--LOG_DIR_PATH",
        type=str,
        default=LOG_DIR_PATH,
        help=""
    )
    parser.add_argument(
        "--SAVE_IMAGE_DIR_PATH",
        type=str,
        default=SAVE_IMAGE_DIR_PATH,
        help=""
    )
    main(parser.parse_args())
