import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def main(args):
    batch_size = np.load(os.path.join(args.LOG_DIR_PATH, "flatness_vs_gen_p2_sens", "batch_size.npy"))
    train_loss = np.load(os.path.join(args.LOG_DIR_PATH, "flatness_vs_gen_p2_sens", "train_loss.npy"))
    train_accuracy = np.load(os.path.join(args.LOG_DIR_PATH, "flatness_vs_gen_p2_sens", "train_acc.npy"))
    valid_loss = np.load(os.path.join(args.LOG_DIR_PATH, "flatness_vs_gen_p2_sens", "valid_loss.npy"))
    valid_accuracy = np.load(os.path.join(args.LOG_DIR_PATH, "flatness_vs_gen_p2_sens", "valid_acc.npy"))
    sensitivity = np.load(os.path.join(args.LOG_DIR_PATH, "flatness_vs_gen_p2_sens", "sensitivity.npy"))

    ax1 = plt.figure()
    # plt.yscale("log")
    plt.xlabel("batch_size")
    plt.ylabel("Cross_entropy", color="blue")
    plt.plot(batch_size, train_loss, c="blue", label="train", linestyle="-", alpha=0.5)
    plt.plot(batch_size, valid_loss, c="blue", label="valid", linestyle="--", alpha=0.5)
    plt.legend()

    ax2 = plt.gca().twinx()
    plt.xlabel("batch_size")
    plt.ylabel("Sensitivity", color="red")
    plt.plot(batch_size, sensitivity, c="red", label="sensitivity", linestyle="-", alpha=0.5)
    plt.legend()
    plt.savefig(os.path.join(args.SAVE_IMAGE_DIR_PATH, "Flatness_vs_gen_p2_sens_loss.png"))
    plt.clf()

    ax1 = plt.figure()
    plt.xlabel("batch_size")
    plt.ylabel("Accuracy", color="blue")
    plt.plot(batch_size, train_accuracy, c="blue", label="train", linestyle="-", alpha=0.5)
    plt.plot(batch_size, valid_accuracy, c="blue", label="valid", linestyle="--", alpha=0.5)
    plt.legend()

    ax2 = plt.gca().twinx()
    plt.xlabel("batch_size")
    plt.ylabel("Sensitivity", color="red")
    plt.plot(batch_size, sensitivity, c="red", label="sensitivity", linestyle="-", alpha=0.5)
    plt.legend()
    plt.savefig(os.path.join(args.SAVE_IMAGE_DIR_PATH, "Flatness_vs_gen_p2_sens_acc.png"))


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
