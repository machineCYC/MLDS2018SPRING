import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def main(args):
    train_loss = np.load(os.path.join(args.LOG_DIR_PATH, "Random_label", "train_loss.npy"))
    train_accuracy = np.load(os.path.join(args.LOG_DIR_PATH, "Random_label", "train_accuracy.npy"))
    valid_loss = np.load(os.path.join(args.LOG_DIR_PATH, "Random_label", "valid_loss.npy"))
    valid_accuracy = np.load(os.path.join(args.LOG_DIR_PATH, "Random_label", "valid_accuracy.npy"))

    plt.figure()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(np.arange(len(train_loss)), train_loss, c="red", label="train", alpha=0.5)
    plt.plot(np.arange(len(valid_loss)), valid_loss, c="blue", label="valid", alpha=0.5)
    plt.legend()
    plt.savefig(os.path.join(args.SAVE_IMAGE_DIR_PATH, "Random_label_loss.png"))
    plt.clf()

    plt.figure()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.plot(np.arange(len(train_accuracy)), train_accuracy, c="red", label="train", alpha=0.5)
    plt.plot(np.arange(len(valid_accuracy)), valid_accuracy, c="blue", label="valid", alpha=0.5)
    plt.legend()
    plt.savefig(os.path.join(args.SAVE_IMAGE_DIR_PATH, "Random_label_accuracy.png"))
    plt.clf()

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

