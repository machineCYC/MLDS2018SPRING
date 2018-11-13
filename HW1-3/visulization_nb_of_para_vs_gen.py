import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def main(args):
    total_para = np.load(os.path.join(args.LOG_DIR_PATH, "nbr_para_gen", "total_para.npy"))
    train_loss = np.load(os.path.join(args.LOG_DIR_PATH, "nbr_para_gen", "train_loss.npy"))
    train_accuracy = np.load(os.path.join(args.LOG_DIR_PATH, "nbr_para_gen", "train_accuracy.npy"))
    valid_loss = np.load(os.path.join(args.LOG_DIR_PATH, "nbr_para_gen", "valid_loss.npy"))
    valid_accuracy = np.load(os.path.join(args.LOG_DIR_PATH, "nbr_para_gen", "valid_accuracy.npy"))

    plt.figure()
    plt.xlabel("# parameters")
    plt.ylabel("Loss")
    plt.scatter(total_para, train_loss, c="red", label="train", alpha=0.5)
    plt.scatter(total_para, valid_loss, c="blue", label="valid", alpha=0.5)
    plt.legend()
    plt.savefig(os.path.join(args.SAVE_IMAGE_DIR_PATH, "Nbr_para_gen_loss.png"))
    plt.clf()

    plt.figure()
    plt.xlabel("# parameters")
    plt.ylabel("Accuracy")
    plt.scatter(total_para, train_accuracy, c="red", label="train", alpha=0.5)
    plt.scatter(total_para, valid_accuracy, c="blue", label="valid", alpha=0.5)
    plt.legend()
    plt.savefig(os.path.join(args.SAVE_IMAGE_DIR_PATH, "Nbr_para_gen_accuracy.png"))
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
