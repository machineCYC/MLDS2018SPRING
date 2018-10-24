import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def main(args):
    ### plot accuracy
    Y_shallow = np.load(os.path.join(args.LOG_DIR_PATH, "cnn_shallow", "cnn_shallow_accuracy.npy"))
    Y_medium = np.load(os.path.join(args.LOG_DIR_PATH, "cnn_medium", "cnn_medium_accuracy.npy"))
    Y_deep = np.load(os.path.join(args.LOG_DIR_PATH, "cnn_deep", "cnn_deep_accuracy.npy"))

    plt.figure()
    plt.title("Accuracy v.s Epochs")
    plt.xlabel("# of epochs")
    plt.ylabel("Accuracy")
    plt.plot(np.arange(len(Y_shallow)), Y_shallow, c="red", label="shallow")
    plt.plot(np.arange(len(Y_medium)), Y_medium, c="green", label="medium")
    plt.plot(np.arange(len(Y_deep)), Y_deep, c="blue", label="deep")
    plt.legend()
    plt.savefig(os.path.join(args.SAVE_IMAGE_DIR_PATH, "cnn_accuracy.png"))
    plt.clf()

    ### plot loss
    shallow_loss = np.load(os.path.join(args.LOG_DIR_PATH, "cnn_shallow", "cnn_shallow_loss.npy"))
    medium_loss = np.load(os.path.join(args.LOG_DIR_PATH, "cnn_medium", "cnn_medium_loss.npy"))
    deep_loss = np.load(os.path.join(args.LOG_DIR_PATH, "cnn_deep", "cnn_deep_loss.npy"))

    plt.figure()
    plt.title("Loss v.s Epochs")
    plt.xlabel("# of epochs")
    plt.ylabel("Loss")
    # plt.yscale("log")
    plt.plot(np.arange(len(shallow_loss)), shallow_loss, c="red", label="shallow")
    plt.plot(np.arange(len(medium_loss)), medium_loss, c="green", label="medium")
    plt.plot(np.arange(len(deep_loss)), deep_loss, c="blue", label="deep")
    plt.legend()
    plt.savefig(os.path.join(args.SAVE_IMAGE_DIR_PATH, "cnn_loss.png"))
    plt.clf()

if __name__ == "__main__":
    PROJECT_DIR_PATH = os.path.dirname(__file__)
    LOG_DIR_PATH = os.path.join(PROJECT_DIR_PATH, "logs")
    SAVE_MODLE_DIR_PATH = os.path.join(PROJECT_DIR_PATH, "save_models")
    SAVE_IMAGE_DIR_PATH = os.path.join(PROJECT_DIR_PATH, "image")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--LOG_DIR_PATH",
        type=str,
        default=LOG_DIR_PATH,
        help=""
    )
    parser.add_argument(
        "--SAVE_MODLE_DIR_PATH",
        type=str,
        default=SAVE_MODLE_DIR_PATH,
        help=""
    )
    parser.add_argument(
        "--SAVE_IMAGE_DIR_PATH",
        type=str,
        default=SAVE_IMAGE_DIR_PATH,
        help=""
    )
    main(parser.parse_args())