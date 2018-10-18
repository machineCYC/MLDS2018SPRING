import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def main(args):
    ### plot predict
    # target function sin(5x*pi) / 5x*pi
    X = np.linspace(0.0001, 1.0, num=1000)[:, np.newaxis]
    Y_true = np.sinc(5 * X)
    Y_shallow = np.load(os.path.join(args.LOG_DIR_PATH, "shallow", "predict_shallow.npy"))
    Y_medium = np.load(os.path.join(args.LOG_DIR_PATH, "medium", "predict_medium.npy"))
    Y_deep = np.load(os.path.join(args.LOG_DIR_PATH, "deep", "predict_deep.npy"))

    plt.figure()
    plt.title("Predict")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot(X, Y_true, label=r"\mathrm{sin}(2 \omega t)$")
    plt.plot(X, Y_shallow, label="shallow")
    plt.plot(X, Y_medium, label="medium")
    plt.plot(X, Y_deep, label="deep")
    plt.legend()
    plt.show()
    plt.clf()

    ### plot loss
    shallow_loss = np.load(os.path.join(args.LOG_DIR_PATH, "shallow", "shallow.npy"))
    medium_loss = np.load(os.path.join(args.LOG_DIR_PATH, "medium", "medium.npy"))
    deep_loss = np.load(os.path.join(args.LOG_DIR_PATH, "deep", "deep.npy"))

    plt.figure()
    plt.title("Loss v.s Epochs")
    plt.xlabel("# of epochs")
    plt.ylabel("Loss")
    plt.plot(np.arange(len(shallow_loss)), shallow_loss, label="shallow")
    plt.plot(np.arange(len(medium_loss)), medium_loss, label="medium")
    plt.plot(np.arange(len(deep_loss)), deep_loss, label="deep")
    plt.legend()
    plt.show()
    plt.clf()

if __name__ == "__main__":
    PROJECT_DIR_PATH = os.path.dirname(__file__)
    LOG_DIR_PATH = os.path.join(PROJECT_DIR_PATH, "logs")
    SAVE_MODLE_DIR_PATH = os.path.join(PROJECT_DIR_PATH, "save_models")

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
    main(parser.parse_args())