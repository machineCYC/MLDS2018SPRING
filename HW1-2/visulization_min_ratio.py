import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def main(args):
    loss = np.load(os.path.join(args.LOG_DIR_PATH, "Grad_zeros", "loss.npy"))
    min_ratio = np.load(os.path.join(args.LOG_DIR_PATH, "Grad_zeros", "min_ratio.npy"))

    plt.figure()
    plt.xlabel("Min_ratio")
    plt.ylabel("Loss")
    plt.scatter(loss, min_ratio, c="red", alpha=0.5)
    plt.savefig(os.path.join(args.SAVE_IMAGE_DIR_PATH, "min_ratio.png"))
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