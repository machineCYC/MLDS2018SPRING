import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def main(args):
    accuracy = np.load(os.path.join(args.LOG_DIR_PATH, "Visulize_grads", "accuracy.npy"))
    grads = np.load(os.path.join(args.LOG_DIR_PATH, "Visulize_grads", "grads.npy"))
    loss = np.load(os.path.join(args.LOG_DIR_PATH, "Visulize_grads", "loss.npy"))

    plt.figure(figsize=(8, 12))
    # plot loss
    plt.subplot(3, 1, 1)
    plt.title("Loss v.s Steps")
    plt.ylabel("Loss")
    plt.plot(np.arange(len(loss)), loss, c="blue")
    
    # plot grads
    plt.subplot(3, 1, 2)
    plt.title("Grads v.s Steps")
    plt.ylabel("Grads")
    plt.plot(np.arange(len(grads)), grads, c="green")

    # plot accuracy
    plt.subplot(3, 1, 3)
    plt.title("Accuracy v.s Steps")
    plt.xlabel("# of Steps")
    plt.ylabel("Accuracy")
    plt.plot(np.arange(len(accuracy)), accuracy, c="red")
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(args.SAVE_IMAGE_DIR_PATH, "visulization_grads.png"))
    plt.clf()


if __name__ == "__main__":
    PROJECT_DIR_PATH = os.path.dirname(__file__)
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