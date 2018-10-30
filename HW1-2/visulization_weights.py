import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def main(args):
    ### compress weights
    accuracy_list = [] # 8, epochs/3
    points_list = [] # 8, epochs/3, 2
    
    for i in range(8):
        weight = np.load(os.path.join(args.LOG_DIR_PATH, "Train_times_" + str(i+1), "weights.npy"))

        weights = np.asarray(weight) # epochs/3, parameter length
        S = (weights - np.mean(weights, axis=0))
        U = np.dot(S, S.transpose()) # epochs/3, epochs/3
        eig_val, eig_vec = np.linalg.eigh(U)
        eig_vec = np.dot(S.T, eig_vec)

        sort_order = np.argsort(eig_val)[::-1]
        W = eig_vec.T[sort_order].T[:, 0:2]
        
        points = np.dot(weights, W) # epochs/3, 2
        points_list.append(points)
        
        accuracy = np.load(os.path.join(args.LOG_DIR_PATH, "Train_times_" + str(i+1), "accuracy.npy"))
        accuracy_list.append(accuracy)
    
    ### plot weights
    listColor = ["#ff0000", "#ffff00", "#00ff00", "#00ffff", "#0000ff",
                 "#ff00ff", "#990000", "#999900"]
    plt.figure()
    plt.title("Visualize weights")
    plt.xlabel("PCA_1")
    plt.ylabel("PCA_2")
    for i in range(8):
        pca_x, pca_y = points_list[i].T
        plt.scatter(pca_x, pca_y, c=listColor[i], alpha=0.5, label="train_time" + str(i+1))
        for j in range(len(pca_x)):
            plt.annotate(str(accuracy_list[i][j]), (pca_x[j]*1.05, pca_y[j]*1.05))
    plt.legend()
    plt.savefig(os.path.join(args.SAVE_IMAGE_DIR_PATH, "Visualize_weights.png"))
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