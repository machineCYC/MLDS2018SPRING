import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from model.basic_model import shallow_model
from model.basic_model import medium_model
from model.basic_model import deep_model
from src.utils import count_model_parameters


def main(args):
    # parameters
    batch_size = 128
    epochs = 20000
    learning_rate = 0.01

    # Fix random seed for reproducibility.
    seed = 416
    np.random.seed(seed)
    print("Fixed random seed for reproducibility.")

    # target function sin(5x*pi) / 5x*pi
    X_train = np.linspace(0.0001, 1.0, num=10000)[:, np.newaxis]
    np.random.shuffle(X_train)
    y_train = np.sinc(5 * X_train)

    # visulization target function
    # plt.scatter(x=X_train, y=y_train, c="r", marker="o")
    # plt.show()

    X_placeholder = tf.placeholder(tf.float32, [None, 1])
    Y_placeholder = tf.placeholder(tf.float32, [None, 1])

    # Weights1 = tf.Variable(tf.truncated_normal([1, 200], mean=0.0, stddev=0.01))
    # biases1 = tf.Variable(tf.zeros([1, 200]) + 0.1)

    # Weights2 = tf.Variable(tf.truncated_normal([200, 1], mean=0.0, stddev=0.01))
    # biases2 = tf.Variable(tf.zeros([1, 1]) + 0.1)

    # dense = tf.matmul(X_placeholder, Weights1) + biases1
    # dense = tf.nn.tanh(dense)
    # predict = tf.matmul(dense, Weights2) + biases2
    if args.MODEL_TYPES == "shallow":
        predict = shallow_model(X_placeholder)
    elif args.MODEL_TYPES == "medium":
        predict = medium_model(X_placeholder)
    elif args.MODEL_TYPES == "deep":
        predict = deep_model(X_placeholder)
    else:
        print("MODEL_TYPES is wrong!!!")

    # mes loss
    mse_loss = tf.reduce_mean(tf.square(predict - Y_placeholder))

    # train step
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(mse_loss)

    # train
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        count_model_parameters()
        nbrof_batch = int(len(X_train)/batch_size)

        mes_loss_list = []
        for e in range(epochs):
            for i in range(nbrof_batch):
                batch_x, batch_y = X_train[i*batch_size: (i+1)*batch_size], y_train[i*batch_size: (i+1)*batch_size]
                _, mse_loss_, predict_ = sess.run([train_step, mse_loss, predict], feed_dict={X_placeholder:batch_x, Y_placeholder:batch_y})
                # W1_, b1_ = sess.run([Weights1, biases1])
                if i %50 == 0:
                    print("epochs:{}, steps:{}, loss={}".format(e, (e*nbrof_batch) + i, mse_loss_))
                    # print(np.mean(W1_))
                    # print(np.mean(b1_))
            mes_loss_list.append(mse_loss_)
        
        # save loss process
        mes_loss_list = np.asarray(mes_loss_list)
        loss_save_path = os.path.join(args.LOG_DIR_PATH, args.MODEL_TYPES, args.MODEL_TYPES + ".npy")
        if not os.path.exists(os.path.dirname(loss_save_path)):
            os.makedirs(os.path.dirname(loss_save_path))
        np.save(loss_save_path, mes_loss_list)
        print("Loss process to path: ", loss_save_path)

        # save model
        saver = tf.train.Saver()
        save_model_dir_path = os.path.join(args.SAVE_MODLE_DIR_PATH, args.MODEL_TYPES, args.MODEL_TYPES)
        if not os.path.exists(os.path.dirname(save_model_dir_path)):
            os.makedirs(os.path.dirname(save_model_dir_path))
        save_path = saver.save(sess, save_model_dir_path)
        print("Save model to path: ", save_model_dir_path)

if __name__ == "__main__":
    PROJECT_DIR_PATH = os.path.dirname(__file__)
    LOG_DIR_PATH = os.path.join(PROJECT_DIR_PATH, "logs")
    SAVE_MODLE_DIR_PATH = os.path.join(PROJECT_DIR_PATH, "save_models")
    MODEL_TYPES = "medium"

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
        "--MODEL_TYPES",
        type=str,
        default=MODEL_TYPES,
        help=""
    )
    main(parser.parse_args())
