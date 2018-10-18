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
    epochs = 10
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
    with tf.name_scope("Inputs"):
        X_placeholder = tf.placeholder(tf.float32, [None, 1])
        Y_placeholder = tf.placeholder(tf.float32, [None, 1])

    if args.MODEL_TYPES == "shallow":
        predict = shallow_model(X_placeholder)
    elif args.MODEL_TYPES == "medium":
        predict = medium_model(X_placeholder)
    elif args.MODEL_TYPES == "deep":
        predict = deep_model(X_placeholder)
    else:
        print("MODEL_TYPES is wrong!!!")

    with tf.name_scope("mean_square_error"):
        mse_loss = tf.reduce_mean(tf.square(predict - Y_placeholder))
        tf.summary.scalar("Loss", mse_loss) # In tensorboard event

    with tf.name_scope("train"):
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(mse_loss)

    # train
    with tf.Session() as sess:
        # Merge all the summaries and write them out to ./logs/Tensorboard/train/
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(args.LOG_DIR_PATH, args.MODEL_TYPES, "Tensorboard/train/"), sess.graph)

        sess.run(tf.global_variables_initializer())
        count_model_parameters()
        nbrof_batch = int(len(X_train)/batch_size)

        mes_loss_list = []
        for e in range(epochs):
            for i in range(nbrof_batch):
                batch_x, batch_y = X_train[i*batch_size: (i+1)*batch_size], y_train[i*batch_size: (i+1)*batch_size]
                _, summary, mse_loss_ = sess.run([train_step, merged, mse_loss], feed_dict={X_placeholder:batch_x, Y_placeholder:batch_y})
                train_writer.add_summary(summary, e)

                if i %50 == 0:
                    print("epochs:{}, steps:{}, loss={}".format(e, (e*nbrof_batch) + i, mse_loss_))
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
        print("Save model to path: {}".format(os.path.dirname(save_model_dir_path)))

if __name__ == "__main__":
    PROJECT_DIR_PATH = os.path.dirname(__file__)
    LOG_DIR_PATH = os.path.join(PROJECT_DIR_PATH, "logs")
    SAVE_MODLE_DIR_PATH = os.path.join(PROJECT_DIR_PATH, "save_models")
    MODEL_TYPES = "deep"

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
