import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as K
from keras.datasets import mnist

from model.basic_model import simple_model
from src.utils import np_save_list


def main(args):
    # parameters
    epochs = 500
    train_size_data = 55000

    # load data
    (X_train, y_train), (X_valid, y_valid) = mnist.load_data() # (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)

    if K.image_data_format() == "channels_first":
        X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
        X_valid = X_valid.reshape(X_valid.shape[0], 1, 28, 28)
    else:
        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
        X_valid = X_valid.reshape(X_valid.shape[0], 28, 28, 1)

    X_train = X_train.astype("float32") / 255.
    X_valid = X_valid.astype("float32") / 255.
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_valid = tf.keras.utils.to_categorical(y_valid, 10)

    # random data order
    random_order = np.arange(len(X_train))
    np.random.shuffle(random_order)
    X_train = X_train[random_order][0:train_size_data]
    y_train = y_train[random_order][0:train_size_data]

    print("x_train shape:", X_train.shape)
    print(X_train.shape[0], "train samples")
    print("X_valid shape:", X_valid.shape)
    print(X_valid.shape[0], "valid samples")
    
    batch_size_list = [1024, 64]
    learning_rate_list = [1e-2, 1e-3]
    for batch_size, learning_rate in zip(batch_size_list, learning_rate_list):
        
        # define graph
        tf.reset_default_graph()
        graph = tf.Graph()
        with graph.as_default():
            with tf.name_scope("Inputs"):
                X_placeholder = tf.placeholder(tf.float32, [None, 28, 28, 1], name="x_placeholder")
                Y_placeholder = tf.placeholder(tf.float32, [None, 10], name="y_placeholder")
            
            with tf.name_scope("Architecture"):
                logits = simple_model(X_placeholder)

            with tf.name_scope("Cross_entropy_loss"):
                cross_entropy = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits_v2(
                        logits=logits, 
                        labels=Y_placeholder
                    ),
                    name="cross_entropy_loss"
                )
                tf.summary.scalar("cross_entropy", cross_entropy) # In tensorboard event

            with tf.name_scope("Train"):
                train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy, name="train_step")

            with tf.name_scope("Accuracy"):
                accuracy = tf.reduce_mean(
                    tf.cast(
                        tf.equal(
                            tf.argmax(Y_placeholder, axis=1),
                            tf.argmax(logits, axis=1)
                        ),
                        tf.float32
                    ),
                    name="accuracy"
                )

        # # train
        # with tf.Session(graph=graph) as sess:
        #     # Merge all the summaries and write them out to ./logs/
        #     merged = tf.summary.merge_all()
        #     train_writer = tf.summary.FileWriter(os.path.join(args.LOG_DIR_PATH, "flatness_vs_gen_p1_{}".format(batch_size), "Tensorboard/train/"), sess.graph)
        #     valid_writer = tf.summary.FileWriter(os.path.join(args.LOG_DIR_PATH, "flatness_vs_gen_p1_{}".format(batch_size), "Tensorboard/vaild/"))

        #     sess.run(tf.global_variables_initializer())

        #     nbrof_batch = int(len(X_train) / batch_size)

        #     for e in range(epochs):
        #         for i in range(nbrof_batch):
        #             batch_x, batch_y = X_train[i * batch_size: (i+1) * batch_size], y_train[i * batch_size: (i+1) * batch_size]
        #             feed_dict = {X_placeholder: batch_x, Y_placeholder: batch_y}

        #             _, train_summary, train_cross_entropy_ = sess.run([train_step, merged, cross_entropy], feed_dict=feed_dict)
                    
        #             train_accuracy_ = sess.run(accuracy, feed_dict=feed_dict)
        #             train_writer.add_summary(train_summary, e)

        #             if i %10 == 0:
        #                 print("epochs:{}, steps:{}, loss={}, accuracy={}".format(e, (e*nbrof_batch) + i, train_cross_entropy_, train_accuracy_))

        #         # valid model
        #         random_choice = np.random.randint(0, len(X_valid), size=batch_size)
        #         batch_x, batch_y = X_valid[random_choice], y_valid[random_choice]
        #         feed_dict = {X_placeholder: batch_x, Y_placeholder: batch_y}

        #         valid_accuracy_, valid_cross_entropy_, valid_summary = sess.run([accuracy, cross_entropy, merged], feed_dict=feed_dict)
        #         valid_writer.add_summary(valid_summary, e)

        #         print("epochs:{}, steps:{}, valid loss={}, valid accuracy={}".format(e, (e*nbrof_batch) + i, valid_cross_entropy_, valid_accuracy_))

        #     weights_list = []
        #     for v in tf.trainable_variables():

        #         weights_list.append(sess.run(v))
            
        #     # save model weights
        #     weights_save_path = os.path.join(args.LOG_DIR_PATH, "flatness_vs_gen_p1_{}".format(batch_size), "weights.npy")
        #     np_save_list(weights_save_path, weights_list)


    wweights_64 = np.load(os.path.join(args.LOG_DIR_PATH, "flatness_vs_gen_p1_64", "weights.npy"))
    wweights_1024 = np.load(os.path.join(args.LOG_DIR_PATH, "flatness_vs_gen_p1_1024", "weights.npy"))

    alpha_list = []
    train_accuracy_list = []
    valid_accuracy_list = []
    train_cross_entropy_list = []
    valid_cross_entropy_list = []
    for alpha in range(-10, 22, 2):
        alpha = alpha * 1e-1
        alpha_list.append(alpha)

        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())

            for idx, v in enumerate(tf.trainable_variables()):
                inter_weights = alpha * wweights_64[idx] + (1-alpha) * wweights_1024[idx]
                assign = tf.assign(v, inter_weights)
                sess.run(assign)
            
            X_placeholder = graph.get_tensor_by_name("Inputs/x_placeholder:0")
            Y_placeholder = graph.get_tensor_by_name("Inputs/y_placeholder:0")
            train_step = graph.get_operation_by_name("Train/train_step")
            cross_entropy = graph.get_tensor_by_name("Cross_entropy_loss/cross_entropy_loss:0")
            accuracy = graph.get_tensor_by_name("Accuracy/accuracy:0")

            feed_dict = {X_placeholder: X_train[0:1000], Y_placeholder: y_train[0:1000]}
            train_accuracy_, train_cross_entropy_ = sess.run([accuracy, cross_entropy], feed_dict=feed_dict)
            train_accuracy_list.append(train_accuracy_)
            train_cross_entropy_list.append(train_cross_entropy_)
        
            feed_dict = {X_placeholder:X_valid[0:1000], Y_placeholder: y_valid[0:1000]}
            test_accuracy_, test_cross_entropy_ = sess.run([accuracy, cross_entropy], feed_dict=feed_dict)
            valid_accuracy_list.append(test_accuracy_)
            valid_cross_entropy_list.append(test_cross_entropy_)
    
    # save alpha
    alpha_save_path = os.path.join(args.LOG_DIR_PATH, "flatness_vs_gen_p1_inter", "alpha.npy")
    np_save_list(alpha_save_path, alpha_list)
    # save train accuracy
    train_accuracy_save_path = os.path.join(args.LOG_DIR_PATH, "flatness_vs_gen_p1_inter", "train_acc.npy")
    np_save_list(train_accuracy_save_path, train_accuracy_list)
    # save valid accuracy
    valid_accuracy_save_path = os.path.join(args.LOG_DIR_PATH, "flatness_vs_gen_p1_inter", "valid_acc.npy")
    np_save_list(valid_accuracy_save_path, valid_accuracy_list)
    # save train cross entropy loss
    train_cross_entropy_save_path = os.path.join(args.LOG_DIR_PATH, "flatness_vs_gen_p1_inter", "train_loss.npy")
    np_save_list(train_cross_entropy_save_path, train_cross_entropy_list)
    # save valid cross entropy loss
    valid_cross_entropy_save_path = os.path.join(args.LOG_DIR_PATH, "flatness_vs_gen_p1_inter", "valid_loss.npy")
    np_save_list(valid_cross_entropy_save_path, valid_cross_entropy_list)

if __name__ == "__main__":
    PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))
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