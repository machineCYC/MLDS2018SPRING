import os
import argparse
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.datasets import mnist

from model.basic_model import dnn_medium_model


def main(args):
    # parameters
    batch_size = 128
    epochs = 30
    learning_rate = 5e-4
    train_size_data = 2000

    # load data
    (X_train, y_train), (X_valid, y_valid) = mnist.load_data() # (60000, 28, 28) (10000,)

    if K.image_data_format() == "channels_first":
        X_train = X_train.reshape(X_train.shape[0], 1*28*28)
    else:
        X_train = X_train.reshape(X_train.shape[0], 1*28*28)

    X_train = X_train.astype("float32") / 255.
    y_train = tf.keras.utils.to_categorical(y_train, 10)

    # random data order
    random_order = np.arange(len(X_train))
    np.random.shuffle(random_order)
    X_train = X_train[random_order][0:train_size_data]
    y_train = y_train[random_order][0:train_size_data]
    print("x_train shape:", X_train.shape)
    print(X_train.shape[0], "train samples")

    # define graph
    with tf.name_scope("Inputs"):
        X_placeholder = tf.placeholder(tf.float32, [None, 28*28*1])
        Y_placeholder = tf.placeholder(tf.float32, [None, 10])

    predict = dnn_medium_model(X_placeholder)

    with tf.name_scope("Cross_entropy_loss"):
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y_placeholder * tf.log(predict), reduction_indices=[1]))
        tf.summary.scalar("cross_entropy", cross_entropy) # In tensorboard event

    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    # train
    with tf.Session() as sess:
        # Merge all the summaries and write them out to ./logs/
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(args.LOG_DIR_PATH, args.TRAIN_TIMES,"Tensorboard/train/"), sess.graph)

        sess.run(tf.global_variables_initializer())
        nbrof_batch = int(len(X_train)/batch_size)

        cross_entropy_epochs = []
        accuracy_epochs = []
        weights_record_epochs = [] # epochs/3, modle parameter
        for e in range(epochs):
            for i in range(nbrof_batch):
                batch_x, batch_y = X_train[i*batch_size: (i+1)*batch_size], y_train[i*batch_size: (i+1)*batch_size]
                feed_dict = {X_placeholder:batch_x, Y_placeholder:batch_y}

                _, summary, cross_entropy_ = sess.run([train_step, merged, cross_entropy], feed_dict=feed_dict)
                
                y_predict_  = sess.run(predict, feed_dict=feed_dict)
                correct_prediction_ = tf.equal(tf.argmax(y_predict_, 1), tf.argmax(batch_y, 1))
                tensor_accuracy = tf.reduce_mean(tf.cast(correct_prediction_, tf.float32))
                accuracy = sess.run(tensor_accuracy, feed_dict=feed_dict)

                train_writer.add_summary(summary, e)

                if i %10 == 0:
                    print("epochs:{}, steps:{}, loss={}, accuracy={}".format(e, (e*nbrof_batch) + i, cross_entropy_, accuracy))
            
            if e % args.COLLECT_WEIGHT_INTERVAL == 0:
                variables = np.asarray([])
                for v in tf.trainable_variables():
                    variables = np.concatenate((variables, sess.run(tf.reshape(v, [-1]))))
                
                weights_record_epochs.append(variables)
                cross_entropy_epochs.append(cross_entropy_)
                accuracy_epochs.append(accuracy)
        
        # save loss process
        cross_entropy_epochs = np.asarray(cross_entropy_epochs)
        loss_save_path = os.path.join(args.LOG_DIR_PATH, args.TRAIN_TIMES, "loss.npy")
        if not os.path.exists(os.path.dirname(loss_save_path)):
            os.makedirs(os.path.dirname(loss_save_path))
        np.save(loss_save_path, cross_entropy_epochs)
        print("Loss process to path: ", loss_save_path)

        # save predict process
        accuracy_epochs = np.asarray(accuracy_epochs)
        accuracy_save_path = os.path.join(args.LOG_DIR_PATH, args.TRAIN_TIMES, "accuracy.npy")
        if not os.path.exists(os.path.dirname(accuracy_save_path)):
            os.makedirs(os.path.dirname(accuracy_save_path))
        np.save(accuracy_save_path, accuracy_epochs)
        print("Accuracy process to path: ", accuracy_save_path)

        # save weight process
        weights_record_epochs = np.asarray(weights_record_epochs)
        weight_save_path = os.path.join(args.LOG_DIR_PATH, args.TRAIN_TIMES, "weights.npy")
        if not os.path.exists(os.path.dirname(weight_save_path)):
            os.makedirs(os.path.dirname(weight_save_path))
        np.save(weight_save_path, weights_record_epochs)
        print("Weights process to path: ", weight_save_path)

        # save model
        saver = tf.train.Saver()
        save_model_dir_path = os.path.join(args.SAVE_MODLE_DIR_PATH, args.TRAIN_TIMES, "model")
        if not os.path.exists(os.path.dirname(save_model_dir_path)):
            os.makedirs(os.path.dirname(save_model_dir_path))
        save_path = saver.save(sess, save_model_dir_path)
        print("Save model to path: {}".format(os.path.dirname(save_model_dir_path)))

if __name__ == "__main__":
    PROJECT_DIR_PATH = os.path.dirname(__file__)
    LOG_DIR_PATH = os.path.join(PROJECT_DIR_PATH, "logs")
    SAVE_MODLE_DIR_PATH = os.path.join(PROJECT_DIR_PATH, "save_models")
    COLLECT_WEIGHT_INTERVAL = 3
    TRAIN_TIMES = "Train_times_2"

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
        "--COLLECT_WEIGHT_INTERVAL",
        type=int,
        default=COLLECT_WEIGHT_INTERVAL,
        help=""
    )
    parser.add_argument(
        "--TRAIN_TIMES",
        type=str,
        default=TRAIN_TIMES,
        help=""
    )
    main(parser.parse_args())