import os
import argparse
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.datasets import mnist

from model.basic_model import simple_model


def main(args):
    # parameters
    batch_size = 128
    epochs = 500
    learning_rate = 1e-4
    train_size_data = 5000

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
    X_train = X_train[0:train_size_data]
    y_train = y_train[random_order][0:train_size_data] # random label

    X_valid = X_valid[0:train_size_data]
    y_valid = y_valid[0:train_size_data]
    print("x_train shape:", X_train.shape)
    print(X_train.shape[0], "train samples")
    print("X_valid shape:", X_valid.shape)
    print(X_valid.shape[0], "valid samples")

    # define graph
    with tf.name_scope("Inputs"):
        X_placeholder = tf.placeholder(tf.float32, [None, 28, 28, 1])
        Y_placeholder = tf.placeholder(tf.float32, [None, 10])

    logits = simple_model(X_placeholder)

    with tf.name_scope("Cross_entropy_loss"):
        # cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y_placeholder * tf.log(predict), reduction_indices=[1]))
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits, 
                labels=Y_placeholder
            )
        )
        tf.summary.scalar("cross_entropy", cross_entropy) # In tensorboard event

    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

    with tf.name_scope("accuracy"):
        accuracy = tf.reduce_mean(
            tf.cast(
                tf.equal(
                    tf.argmax(Y_placeholder, axis=1),
                    tf.argmax(logits, axis=1)
                ),
                tf.float32
            )
        )

    # train
    with tf.Session() as sess:
        # Merge all the summaries and write them out to ./logs/
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(args.LOG_DIR_PATH, "Random_label", "Tensorboard/train/"), sess.graph)

        sess.run(tf.global_variables_initializer())
        nbrof_batch = int(len(X_train) / batch_size)

        train_cross_entropy_list = []
        train_accuracy_list = []
        valid_cross_entropy_list = []
        valid_accuracy_list = []
        for e in range(epochs):
            for i in range(nbrof_batch):
                batch_x, batch_y = X_train[i * batch_size: (i+1) * batch_size], y_train[i * batch_size: (i+1) * batch_size]
                feed_dict = {X_placeholder: batch_x, Y_placeholder: batch_y}

                _, summary, train_cross_entropy_ = sess.run([train_step, merged, cross_entropy], feed_dict=feed_dict)
                
                train_accuracy_ = sess.run(accuracy, feed_dict=feed_dict)
                train_writer.add_summary(summary, e)

                if i %10 == 0:
                    print("epochs:{}, steps:{}, loss={}, accuracy={}".format(e, (e*nbrof_batch) + i, train_cross_entropy_, train_accuracy_))
            
            # valid model
            random_choice = np.random.randint(0, len(X_valid), size=batch_size) # need check
            batch_x, batch_y = X_valid[random_choice], y_valid[random_choice]
            feed_dict = {X_placeholder: batch_x, Y_placeholder: batch_y}

            valid_accuracy_ = sess.run(accuracy, feed_dict=feed_dict)
            valid_cross_entropy_ = sess.run(cross_entropy, feed_dict=feed_dict)

            print("epochs:{}, steps:{}, valid loss={}, valid accuracy={}".format(e, (e*nbrof_batch) + i, valid_cross_entropy_, valid_accuracy_))

            train_cross_entropy_list.append(train_cross_entropy_)
            train_accuracy_list.append(train_accuracy_)

            valid_cross_entropy_list.append(valid_cross_entropy_)
            valid_accuracy_list.append(valid_accuracy_)

        # save loss process
        train_cross_entropy_epochs = np.asarray(train_cross_entropy_list)
        train_loss_save_path = os.path.join(args.LOG_DIR_PATH, "Random_label", "train_loss.npy")
        if not os.path.exists(os.path.dirname(train_loss_save_path)):
            os.makedirs(os.path.dirname(train_loss_save_path))
        np.save(train_loss_save_path, train_cross_entropy_epochs)
        print("Train loss process to path: ", train_loss_save_path)

        valid_cross_entropy_epochs = np.asarray(valid_cross_entropy_list)
        valid_loss_save_path = os.path.join(args.LOG_DIR_PATH, "Random_label", "valid_loss.npy")
        if not os.path.exists(os.path.dirname(valid_loss_save_path)):
            os.makedirs(os.path.dirname(valid_loss_save_path))
        np.save(valid_loss_save_path, valid_cross_entropy_epochs)
        print("Valid loss process to path: ", valid_loss_save_path)


        # save accuracy process
        train_accuracy_epochs = np.asarray(train_accuracy_list)
        train_accuracy_save_path = os.path.join(args.LOG_DIR_PATH, "Random_label", "train_accuracy.npy")
        if not os.path.exists(os.path.dirname(train_accuracy_save_path)):
            os.makedirs(os.path.dirname(train_accuracy_save_path))
        np.save(train_accuracy_save_path, train_accuracy_epochs)
        print("Train accuracy process to path: ", train_accuracy_save_path)
        
        valid_accuracy_epochs = np.asarray(valid_accuracy_list)
        valid_accuracy_save_path = os.path.join(args.LOG_DIR_PATH, "Random_label", "valid_accuracy.npy")
        if not os.path.exists(os.path.dirname(valid_accuracy_save_path)):
            os.makedirs(os.path.dirname(valid_accuracy_save_path))
        np.save(valid_accuracy_save_path, valid_accuracy_epochs)
        print("Valid accuracy process to path: ", valid_accuracy_save_path)


        # save model
        saver = tf.train.Saver()
        save_model_dir_path = os.path.join(args.SAVE_MODLE_DIR_PATH, "Random_label", "model")
        if not os.path.exists(os.path.dirname(save_model_dir_path)):
            os.makedirs(os.path.dirname(save_model_dir_path))
        save_path = saver.save(sess, save_model_dir_path)
        print("Save model to path: {}".format(os.path.dirname(save_model_dir_path)))


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