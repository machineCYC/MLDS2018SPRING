import os
import argparse
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.datasets import mnist
import time
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
        # X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
        X_train = X_train.reshape(X_train.shape[0], 1*28*28)
    else:
        # X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
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
        # Merge all the summaries and write them out to ./logs/Tensorboard/train/
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(args.LOG_DIR_PATH, "Visulize_grads/Tensorboard/train/"), sess.graph)

        sess.run(tf.global_variables_initializer())
        nbrof_batch = int(len(X_train) / batch_size)

        cross_entropy_steps = []
        accuracy_steps = []
        grads_steps = []
        for e in range(epochs):
            for i in range(nbrof_batch):
                step_start = time.time()

                batch_x, batch_y = X_train[i*batch_size: (i+1)*batch_size], y_train[i*batch_size: (i+1)*batch_size]
                feed_dict = {X_placeholder:batch_x, Y_placeholder:batch_y}

                _, summary, cross_entropy_ = sess.run([train_step, merged, cross_entropy], feed_dict=feed_dict)
                
                y_predict_ = sess.run(predict, feed_dict=feed_dict)
                correct_prediction_ = tf.equal(tf.argmax(y_predict_, 1), tf.argmax(batch_y, 1))
                tensor_accuracy = tf.reduce_mean(tf.cast(correct_prediction_, tf.float32))
                accuracy = sess.run(tensor_accuracy, feed_dict=feed_dict)

                train_writer.add_summary(summary, e)
            
                grads_total = 0
                for variable in tf.trainable_variables():
                    [grad] = sess.run(tf.gradients(ys=cross_entropy, xs=variable), feed_dict=feed_dict) # [] 是取 list 中的值
                    grad_norm = np.sum(grad ** 2)
                    grads_total += grad_norm
                grads_total = grads_total ** 0.5

                step_end = time.time()
                if i %10 == 0:
                    print("time:{} sec, epochs:{}, steps:{}, loss={}, accuracy={}, grads_norm={}".format(step_end-step_start, e, (e*nbrof_batch) + i, cross_entropy_, accuracy, grads_total))

                cross_entropy_steps.append(cross_entropy_)
                accuracy_steps.append(accuracy)
                grads_steps.append(grads_total)
        
        # save loss process
        cross_entropy_steps = np.asarray(cross_entropy_steps)
        loss_save_path = os.path.join(args.LOG_DIR_PATH, "Visulize_grads", "loss.npy")
        if not os.path.exists(os.path.dirname(loss_save_path)):
            os.makedirs(os.path.dirname(loss_save_path))
        np.save(loss_save_path, cross_entropy_steps)
        print("Loss process to path: ", loss_save_path)

        # save accuracy process
        accuracy_steps = np.asarray(accuracy_steps)
        accuracy_save_path = os.path.join(args.LOG_DIR_PATH, "Visulize_grads", "accuracy.npy")
        if not os.path.exists(os.path.dirname(accuracy_save_path)):
            os.makedirs(os.path.dirname(accuracy_save_path))
        np.save(accuracy_save_path, accuracy_steps)
        print("Accuracy process to path: ", accuracy_save_path)

        # save grads process
        grads_steps = np.asarray(grads_steps)
        grads_save_path = os.path.join(args.LOG_DIR_PATH, "Visulize_grads", "grads.npy")
        if not os.path.exists(os.path.dirname(grads_save_path)):
            os.makedirs(os.path.dirname(grads_save_path))
        np.save(grads_save_path, grads_steps)
        print("Accuracy process to path: ", grads_save_path)

        # save model
        saver = tf.train.Saver()
        save_model_dir_path = os.path.join(args.SAVE_MODLE_DIR_PATH, "Visulize_grads", "model")
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