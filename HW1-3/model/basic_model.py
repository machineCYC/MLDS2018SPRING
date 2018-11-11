import tensorflow as tf


def simple_model(X_placeholder):

    conv = tf.layers.conv2d(inputs=X_placeholder, filters=16
                            , kernel_size=3
                            , kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
                            , activation=tf.nn.relu
                            , name="conv1")

    conv = tf.layers.conv2d(inputs=conv, filters=32
                            , kernel_size=3
                            , kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
                            , activation=tf.nn.relu
                            , name="conv2")

    conv = tf.layers.conv2d(inputs=conv, filters=64
                            , kernel_size=3
                            , kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
                            , activation=tf.nn.relu
                            , name="conv3")

    flatten = tf.layers.flatten(conv, name="flatten")

    dense = tf.layers.dense(inputs=flatten, units=128
                                , kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
                                , activation=tf.nn.relu
                                , name="dense1")

    output = tf.layers.dense(inputs=dense, units=10
                            , kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
                            , activation=tf.nn.softmax
                            , name="output")

    return output



def similar_model(X_placeholder, time):

    conv = tf.layers.conv2d(inputs=X_placeholder, filters=16
                            , kernel_size=3
                            , kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
                            , activation=tf.nn.relu
                            , name="conv1")

    conv = tf.layers.conv2d(inputs=conv, filters=32
                            , kernel_size=3
                            , kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
                            , activation=tf.nn.relu
                            , name="conv2")

    conv = tf.layers.conv2d(inputs=conv, filters=64
                            , kernel_size=3
                            , kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
                            , activation=tf.nn.relu
                            , name="conv3")

    flatten = tf.layers.flatten(conv, name="flatten")

    dense = tf.layers.dense(inputs=flatten, units=(times + 1)*10
                                , kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
                                , activation=tf.nn.relu
                                , name="dense1")

    output = tf.layers.dense(inputs=dense, units=10
                            , kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
                            , activation=tf.nn.softmax
                            , name="output")

    return output