import tensorflow as tf


def shallow_model(X_placeholder):

    dense = tf.layers.dense(inputs=X_placeholder, units=190
                            , kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
                            , activation=tf.nn.relu
                            , name="dense1")

    output = tf.layers.dense(inputs=dense, units=1
                            , kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
                            , name="output")

    return output


def medium_model(X_placeholder):

    dense = tf.layers.dense(inputs=X_placeholder, units=10
                            , kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
                            , activation=tf.nn.relu
                            , name="dense1")

    dense = tf.layers.dense(inputs=dense, units=18
                            , kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
                            , activation=tf.nn.relu
                            , name="dense2")

    dense = tf.layers.dense(inputs=dense, units=15
                            , kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
                            , activation=tf.nn.relu
                            , name="dense3")

    dense = tf.layers.dense(inputs=dense, units=4
                            , kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
                            , activation=tf.nn.relu
                            , name="dense4")

    output = tf.layers.dense(inputs=dense, units=1
                            , kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
                            , name="output")

    return output


def deep_model(X_placeholder):

    dense = tf.layers.dense(inputs=X_placeholder, units=5
                            , kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
                            , activation=tf.nn.relu
                            , name="dense1")

    dense = tf.layers.dense(inputs=dense, units=10
                            , kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
                            , activation=tf.nn.relu
                            , name="dense2")
    
    dense = tf.layers.dense(inputs=dense, units=10
                            , kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
                            , activation=tf.nn.relu
                            , name="dense3")

    dense = tf.layers.dense(inputs=dense, units=10
                            , kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
                            , activation=tf.nn.relu
                            , name="dense4")

    dense = tf.layers.dense(inputs=dense, units=10
                            , kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
                            , activation=tf.nn.relu
                            , name="dense5")
    
    dense = tf.layers.dense(inputs=dense, units=10
                            , kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
                            , activation=tf.nn.relu
                            , name="dense6")

    dense = tf.layers.dense(inputs=dense, units=5
                            , kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
                            , activation=tf.nn.relu
                            , name="dense7")

    output = tf.layers.dense(inputs=dense, units=1
                            , kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
                            , name="output")

    return output


def cnn_shallow_model(X_placeholder):

    conv = tf.layers.conv2d(inputs=X_placeholder, filters=15
                            , kernel_size=3
                            , kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
                            , activation=tf.nn.relu
                            , name="conv1")

    flatten = tf.layers.flatten(conv, name="flatten")

    output = tf.layers.dense(inputs=flatten, units=10, name="output", activation=tf.nn.softmax)
    
    return output


def cnn_medium_model(X_placeholder):

    conv = tf.layers.conv2d(inputs=X_placeholder, filters=10
                            , kernel_size=3
                            , kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
                            , activation=tf.nn.relu
                            , name="conv1")

    conv = tf.layers.conv2d(inputs=conv, filters=16
                            , kernel_size=3
                            , kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
                            , activation=tf.nn.relu
                            , name="conv2")

    conv = tf.layers.conv2d(inputs=conv, filters=20
                            , kernel_size=3
                            , kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
                            , activation=tf.nn.relu
                            , name="conv3")

    flatten = tf.layers.flatten(conv, name="flatten")

    output = tf.layers.dense(inputs=flatten, units=10, name="output", activation=tf.nn.softmax)
    
    return output


def cnn_deep_model(X_placeholder):

    conv = tf.layers.conv2d(inputs=X_placeholder, filters=8
                            , kernel_size=3
                            , kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
                            , activation=tf.nn.relu
                            , name="conv1")
    
    conv = tf.layers.conv2d(inputs=conv, filters=12
                            , kernel_size=3
                            , kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
                            , activation=tf.nn.relu
                            , name="conv2")

    conv = tf.layers.conv2d(inputs=conv, filters=16
                            , kernel_size=3
                            , kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
                            , activation=tf.nn.relu
                            , name="conv3")

    conv = tf.layers.conv2d(inputs=conv, filters=20
                            , kernel_size=3
                            , kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
                            , activation=tf.nn.relu
                            , name="conv4")
    
    conv = tf.layers.conv2d(inputs=conv, filters=28
                            , kernel_size=3
                            , kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
                            , activation=tf.nn.relu
                            , name="conv5")

    flatten = tf.layers.flatten(conv, name="flatten")

    output = tf.layers.dense(inputs=flatten, units=10, name="output", activation=tf.nn.softmax)
    
    return output