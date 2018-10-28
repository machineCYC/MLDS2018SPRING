import tensorflow as tf


def dnn_medium_model(X_placeholder):

    dense = tf.layers.dense(inputs=X_placeholder, units=64
                            , kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
                            , activation=tf.nn.relu
                            , name="dense1")

    dense = tf.layers.dense(inputs=dense, units=64
                            , kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
                            , activation=tf.nn.relu
                            , name="dense2")

    dense = tf.layers.dense(inputs=dense, units=64
                            , kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
                            , activation=tf.nn.relu
                            , name="dense3")

    output = tf.layers.dense(inputs=dense, units=10
                            , kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
                            , activation=tf.nn.softmax
                            , name="output")
    
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