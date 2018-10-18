import tensorflow as tf


def shallow_model(X_placeholder):

    dense = tf.layers.dense(inputs=X_placeholder, units=200
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

    dense = tf.layers.dense(inputs=dense, units=15
                            , kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
                            , activation=tf.nn.relu
                            , name="dense2")

    dense = tf.layers.dense(inputs=dense, units=15
                            , kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
                            , activation=tf.nn.relu
                            , name="dense3")

    dense = tf.layers.dense(inputs=dense, units=10
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
                            , activation=tf.nn.relu)

    dense = tf.layers.dense(inputs=dense, units=5
                            , kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
                            , activation=tf.nn.relu)

    dense = tf.layers.dense(inputs=dense, units=10
                            , kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
                            , activation=tf.nn.relu)
    
    dense = tf.layers.dense(inputs=dense, units=10
                            , kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
                            , activation=tf.nn.relu)

    dense = tf.layers.dense(inputs=dense, units=10
                            , kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
                            , activation=tf.nn.relu)

    dense = tf.layers.dense(inputs=dense, units=10
                            , kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
                            , activation=tf.nn.relu)
    
    dense = tf.layers.dense(inputs=dense, units=10
                            , kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
                            , activation=tf.nn.relu)

    dense = tf.layers.dense(inputs=dense, units=5
                            , kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
                            , activation=tf.nn.relu)

    output = tf.layers.dense(inputs=dense, units=1
                            , kernel_initializer=tf.truncated_normal_initializer(stddev=0.01)
                            , name="output")

    return output