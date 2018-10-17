import tensorflow as tf


# define layer
def add_layer(inputs, in_size, out_size, activation_function = None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function == None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

def shallow_model(X_placeholder):

    dense = tf.layers.dense(inputs=X_placeholder, units=200, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01), activation=tf.nn.relu)

    output = tf.layers.dense(inputs=dense, units=1, kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

    return output


def medium_model(X_placeholder):

    dense = tf.layers.dense(inputs=X_placeholder, units=5, activation=tf.nn.relu)

    dense = tf.layers.dense(inputs=dense, units=20, activation=tf.nn.relu)

    dense = tf.layers.dense(inputs=dense, units=20, activation=tf.nn.relu)

    dense = tf.layers.dense(inputs=dense, units=5, activation=tf.nn.relu)

    output = tf.layers.dense(inputs=dense, units=1)

    return output


def deep_model(X_placeholder):

    dense = tf.layers.dense(inputs=X_placeholder, units=5, activation=tf.nn.relu)

    dense = tf.layers.dense(inputs=dense, units=5, activation=tf.nn.relu)

    dense = tf.layers.dense(inputs=dense, units=10, activation=tf.nn.relu)
    
    dense = tf.layers.dense(inputs=dense, units=10, activation=tf.nn.relu)

    dense = tf.layers.dense(inputs=dense, units=5, activation=tf.nn.relu)

    dense = tf.layers.dense(inputs=dense, units=5, activation=tf.nn.relu)

    output = tf.layers.dense(inputs=dense, units=1)

    return output