import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# Fix random seed for reproducibility.
seed = 777
np.random.seed(seed)
print("Fixed random seed for reproducibility.")

# target function sin(5x*pi) / 5x*pi
X_train = np.linspace(0.0001, 1.0, num=10000)
y_train = np.sinc(5 * X_train)

# visulization target function
plt.scatter(x=X_train, y=y_train, c="r")
plt.show()

# model parameters
batch_size = 128
epochs = 20000

X_placeholder = tf.placeholder(tf.float32, shape=(None, 1))
Y_placeholder = tf.placeholder(tf.float32, shape=(None, 1))

# define model

def shallow_model(X_placeholder):

    dense1 = tf.layers.dense(inputs=X_placeholder, units=200, activation=tf.nn.relu)

    output = tf.layers.dense(inputs=dense1, units=1)

    return output



