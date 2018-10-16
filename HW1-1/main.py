import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from model.basic_model import shallow_model
from model.basic_model import medium_model
from model.basic_model import deep_model


# Fix random seed for reproducibility.
seed = 416
np.random.seed(seed)
print("Fixed random seed for reproducibility.")

# target function sin(5x*pi) / 5x*pi
X_train = np.linspace(0.0001, 1.0, num=10000)[:, np.newaxis]
np.random.shuffle(X_train)
y_train = np.sinc(5 * X_train)

# visulization target function
# plt.scatter(x=X_train, y=y_train, c="r", marker="o")
# plt.show()

# model parameters
batch_size = 128
epochs = 20000
learning_rate = 0.01


X_placeholder = tf.placeholder(tf.float32, [None, 1])
Y_placeholder = tf.placeholder(tf.float32, [None, 1])

Weights1 = tf.Variable(tf.random_normal([1, 200]))
biases1 = tf.Variable(tf.zeros([1, 200]) + 0.1)

Weights2 = tf.Variable(tf.random_normal([200, 1]))
biases2 = tf.Variable(tf.zeros([1, 1]) + 0.1)

dense = tf.matmul(X_placeholder, Weights1) + biases1
dense = tf.nn.selu(dense)
predict = tf.matmul(dense, Weights2) + biases2

# predict = shallow_model(X_placeholder)

mse_loss = tf.reduce_mean(tf.square(predict - Y_placeholder))

# train step
train_step = tf.train.AdamOptimizer(learning_rate).minimize(mse_loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


mes_loss_list = []
for i in range(epochs):
    batch_x, batch_y = X_train[i*batch_size: (i+1)*batch_size], y_train[i*batch_size: (i+1)*batch_size]
    _, mse_loss_ = sess.run([train_step, mse_loss], feed_dict={X_placeholder:batch_x, Y_placeholder:batch_y})
    W1_, b1_ = sess.run([Weights1, biases1])
    if i %50 == 0:
        # loss = sess.run(mse_loss, feed_dict={X_placeholder:batch_x, Y_placeholder:batch_y})
        mes_loss_list.append(mse_loss_)
        print(mse_loss_)
        print(np.mean(W1_))
        print(np.mean(b1_))