import os
import time
import random
import numpy as np
import tensorflow as tf

class BaseLineModel(object):
    model_name = "BaseLine"

    def __init__(self, ):
        pass

    def discriminator(self, x):
        # x: 64*64*3
        conv1 = tf.layers.conv2d(
            inputs=x, filters=32, kernel_size=4, activation=tf.nn.relu, name="conv1")

        conv2 = tf.layers.conv2d(
            inputs=conv1, filters=64, kernel_size=4, activation=tf.nn.relu, padding="same", name="conv2")

        conv3 = tf.layers.conv2d(
            inputs=x, filters=128, kernel_size=4, activation=tf.nn.relu, name="conv3")

        conv4 = tf.layers.conv2d(
            inputs=conv3, filters=256, kernel_size=4, activation=tf.nn.relu, name="conv4")

        flatten = tf.layers.flatten(conv4, name="flatten")

        dense1 = tf.layers.dense(
            inputs=flatten, units=1, activation=tf.nn.sigmoid, name="dense1")

    def generator(self, x):
        dense1 = tf.layers.dense(
            inputs=x, units=128*16*16, activation=tf.nn.relu, name="dense1")

        reshape1 = tf.reshape(tensor=dense1, shape=(16, 16, 128), name="reshape1")

        # Upsampling
        upsampling1 = tf.nn.conv2d_transpose(reshape1, output_shape=(32, 32, 128), strides=2, name="upsampling1")
        # upsampling1 = tf.keras.layers.UpSampling2D()

        conv1 = tf.layers.conv2d(
            inputs=upsampling1, filters=128, kernel_size=4, activation=tf.nn.relu, name="conv1")

        # Upsampling
        upsampling2 = tf.nn.conv2d_transpose(conv1, output_shape=(32, 32, 128), strides=2, name="upsampling2")

        conv2 = tf.layers.conv2d(
            inputs=upsampling2, filters=64, kernel_size=4, activation=tf.nn.relu, name="conv2")

        conv3 = tf.layers.conv2d(
            inputs=conv2, filters=3, kernel_size=4, activation=tf.nn.tanh, name="conv3")

    def build(self):
        pass

    def train(self):
        pass

    def save(self):
        pass

    def load(self):
        pass