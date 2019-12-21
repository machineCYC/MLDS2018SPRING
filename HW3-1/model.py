import os
import random
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Dropout, Flatten, UpSampling2D, Reshape
from src.data_loader import GenerateDataSet
class BaseLineModel(object):

    def __init__(self, img_width, img_height, img_channel,
        batch_size, learning_rate, beta1, max_epoch, noise_dim):
        self.img_width = img_width
        self.img_height = img_height
        self.img_channel = img_channel
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.noise_dim = noise_dim

    def discriminator(self, x, reuse=False):
        # if reuse:
        #     tf.get_variable_scope().reuse_variables()
        # x: 64*64*3
        # -->(None, 64,64,32)
        conv1 = tf.layers.conv2d(
            inputs=x, filters=32, kernel_size=4, activation=tf.nn.relu, padding='same', name="d_conv1")

        # -->(None, 64,64,64)
        conv2 = tf.layers.conv2d(
            inputs=conv1, filters=64, kernel_size=4, activation=tf.nn.relu, padding="same", name="d_conv2")

        # -->(None, 64,64,128)
        conv3 = tf.layers.conv2d(
            inputs=conv2, filters=128, kernel_size=4, activation=tf.nn.relu, padding='same', name="d_conv3")

        # -->(None, 64,64,256)
        conv4 = tf.layers.conv2d(
            inputs=conv3, filters=256, kernel_size=4, activation=tf.nn.relu, padding='same', name="d_conv4")

        # -->(None, 64*64*128)
        flatten = tf.layers.flatten(conv4, name="d_flatten")

        # -->(None, 1)
        dense1 = tf.layers.dense(
            inputs=flatten, units=1, activation=tf.nn.sigmoid, name="d_dense1")

        return dense1


    def generator(self, x):
        # -->(None, 32768)
        dense1 = tf.layers.dense(
            inputs=x, units=128*16*16, activation=tf.nn.relu, name="g_dense1")

        # -->(None, 16,16,128)
        reshape1 = tf.reshape(tensor=dense1, shape=[-1, 16, 16, 128], name="g_reshape1")

        # Upsampling
        # -->(None, 32,32,128)
        upsampling1 = tf.layers.conv2d_transpose(
            reshape1, filters=128, kernel_size=2, strides=2, name="g_upsampling1")

        # -->(None, 32,32,128)
        conv1 = tf.layers.conv2d(
            inputs=upsampling1, filters=128, kernel_size=4, activation=tf.nn.relu,
            padding='same', name="g_conv1")

        # Upsampling
        # -->(None, 64,64,128)
        upsampling2 = tf.layers.conv2d_transpose(
            conv1, filters=128, kernel_size=2, strides=2, name="g_upsampling2")

        # -->(None, 64,64,64)
        conv2 = tf.layers.conv2d(
            inputs=upsampling2, filters=64, kernel_size=4, activation=tf.nn.relu,
            padding='same', name="g_conv2")

        # -->(None, 64,64,3)
        conv3 = tf.layers.conv2d(
            inputs=conv2, filters=3, kernel_size=4, activation=tf.nn.tanh,
            padding='same', name="g_conv3")

        return conv3

    def build(self):
        real_images = tf.placeholder(tf.float32,
            [None, self.img_width, self.img_height, self.img_channel], name='image_placeholder')
        noise = tf.placeholder(tf.float32, [None, self.noise_dim], name='noise_placeholder')

        with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
            fake_images = self.generator(noise)

        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
            fake_predicts = self.discriminator(fake_images)
            real_predicts = self.discriminator(real_images)

        with tf.variable_scope("loss", reuse=tf.AUTO_REUSE):
            # FIXME: sigmoid_cross_entropy_with_logits
            d_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=real_predicts, labels=tf.ones_like(real_predicts)))
            d_loss_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=fake_predicts, labels=tf.zeros_like(fake_predicts)))
            d_loss = d_loss_real + d_loss_fake

            g_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=fake_predicts, labels=tf.ones_like(fake_predicts)))

        all_var = tf.trainable_variables()
        d_vars = [var for var in all_var if 'd_' in var.name]
        g_vars = [var for var in all_var if 'g_' in var.name]

        clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in d_vars]

        inputs = {
            'real_image': real_images,
            'noise': noise
        }

        variables = {
            'd_vars': d_vars,
            'g_vars': g_vars,
            'clip_D': clip_D
        }

        loss = {
            'g_loss': g_loss,
            'd_loss': d_loss,
        }

        return inputs, variables, loss

    def train(self, image_dir):
        graph = tf.Graph()
        with graph.as_default():
            inputs, variables, loss = self.build()

            with tf.variable_scope("optimizer", reuse=tf.AUTO_REUSE):
                # FIXME:
                d_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                    .minimize(loss['d_loss'], var_list=variables['d_vars'])

                g_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                    .minimize(loss['g_loss'], var_list=variables['g_vars'])

        real_images = GenerateDataSet(image_dir, self.img_width, self.img_height, self.img_channel)

        init = tf.global_variables_initializer()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        with tf.Session(graph=graph,
            config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)) as sess:

            sess.run(init)

            epoch = -1
            while epoch < self.max_epoch:
                real_image = data.next_batch(batch_size=self.batch_size)
                noise = np.random.uniform(-1, 1, [self.batch_size, self.noise_dim])

                sess.run(d_optimizer, feed_dict={
                    inputs['real_image']: real_images,
                    inputs['noise']: noise
                })

                sess.run(g_optimizer, feed_dict={
                    inputs['real_image']: real_images,
                    inputs['noise']: noise
                })

    def infer():
        pass

    def save(self):
        pass

    def load(self):
        pass