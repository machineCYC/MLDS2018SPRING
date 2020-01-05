import os
import time
import random
import scipy.misc
import numpy as np
import tensorflow as tf

from keras.models import Sequential
from src.data_loader import GenerateDataSet
from src.file import check_Directory_Exists

class BaseLineModel(object):

    def __init__(self, img_width, img_height, img_channel,
        batch_size, learning_rate, beta1, max_epoch, noise_dim,
        gpu_memory_fraction):
        self.img_width = img_width
        self.img_height = img_height
        self.img_channel = img_channel
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.noise_dim = noise_dim
        self.gpu_memory_fraction = gpu_memory_fraction
        self.model_file = r'D:/workspace/MLDS2018SPRING/HW3-1/log/model/'

    def discriminator(self, x, reuse=False):
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
            inputs=flatten, units=1, name="d_dense1")

        return dense1, tf.nn.sigmoid(dense1)


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
        graph = tf.Graph()
        with graph.as_default():

            real_images = tf.placeholder(tf.float32,
                [None, self.img_width, self.img_height, self.img_channel], name='image_placeholder')
            noise = tf.placeholder(tf.float32, [None, self.noise_dim], name='noise_placeholder')

            with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
                self.fake_images = self.generator(noise)

            with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
                fake_predicts_logits, fake_predicts = self.discriminator(self.fake_images)
                real_predicts_logits, real_predicts = self.discriminator(real_images)

            with tf.variable_scope("loss", reuse=tf.AUTO_REUSE):
                # FIXME: sigmoid_cross_entropy_with_logits
                d_loss_real = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=real_predicts_logits, labels=tf.ones_like(real_predicts)))
                d_loss_fake = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=fake_predicts_logits, labels=tf.zeros_like(fake_predicts)))
                d_loss = d_loss_real + d_loss_fake

                g_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=fake_predicts, labels=tf.ones_like(fake_predicts)))

            all_var = tf.trainable_variables()
            d_vars = [var for var in all_var if 'd_' in var.name]
            g_vars = [var for var in all_var if 'g_' in var.name]

            clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in d_vars]

            with tf.variable_scope("optimizer", reuse=tf.AUTO_REUSE):
                d_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                    .minimize(d_loss, var_list=d_vars)

                g_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                    .minimize(g_loss, var_list=g_vars)

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

        optim = {
            'g_optimizer': g_optimizer,
            'd_optimizer': d_optimizer,
        }
        return inputs, variables, loss, optim, graph

    def train(self, image_dir):
        inputs, variables, loss, optim, graph = self.build()

        data = GenerateDataSet(image_dir, self.img_width, self.img_height, self.img_channel)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_memory_fraction)
        with tf.Session(graph=graph,
            config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            epoch = -1
            start_time = time.time()
            while epoch < self.max_epoch:
                real_image = data.next_batch(batch_size=self.batch_size)
                noise = np.random.uniform(-1, 1, [self.batch_size, self.noise_dim])

                sess.run(optim['d_optimizer'], feed_dict={
                    inputs['real_image']: real_image,
                    inputs['noise']: noise
                })

                sess.run(optim['g_optimizer'], feed_dict={
                    inputs['real_image']: real_image,
                    inputs['noise']: noise
                })

                if epoch != data.N_epoch:
                    epoch = data.N_epoch
                    self.save(sess, self.model_file)

                    used_time = time.time() - start_time
                    start_time = time.time()

                    d_loss, g_loss = sess.run(
                        [loss['d_loss'], loss['g_loss']],
                        feed_dict={
                            inputs['real_image']: real_image,
                            inputs['noise']: noise
                        })

                    print(str(epoch) + '/' + str(self.max_epoch) + ' epoch: ' +
                        'd_loss = ' + str(d_loss) + ' ' +
                        'g_loss = ' + str(g_loss) + ' ' +
                        'time = ' + str(used_time) + ' secs')

    def infer(self):
        sample_path = check_Directory_Exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'samples'))
        generate_images = []

        inputs, variables, loss, optim, graph = self.build()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.gpu_memory_fraction)
        with tf.Session(graph=graph,
            config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            self.load(sess, self.model_file)

            noise = np.random.uniform(-1, 1, [self.batch_size, self.noise_dim])
            generate_images.append(
                sess.run(self.fake_images, feed_dict={inputs['noise']: noise})
            )

        for i, batch_images in enumerate(generate_images, start=1):
            for j, image in enumerate(batch_images, start=1):
                scipy.misc.imsave(os.path.join(sample_path, 'sample_{}_{}.jpg'.format(i, j)), image)

    def save(self, sess, model_file):
        model_file = check_Directory_Exists(model_file)
        saver = tf.train.Saver()
        saver.save(sess, model_file)
        return

    def load(self, sess, model_file):
        if os.path.isdir(model_file):
            saver = tf.train.Saver()
            saver.restore(sess, model_file)