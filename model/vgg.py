import tensorflow as tf
import numpy as np


class Vgg16(object):

    def __init__(self):
        self.img = tf.placeholder(shape=[None, 224, 224, 3], dtype=tf.float32)

        self.conv1_1 = tf.layers.conv2d(inputs=self.img, filters=64, kernel_size=(3, 3), padding='same')
        self.conv1_2 = tf.layers.conv2d(inputs=self.conv1_1, filters=64, kernel_size=(3, 3), padding='same')
        self.max_pool1 = self.get_max_pool(self.conv1_2)

        self.conv2_1 = tf.layers.conv2d(inputs=self.max_pool1, filters=128, kernel_size=(3, 3), padding='same')
        self.conv2_2 = tf.layers.conv2d(inputs=self.conv2_1, filters=128, kernel_size=(3, 3), padding='same')
        self.max_pool2 = self.get_max_pool(self.conv2_2)

        self.conv3_1 = tf.layers.conv2d(inputs=self.max_pool2, filters=256, kernel_size=(3, 3), padding='same')
        self.conv3_2 = tf.layers.conv2d(inputs=self.conv3_1, filters=256, kernel_size=(3, 3), padding='same')
        self.conv3_3 = tf.layers.conv2d(inputs=self.conv3_2, filters=256, kernel_size=(3, 3), padding='same')
        self.max_pool3 = self.get_max_pool(self.conv3_3)

        self.conv4_1 = tf.layers.conv2d(inputs=self.max_pool3, filters=512, kernel_size=(3, 3), padding='same')
        self.conv4_2 = tf.layers.conv2d(inputs=self.conv4_1, filters=512, kernel_size=(3, 3), padding='same')
        self.conv4_3 = tf.layers.conv2d(inputs=self.conv4_1, filters=512, kernel_size=(3, 3), padding='same')
        self.max_pool4 = self.get_max_pool(self.conv4_3)

        self.conv5_1 = tf.layers.conv2d(inputs=self.max_pool4, filters=512, kernel_size=(3, 3), padding='same')
        self.conv5_2 = tf.layers.conv2d(inputs=self.conv5_1, filters=512, kernel_size=(3, 3), padding='same')
        self.conv5_3 = tf.layers.conv2d(inputs=self.conv5_2, filters=512, kernel_size=(3, 3), padding='same')
        self.max_pool5 = self.get_max_pool(self.conv5_3)

        self.max_pool5_plain = tf.reshape(self.max_pool5, [-1, 25088])

        self.fc6 = self.get_relu_layer(self.max_pool5_plain, 4096)
        self.fc7 = self.get_relu_layer(self.fc6, 4096)
        self.fc8 = self.get_relu_layer(self.fc7, 1000)

    def get_max_pool(self, x):
        return tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def get_relu_layer(self, x, out_d):
        in_d = int(x.shape[1])
        weights = tf.random_normal(shape=[in_d, out_d])
        bias = tf.random_normal(shape=[out_d])
        return tf.nn.relu_layer(x=x, weights=weights, biases=bias)

    def get_feature_map(self):
        return self.max_pool5

    def get_result(self):
        return self.fc8



if __name__ == '__main__':
    vgg = Vgg16()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('initialized')
        print(sess.run(fetches=vgg.get_feature_map(), feed_dict={vgg.img: np.ones(dtype=np.float32, shape=[1, 224, 224, 3])}))
