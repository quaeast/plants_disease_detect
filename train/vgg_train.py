import tensorflow as tf
import numpy as np
import model.vgg as vgg
from data_process.preprocess import read_test_set

vgg_ins = vgg.Vgg16()
test_set = read_test_set()

labels = test_set['labels']
photos = test_set['photos']

depth = 15
labels_one_hot = tf.one_hot(labels, depth)

# total: 1500

vgg = vgg.Vgg16()

with tf.Session() as sess:
    for i in range(30):
        # 50 a batch
        labels_batch = labels_one_hot[i*50: i*50+50]
        photos_batch = photos[i*50: i*50+50, 0:224, 0:224]
        loss = tf.losses.softmax_cross_entropy(onehot_labels=labels_batch, logits=vgg.get_result())
        optimazer = tf.train.AdamOptimizer(0.01).minimize(loss)
        sess.run(tf.global_variables_initializer())
        sess.run([optimazer, loss], feed_dict={vgg.img: photos_batch})
        print(i)
        print('>---------------------------------')

