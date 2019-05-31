"""
Simple tester for the vgg19_trainable
"""

import tensorflow as tf
import vgg16_trainable as vgg16
import utils
import os

os.environ['CUDA_VISIBLE_DEVICES']='2'
with tf.Session() as sess:

    input = tf.placeholder(tf.float32, [None, 250, 500, 3])
    label = tf.placeholder(tf.int32, [None, ])
    train_mode = tf.placeholder(tf.bool)

    vgg = vgg16.Vgg16('./vgg16.npy')
    vgg.build(input, train_mode)

    # print number of variables used:
    sess.run(tf.global_variables_initializer())


    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=vgg.prob, labels=label))
    optimizer = tf.train.GradientDescentOptimizer(0.0002).minimize(loss)

    correct_prediction = tf.equal(tf.cast(label, tf.int32), tf.cast(tf.argmax(vgg.prob, 1), tf.int32))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 创建saver
    saver = tf.train.Saver()
    for i in range(50):
        batch_images, batch_labels = utils.get_batch_data('./train_data', "./train_face_value_label.csv")
        train_dict = {input: batch_images, label: batch_labels, train_mode: True}
        sess.run(optimizer, feed_dict=train_dict)

        loss_, acc_ = sess.run([loss, accuracy], feed_dict=train_dict)

        train_text = 'step: {}, loss: {}, acc: {}'.format(i + 1, loss_, acc_)
        print(train_text)

    saver.save(sess, "./VGG16_model/model.ckpt")
