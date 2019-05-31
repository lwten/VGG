"""
Simple tester for the vgg19_trainable
"""

import tensorflow as tf
import os
import cv2
import pandas as pd
import numpy as np
input_path = '../public_test_data'

# count= 0
batch = []
name = []
label_ = []
cnt = 0
# data = pd.read_csv("./train_face_value_label.csv")
# target_num_map = {0.1: 0, 0.2: 1, 0.5: 2, 1: 3, 2: 4, 5: 5, 10: 6, 50: 7, 100: 8}
# data[' label'] = data[' label'].apply(lambda x: target_num_map[x])
# dict = data.set_index('name').T.to_dict('list')
for image_file in os.listdir(input_path):
    cnt =cnt + 1
    if cnt % 10000 == 0:
        print(cnt)
    name.append(image_file)
    image = cv2.imread(input_path + '/' + image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (500, 250), interpolation=cv2.INTER_LINEAR)
    batch.append(image)
print(len(batch))
res = pd.DataFrame(name, columns=['name'])
cnt0 = 0
cnt1 = 0
with tf.Session() as sess:
    sess = tf.Session()

    saver = tf.train.import_meta_graph('./VGG16_model/model.ckpt.meta')
    saver.restore(sess, './VGG16_model/model.ckpt')
    # print([n.name for n in tf.get_default_graph().as_graph_def().node])
    inputs = tf.get_default_graph().get_tensor_by_name('Placeholder:0')
    train_mode = tf.get_default_graph().get_tensor_by_name('Placeholder_2:0')
    label = tf.get_default_graph().get_tensor_by_name('prob:0')
    # 使用y进行预测
    minibatch = []
    for i in range(len(batch)):
        minibatch.append(batch[i])
        if (i+1) % 50 == 0 or i == len(batch)-1:
            print(i+1)
            prob = sess.run(label, feed_dict={inputs: minibatch, train_mode: False})
            # print(prob.shape[0])
            target_num_map = {0:"0.1", 1:"0.2", 2:"0.5", 3:"1", 4:"2", 5: "5", 6:"10", 7:"50", 8:"100"}
            # target_num_map = {0: 0.1, 1: 0.2, 2: 0.5, 3: 1, 4: 2, 5: 5, 6: 10, 7: 50, 8: 100}
            for j in range(prob.shape[0]):
                pred = np.argsort(prob[j])[::-1]
                # print(prob[j])
                # print(pred)
                label_.append(target_num_map[pred[0]])
            minibatch = []

res['label'] = label_
print(res.head())
res.to_csv("result_1.csv", encoding='utf8', index=False)



