#coding=utf-8

import os
import time
import shutil
import random
#图像读取库
from PIL import Image
from skimage import transform,io
import matplotlib.pyplot as plt  
import matplotlib.image as mpimg 
#矩阵运算库
import numpy as np
import tensorflow as tf
import check_mohu
######
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
#####

# 数据文件夹
data_dir = "./traindata/"
#test_dir = './testdata/'
test_dir = '/home/wangzj/WORK/shuibiao/data_shuibiao/shuibiao_labelEdit/relabel_test_easy/'
# 模型文件路径
model_path = "model/image_model"
train = False
#
log_dir = './log'

# 从文件夹读取图片和标签到numpy数组中
# 标签信息在文件名中，例如1_40.jpg表示该图片的标签为1
height = 75
width = 50
channels = 3
n_inputs = height * width

conv1_fmaps = 32
conv1_ksize = 5
conv1_stride = 1 
conv1_pad = "SAME"

conv2_fmaps = 64
conv2_ksize = 3
conv2_stride = 1
conv2_pad = "SAME"

conv3_fmaps = 128
conv3_ksize = 3
conv3_stride = 1
conv3_pad = "SAME"

conv4_fmaps = 256
conv4_ksize = 3
conv4_stride = 1
conv4_pad = "SAME"


#X_dropout_rate = 0
pool4_dropout_rate =0.25

n_fc1 = 1024
fc1_dropout_rate = 0
n_outputs = 20
#learning_rate = 0.001

sess = tf.InteractiveSession(config=config)

with tf.name_scope("input"):
    X = tf.placeholder(tf.float32, shape=[None, n_inputs, channels], name="X")
    X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
    y = tf.placeholder(tf.int32, shape=[None], name="y")
    training = tf.placeholder_with_default(False, shape=[], name='training')

# 保存图像信息
with tf.name_scope('input_reshape'):
    tf.summary.image('input', X_reshaped, 10)

with tf.name_scope('cnn'):
#    X_reshaped_drop = tf.layers.dropout(X_reshaped, X_dropout_rate, training=training)
    conv1 = tf.layers.conv2d(X_reshaped, filters=conv1_fmaps, kernel_size=conv1_ksize,
                         strides=conv1_stride, padding=conv1_pad,
                         activation=None, name="conv1")
#                         , kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
#                         ,kernel_regularizer = max_norm_reg)
    conv1_bn = tf.layers.batch_normalization(conv1,training=training)
    conv1_bn_act = tf.nn.relu(conv1_bn)
#    conv1_bn_act = tf.nn.relu(conv1)
    pool1 = tf.nn.max_pool(conv1_bn_act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    conv2 = tf.layers.conv2d(pool1, filters=conv2_fmaps, kernel_size=conv2_ksize,
                         strides=conv2_stride, padding=conv2_pad,
                         activation=None, name="conv2")
#                         ,kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
#                         ,kernel_regularizer = max_norm_reg)
    conv2_bn = tf.layers.batch_normalization(conv2,training=training)
    conv2_bn_act = tf.nn.relu(conv2_bn)
#    conv2_bn_act = tf.nn.relu(conv2)
    pool2 = tf.nn.max_pool(conv2_bn_act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    conv3 = tf.layers.conv2d(pool2, filters=conv3_fmaps, kernel_size=conv3_ksize,
                         strides=conv3_stride, padding=conv3_pad,
                         activation=None, name="conv3")
    conv3_bn = tf.layers.batch_normalization(conv3,training=training)
    conv3_bn_act = tf.nn.relu(conv3_bn)

    conv4 = tf.layers.conv2d(conv3_bn_act, filters=conv4_fmaps, kernel_size=conv4_ksize,
                         strides=conv4_stride, padding=conv4_pad,
                         activation=None, name="conv4")
    conv4_bn = tf.layers.batch_normalization(conv4,training=training)
    conv4_bn_act = tf.nn.relu(conv4_bn)
    pool4 = tf.nn.max_pool(conv4_bn_act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME") 



    pool4_flat = tf.layers.flatten(pool4)
    pool4_flat_drop = tf.layers.dropout(pool4_flat, pool4_dropout_rate, training=training)
    fc1 = tf.layers.dense(pool4_flat_drop, n_fc1, activation=None
            , name="fc1")
#            ,kernel_regularizer = max_norm_reg)
#    fc1_bn =  tf.layers.batch_normalization(fc1,training=training)
#    fc1_bn_act = tf.nn.relu(fc1_bn)
    fc1_act = tf.nn.relu(fc1)
    fc1_drop = tf.layers.dropout(fc1_act, fc1_dropout_rate, training=training)

with tf.name_scope("output"):
    logits = tf.layers.dense(fc1_drop, n_outputs, name="output")
    Y_proba = tf.nn.softmax(logits, name="Y_proba")


with tf.name_scope("loss"):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.one_hot(y, n_outputs))
    with tf.name_scope("total"):
        loss = tf.reduce_mean(xentropy)
    tf.summary.scalar('loss',loss)

with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer()
#    optimizer = tf.train.AdamOptimizer(learning_rate)
#    training_op = optimizer.minimize(loss)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops): #要先执行完update_ops操作后才能开始学习
    training_op = optimizer.minimize(loss)


with tf.name_scope("accuracy"):
    with tf.name_scope('correct_prediction'):
        correct = tf.nn.in_top_k(logits, y, 1)
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
tf.summary.scalar('accuracy',accuracy)

# summaries合并
merged = tf.summary.merge_all()
# 写到指定的磁盘路径中
train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
test_writer = tf.summary.FileWriter(log_dir + '/test')

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

def read_data(data_dir):
    datas = []
    labels = []
    fpaths = []
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        fpaths.append(fpath)
        image = mpimg.imread(fpath)[:, :, :channels]
#        print(image.shape)
        image = transform.resize(image, (height, width))
        label = int(fname.split("_")[0])
        datas.append(image)
        labels.append(label)
#        print('reading the images:%s' % fpath)

    datas = np.array(datas)
    labels = np.array(labels)

#    print("shape of datas: {}\tshape of labels: {}".format(datas.shape, labels.shape))
    return fpaths, datas, labels

fpaths, X_train, y_train  = read_data(data_dir)
fpaths, X_test, y_test = read_data(test_dir)



X_train = X_train.astype(np.float32).reshape(-1, height*width, 3) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, height*width, 3) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:350], X_train[350:]
y_valid, y_train = y_train[:350], y_train[350:]


def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
#    print(len(X),rnd_idx)
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch

def get_model_params():
    gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    return {gvar.op.name: value for gvar, value in zip(gvars, tf.get_default_session().run(gvars))}

def restore_model_params(model_params):
    gvar_names = list(model_params.keys())
    assign_ops = {gvar_name: tf.get_default_graph().get_operation_by_name(gvar_name + "/Assign")
                  for gvar_name in gvar_names}
    init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
    feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
    tf.get_default_session().run(assign_ops, feed_dict=feed_dict)

n_epochs = 5000
batch_size = 200
iteration = 0

best_loss_val = np.infty
check_interval = 100
checks_since_last_progress = 0
max_checks_without_progress = 100
best_model_params = None 


#with tf.Session() as sess:
if train:
    print("训练模式")
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            iteration += 1
            summary, _ = sess.run([merged,training_op], feed_dict={X: X_batch, y: y_batch, training: True})
#               sess.run(clip_all_weights)
            if iteration % check_interval == 0:
                loss_val = loss.eval(feed_dict={X: X_valid, y: y_valid})
                train_writer.add_summary(summary, iteration)
                if loss_val < best_loss_val:
                    best_loss_val = loss_val
                    checks_since_last_progress = 0
                    best_model_params = get_model_params()
                else:
                    checks_since_last_progress += 1
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
#            acc_batch = accuracy.eval(feed_dict={X: X_train, y: y_train})
        acc_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print("Epoch {}, last batch accuracy: {:.4f}%, valid. accuracy: {:.4f}%, valid. best loss: {:.6f}".format(
                epoch, acc_batch * 100, acc_val * 100, best_loss_val))
        if checks_since_last_progress > max_checks_without_progress:
            print("Early stopping!")
            break

    if best_model_params:
        restore_model_params(best_model_params)

#        acc_train = accuracy.eval(feed_dict={X: X_train, y: y_train})
#        acc_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
#        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
#        print("Final train accuracy: {:.4f}%, valid. accuracy: {:.4f}%, Final accuracy on test set: {:.4f}%".format(acc_batch * 100, acc_val * 100, acc_test * 100))
    save_path = saver.save(sess, model_path)
else:
    print("测试模式")
    saver.restore(sess, model_path)
    print("从{}载入模型".format(model_path))

        # label和名称的对照关系
    label_name_dict = {
            0: "0",
            1: "1",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
            10:"10",
            11:"11",
            12:"12",
            13:"13",
            14:"14",
            15:"15",
            16:"16",
            17:"17",
            18:"18",
            19:"19"
    }
    predicted_labels_val = sess.run(Y_proba, feed_dict={X: X_test, y: y_test})
    acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
    print('Testdata accuracy is {:.4f}% '.format(acc_test*100))
    fo = open("resu_wrong.txt", "w")
    fo2 = open("resu_right.txt", "w")
    if not os.path.exists('./wrong_pre_pic'):
        os.mkdir('./wrong_pre_pic')
    if not os.path.exists('./feature_map'):
        os.mkdir('./feature_map')
    fo.write("Picture_Path\tReal_Label\tPredicted_Label\tMax1_label\tMax1_Probability\tMax2_label\tMax2_Probability\tMax3_label\tMax3_Probability\n")
    fo2.write("Picture_Path\tReal_Label\tPredicted_Label\tMax1_label\tMax1_Probability\tMax2_label\tMax2_Probability\tMax3_label\tMax3_Probability\n")
    for index, fpath, real_label, predicted_label in zip(range(len(X_test)), fpaths, y_test, predicted_labels_val):
        var = check_mohu.lapla(fpath)
        if 700 <var < 800:
            print("{}\tdegree of clearity:\t{}\tRecommand retake the picture!".format(fpath,var))
        else:
            real_label_name = label_name_dict[real_label]
            p = np.array(predicted_label)
            max1_ind = p.argsort()[::-1][0]
            max2_ind = p.argsort()[::-1][1]
            max3_ind = p.argsort()[::-1][2]
            max1 = label_name_dict[max1_ind]
            max2 = label_name_dict[max2_ind]
            max3 = label_name_dict[max3_ind]
#        print("%s Probability:[%s %0.2f%%]    [%s %0.2f%%]    [%s %0.2f%%]" % (fpath, max1,predicted_label[max1_ind]*100,max2,predicted_label[max2_ind]*100,max3,predicted_label[max3_ind]*100)) 
            predicted_label_name = label_name_dict[np.argmax(predicted_label)]

            if real_label_name != predicted_label_name:
                fname = real_label_name+'_to_'+predicted_label_name+'_'+str(random.randint(1,10000))+'.png'
                npath = './wrong_pre_pic/'+fname
                shutil.copyfile(fpath,npath)
                print("{}\t{} => {}".format(fpath, real_label_name, predicted_label_name))
                fo.write("%s\t%s\t%s\t%s\t%0.2f%%\t%s\t%0.2f%%\t%s\t%0.2f%%\n" % (npath,real_label_name,predicted_label_name,max1,predicted_label[max1_ind]*100,max2,predicted_label[max2_ind]*100,max3,predicted_label[max3_ind]*100))
            else:
                fo2.write("%s\t%s\t%s\t%s\t%0.2f%%\t%s\t%0.2f%%\t%s\t%0.2f%%\n" % (fpath,real_label_name,predicted_label_name,max1,predicted_label[max1_ind]*100,max2,predicted_label[max2_ind]*100,max3,predicted_label[max3_ind]*100))

'''
            input_image = X_test[index:index+1]
            conv1_32 = sess.run(conv1_bn_act, feed_dict={X:input_image})     # [1, 28, 28 ,16]
            conv1_transpose = sess.run(tf.transpose(conv1_32, [3, 0, 1, 2]))
            fig3,ax3 = plt.subplots(nrows=1, ncols=32, figsize = (32,1))
            for i in range(32):
                ax3[i].imshow(conv1_transpose[i][0])                      # tensor的切片[row, column]
            plt.title('Conv1 32x75x50')
            plt.savefig('./feature_map/'+fname+'_Conv1.png')

            pool1_32 = sess.run(pool1, feed_dict={X:input_image})     # [1, 14, 14, 16]
            pool1_transpose = sess.run(tf.transpose(pool1_32, [3, 0, 1, 2]))
            fig4,ax4 = plt.subplots(nrows=1, ncols=32, figsize=(32,1))
            for i in range(32):
                ax4[i].imshow(pool1_transpose[i][0])
            plt.title('Pool1 32x75x50')
            plt.savefig('./feature_map/'+fname+'_Pool1.png')

            conv2_64 = sess.run(conv2_bn_act, feed_dict={X:input_image})     # [1, 28, 28 ,16]
            conv2_transpose = sess.run(tf.transpose(conv2_64, [3, 0, 1, 2]))
            fig5,ax5 = plt.subplots(nrows=1, ncols=64, figsize = (64,1))
            for i in range(64):
                ax5[i].imshow(conv2_transpose[i][0])                      # tensor的切片[row, column]
            plt.title('Conv2 64')
            plt.savefig('./feature_map/'+fname+'_Conv2.png')

            pool2_64 = sess.run(pool2, feed_dict={X:input_image})     # [1, 14, 14, 16]
            pool2_transpose = sess.run(tf.transpose(pool2_64, [3, 0, 1, 2]))
            fig6,ax6 = plt.subplots(nrows=1, ncols=64, figsize=(64,1))
            for i in range(64):
                ax6[i].imshow(pool2_transpose[i][0])
            plt.title('Pool2 64')
            plt.savefig('./feature_map/'+fname+'_Pool2.png')

    acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
    print("Test accuracy: {:.4f}%".format(acc_test * 100))
#        for X_batch, y_batch in shuffle_batch(X_test, y_test, 1):
#            st = time.time()
#            predicted_labels_val=sess.run(Y_proba, feed_dict={X: X_batch, y: y_batch})
#            print('Elapsed time: {}'.format(time.time()-st))
#            predicted_label_name = label_name_dict[np.argmax(predicted_labels_val)]
#            print("{} => {}".format(y_batch, predicted_label_name))
'''
train_writer.close()
test_writer.close()









