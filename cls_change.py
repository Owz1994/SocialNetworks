# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random
# params
#in_units = 30     ##输入维度
#h1_units = 10     ##隐藏神经元
#h2_units = 5
#out_units = 30    ##输出维度

units =[34,2400,34]  ##此处为输入输出及隐藏各层维度，第一为输出，最后为输出

batch_size = 100  ##单批次训练量
iteration = 200  ##迭代次数
tot_num = 4000   ##train test 样例总数
learn_rate = 0.01 ##学习率
def get_data():
    train_feature = []
    test_feature = []
    a = 0
    b = 0
    with open("trainInput_10000_030_2-4.txt", "r") as inputFile:
        for line in inputFile:  
            a+=1
            data = line.split(" ")
            #data = line.encode('utf-8').decode('utf-8-sig').split(" ")
            #print(data)
            data=[int(x) for x in data[:units[0]]]
            #for x in data[:30]:
                #data2 = data2.append(int(x))
            #print(data)
            if a<=tot_num:
                train_feature.append(data[:units[0]])
            elif a<=tot_num+tot_num:
                test_feature.append(data[:units[0]])
    train_label = []
    test_label = []
    with open("trainOotput_10000_030_2-4.txt", "r") as outputFile:
        for line in outputFile: 
            b+=1
            data=[]
            data = line.split(" ")
            data=[int(x) for x in data[:units[0]]]
            if b<=tot_num:
                train_label.append(data[:units[0]])
            elif b<=tot_num+tot_num:
                test_label.append(data[:units[0]])
    return train_feature,test_feature, train_label,test_label

def get_batch_data(batch_size):
    train_feature,test_feature, train_label,test_label = get_data()
    rand_idx = np.random.permutation(tot_num)
    x_train_batch,x_test_batch,y_train_batch,y_test_batch = [], [], [], []
    for i in range(batch_size):
        x_train_batch.append(train_feature[rand_idx[i]])
        y_train_batch.append(train_label[rand_idx[i]])
    for i in range(batch_size):
        x_test_batch.append(test_feature[rand_idx[i]])
        y_test_batch.append(test_label[rand_idx[i]])
    return x_train_batch,x_test_batch, y_train_batch,y_test_batch
###输出便捷加入神经元
W=[]
b=[]
for i in range(len(units)-1):
    W.append(tf.Variable(tf.truncated_normal([units[i], units[i+1]], stddev=0.1),name="weight"+str(i)))
    b.append(tf.Variable(tf.zeros([units[i+1]]),name="bias"+str(i)))
x = tf.placeholder(tf.float32, [None, units[0]],name="x")
y_ = tf.placeholder(tf.float32, [None, units[-1]],name="y_")
keep_prob = tf.placeholder(tf.float32,name="keep_prob")
hidden=[]
hidden.append(x)
for i in range(len(units)-2):
    hidden.append(tf.nn.relu(tf.matmul(hidden[-1], W[i]) + b[i]))
hidden_drop = tf.nn.dropout(hidden[-1], keep_prob)
y = tf.matmul(hidden_drop, W[-1]) + b[-1]
###简单两层隐含层
#W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
#b1 = tf.Variable(tf.zeros([h1_units]))
#W2 = tf.Variable(tf.truncated_normal([h1_units, h2_units], stddev=0.1))
#b2 = tf.Variable(tf.zeros([h2_units]))
#W3 = tf.Variable(tf.truncated_normal([h2_units, out_units], stddev=0.1))
#b3 = tf.Variable(tf.zeros([out_units]))
#x = tf.placeholder(tf.float32, [None, in_units])
#y_ = tf.placeholder(tf.float32, [None, out_units])
#keep_prob = tf.placeholder(tf.float32)
#hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
#hidden2 = tf.nn.relu(tf.matmul(hidden1, W2) + b2)
#hidden2_drop = tf.nn.dropout(hidden2, keep_prob)
#y = tf.matmul(hidden2_drop, W3) + b3

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y,labels=y_)
loss = tf.reduce_mean(cross_entropy)
training_op = tf.train.AdamOptimizer(learn_rate).minimize(loss)
logits = tf.cast(tf.greater(tf.nn.sigmoid(y), tf.fill([tf.shape(x)[0], units[-1]], 0.5)), tf.float32,name="answer")
correct = tf.equal(tf.reduce_sum(logits-y_, axis=1), tf.zeros([tf.shape(y)[0]],tf.float32))
#correct = tf.equal(tf.reduce_sum(logits-y_, axis=1), tf.ones([tf.shape(y)[0]],tf.float32))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
saver = tf.train.Saver(write_version=tf.train.SaverDef.V2) # 声明tf.train.Saver类用于保存模型
init = tf.global_variables_initializer()
tf.add_to_collection('answer', logits)

with tf.Session() as sess:
    init.run()
    for iter in range(iteration):
        x_train_batch,x_test_batch, y_train_batch,y_test_batch = get_batch_data(batch_size)
        sess.run(training_op, feed_dict={x: x_train_batch, y_:y_train_batch, keep_prob:0.5})
        train = sess.run([accuracy,loss], feed_dict={x: x_train_batch, y_:y_train_batch, keep_prob:0.5})
        test = sess.run([accuracy,loss], feed_dict={x: x_test_batch, y_:y_test_batch, keep_prob:1.0})
        saver_path = saver.save(sess, "save/tensor_net.ckpt")  # 将模型保存到save/model.ckpt文件
        print("iteration ", iter, ",Train_accuracy_loss=", train, ",Test_accuracy_loss=", test)
        # if iter == iteration-1:
        #     with open("save/weight.txt" , "r") as weight:
        #         weight.write(W)
