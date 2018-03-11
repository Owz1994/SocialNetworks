import tensorflow as tf
import numpy as np
units =[34,1024,256,34]  ##此处为输入输出及隐藏各层维度，第一为输出，最后为输出
batch_size = 100  ##单批次训练量
tot_num = 4000   ##train test 样例总数
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
with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('save/tensor_net.ckpt.meta')
    new_saver.restore(sess, "save/tensor_net.ckpt")
    graph = tf.get_default_graph()
    x = graph.get_operation_by_name('x').outputs[0]
    y = tf.get_collection("answer")[0]
    keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]
    x_train_batch, x_test_batch, y_train_batch, y_test_batch = get_batch_data(batch_size)
    result = sess.run(y, feed_dict={x: x_test_batch, keep_prob:1.0})
    #np.savetxt("y.txt", )
    np.savetxt("save/result.txt", y_test_batch-result,fmt='%d')
    # with open("save/result.txt","w") as result:
    #     sess.run(result)