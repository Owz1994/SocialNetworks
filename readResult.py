import tensorflow as tf
reader = tf.train.NewCheckpointReader("save/tensor_net.ckpt")
W1 = reader.get_tensor("weight0")
#answer = reader.get_tensor("answer")
#print(answer)
with open("save/result.txt","w") as result:
    for line in W1:
        for x in line:
            result.write(str(x)+" ")
        result.write("\n")