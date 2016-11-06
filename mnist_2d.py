#!/usr/bin/python
#-*-coding:utf-8-*- 
import input_bear_data
import numpy
images = numpy.load("images.npy")
labels = numpy.load("labels.npy")
test_images = numpy.load('test_images.npy')
test_labels = numpy.load('test_labels.npy')
bear = input_bear_data.read_data_sets(images, labels, test_images, test_labels)
acc = numpy.ones([25,25])
import tensorflow as tf
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32,[None,2400])#input 
y_ = tf.placeholder(tf.float32,[None,10])
#valid = tf.placeholder(tf.float32,[None])
rate = tf.placeholder(tf.float32)
#initial weights and bias
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
#def accuracy(shape):
 #   initial = tf.constant(1,shape=shape)
  #  return tf.Variable(initial)
#difine the convolutional operation and pooling operation
#the padding choose 'SAME' means the output is the same size as the input
def conv2d(x, W):
    return tf.nn.conv2d(x , W, strides = [1,1,1,1], padding= 'SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1, 2, 2, 1], padding='SAME')
#the first conv layer
#valid = accuracy([2])
W_conv1 = weight_variable([5, 5, 1, 25])
b_conv1 = bias_variable([25])
#reshape the input image into a 4d tensor
x_image = tf.reshape(x, [-1, 60, 40, 1])
#convolve x_image with the weight tensor
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
#the second convolutional layer
W_conv2 = weight_variable([5, 5, 25, 50])
b_conv2 = bias_variable([50])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
# fully_connected layer
W_fc1 = weight_variable([ 15 * 10 * 50, 500])
b_fc1 = bias_variable([500])
h_pool2_flat = tf.reshape(h_pool2, [-1, 15 * 10 * 50])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1)
#Dropout reduce overfitting
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#readout layer
W_fc2 = weight_variable([500, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
#train and evaluate the model
#cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(tf.matmul(h_fc1, W_fc2) + b_fc2, y_)
train_step = tf.train.AdamOptimizer(rate).minimize(cross_entropy)
#train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)
#the output of correct_prediction is boolean value
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
#cast turns the boolean value into flost
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
for j in range(25):
    sess.run(tf.initialize_all_variables())
    for i in range(3000):
        batch = bear.train.next_batch(50)
        #show the accuracy every 100th iteration 
        if i%50 == 0:
            train_accuracy = accuracy.eval(feed_dict = {x: batch[0], y_: batch[1], keep_prob: 1.0})
            print "step %d, training accuracy %g" %(i,train_accuracy)
        if i <=600:
            train_step.run(feed_dict = {x: batch[0], y_: batch[1],rate: 0.001,keep_prob: 0.5})
        elif (i>600 and i<=2000):
             train_step.run(feed_dict = {x: batch[0], y_: batch[1],rate: 0.0008,keep_prob: 0.5})
        else:
            train_step.run(feed_dict = {x: batch[0], y_: batch[1],rate: 0.0002,keep_prob: 0.5})
    acc[0,j] = accuracy.eval(feed_dict = {x: bear.validation.images, y_: bear.validation.labels, keep_prob: 1.0})
    print ("validation accuracy %g" %acc[0,j])
    acc[1,j] = accuracy.eval(feed_dict = {x: bear.test.images, y_: bear.test.labels, keep_prob: 1.0})
    print ("test accuracy %g" %acc[1,j])
   # print ("test accuracy1 %g" %accuracy.eval(feed_dict = {x: bear.test.images[0:1000,:], y_: bear.test.labels[0:1000,:], keep_prob: 1.0}))
   # print ("test accuracy1 %g" %accuracy.eval(feed_dict = {x: bear.test.images[1000:2000,:], y_: bear.test.labels[1000:2000,:], keep_prob: 1.0}))
   # print ("test accuracy1 %g" %accuracy.eval(feed_dict = {x: bear.test.images[2000:2501,:], y_: bear.test.labels[2000:2501,:], keep_prob: 1.0}))
#valid = sess.run(valid)
numpy.save("acc.npy",acc)
