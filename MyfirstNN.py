import tensorflow as tf
import numpy as numpy
import os

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST Data', one_hot=True)
sess=tf.InteractiveSession()

#shape=[None,784]<- None can be anything no restrictions, 784=28*28 [No of pixels per image]
#for y 10 <- No of digits 
#x is I/P, y is O/P class


x=tf.placeholder(tf.float32,shape = [None,784])
y_=tf.placeholder(tf.float32,shape = [None,10])

#W<-weights, b<-bias
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
sess.run(tf.global_variables_initializer())

#Applying regression model
y = tf.matmul(x,W) + b
#Loss function is ...
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

#Training the model
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#Train it a 1000 times for good measure
for i in range(1000):
	#Loading a 100 training examples per iteration
	batch = mnist.train.next_batch(100)
	train_step.run(feed_dict={x:batch[0],y_:batch[1]})

#Evaluating the model
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#Casting the boolean to float so that a mean can be taken
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#Print maadi
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
