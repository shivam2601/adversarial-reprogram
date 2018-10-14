import numpy as np

import tensorflow as tf
import network as resnet_v2

from network import resnet_v2_50
from PIL import Image
import tensorflow.contrib.slim as slim
import time
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image


size = 299
batch_size = 50
starter_learning_rate = 0.01
decay_rate = 0.96
num_batches = int(55000/ batch_size)
decay_steps = 2* num_batches
reg_lambda = 0.005
display_step = 100
total_steps = 100000

output_mapping = np.zeros([1001, 10])
output_mapping[0:10, 0:10] = np.eye(10)

input_mask = np.pad(np.zeros([1, 28, 28, 3]),[[0,0], [136, 135], [136, 135], [0,0]],'constant', constant_values = 1)

def preprocess(img) :
	return (img - 0.5)*2.0

def deprocess(img) :
	return (img +1)/2.0	

with tf.Graph().as_default():

	with slim.arg_scope(resnet_v2.resnet_arg_scope()):

		global_step = tf.Variable(0, trainable=False)
		# mask = tf.pad(tf.constant(np.zeros([1, 28, 28, 3]), dtype = tf.float32), 
		# 		paddings = tf.constant([[0,0], [136, 135], [136, 135], [0,0]]), constant_values=1)
		mask = tf.constant(input_mask, dtype = tf.float32)
		
		weights = tf.get_variable('adv_weight', shape = [1, size, size, 3], dtype = tf.float32)
		input_image = tf.placeholder(shape = [None, 28,28,1], dtype = tf.float32)
		channel_image = tf.concat([input_image, input_image, input_image], axis = -1)
		rgb_image = tf.pad(tf.concat([input_image, input_image, input_image], axis = -1), 
					paddings = tf.constant([[0,0], [136, 135], [136, 135], [0,0]]))

		adv_image = tf.nn.tanh(tf.multiply(weights, mask)) + rgb_image

		labels = tf.placeholder(tf.float32, shape=[None, 10])
		
		logits,_ = resnet_v2_50(adv_image,num_classes = 1001,is_training=False)
		
		output_mapping_tensor = tf.constant(output_mapping, dtype = tf.float32)
		new_logits = tf.matmul(logits, output_mapping_tensor)

		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = new_logits, labels = labels)) + reg_lambda * tf.nn.l2_loss(weights)

		learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
											   decay_steps, decay_rate, staircase=True)

		correct_prediction = tf.equal(tf.argmax(new_logits,1), tf.argmax(labels,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step = global_step, var_list = [weights])
		variables_to_restore = slim.get_model_variables('resnet_v2_50')
		
		
		
		
		sess = tf.InteractiveSession()
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver(variables_to_restore)
		saver1 = tf.train.Saver([weights])

		saver.restore(sess,'./resnet_v2_50.ckpt')
		
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  
for i in range(total_steps) :
 	batch = mnist.train.next_batch(batch_size)
 	_, l, acc = sess.run([train_step, loss, accuracy], feed_dict = {input_image : preprocess(np.reshape(batch[0], [-1, 28, 28, 1])),
 									labels : batch[1]})
 	if i%display_step == 0 :
 		print('after %d steps the loss is %g and train_acc is %g'%(i, l, acc))

saver1.save(sess, 'adv_weights')

test_iterations = int(len(mnist.test.images)/ batch_size)
total_correct = 0
for i in range(test_iterations) :
 	test_imgs = preprocess(np.reshape(mnist.test.images[i*batch_size:(i+1)*batch_size], [-1,28,28,1]))
 	test_labels = mnist.test.labels[i*batch_size:(i+1)*batch_size]
 	corrects = sess.run(accuracy, feed_dict = {input_image : test_imgs, labels : test_labels})
 	total_correct += corrects * batch_size

print('test acc is ', float(total_correct)/len(mnist.test.images))



		
				
				        
	   

