import tensorflow as tf
import numpy as np
import random, math
from numpy import genfromtxt
from numpy import fft

############################### Hyperparameters #########################################
dataLength=16704
train_batch_size= 80
test_batch_size= 25
dataDir= './...'  # the directory of data
saveDir= './...'  # the directory for parameter saving and loss saving

global_step= tf.Variable(0, trainable= False)
learningRate=tf.train.exponential_decay(learning_rate= 0.001, global_step=global_step, decay_steps= 1800, decay_rate= 0.1, staircase = True)

############################### Training data import ############################

x_all= np.zeros((1000, 1, dataLength, 1))
y_all= np.zeros((1000, 1, dataLength, 1))

x_train= np.zeros((900, 1, dataLength, 1))
y_train= np.zeros((900, 1, dataLength, 1))

x_test= np.zeros((100, 1, dataLength, 1))
y_test= np.zeros((100, 1, dataLength, 1))

for i in range(1000):          
	filename= dataDir+'/%d.txt' % (i+1)
	# print(filename)
	data = genfromtxt(filename, delimiter=',')
	x_all[i, 0, :, 0]=data[0:dataLength, 0]
	y_all[i, 0, :, 0]=data[0:dataLength, 1]

trainset_no= random.sample(range(1000), 900)
testset_no= list(set(range(1000))-set(trainset_no))

x_train= x_all[trainset_no]
y_train= y_all[trainset_no]

x_test= x_all[testset_no]
y_test= y_all[testset_no]

############################### Model architecture ############################
with tf.name_scope('Inputs'):
	x_t1= tf.placeholder(dtype=tf.float32, shape=[train_batch_size, 1, dataLength, 1])
	y_t1= tf.placeholder(dtype=tf.float32, shape=[train_batch_size, 1, dataLength, 1])
	x_ph_test= tf.placeholder(dtype=tf.float32, shape=[test_batch_size, 1, dataLength, 1])
	y_ph_test= tf.placeholder(dtype=tf.float32, shape=[test_batch_size, 1, dataLength, 1])

with tf.variable_scope('trainable_variables', reuse= tf.AUTO_REUSE):
	W_fw_11= tf.get_variable(name= 'weight_fw_11', shape= [1, 3, 1, 32], initializer= tf.truncated_normal_initializer(mean= 0, stddev= 0.1))
	b_fw_11= tf.get_variable(name= 'bias_fw_11', shape= [32], initializer= tf.zeros_initializer())
	W_fw_12= tf.get_variable(name= 'weight_fw_12', shape= [1, 3, 32, 32], initializer= tf.truncated_normal_initializer(mean= 0, stddev= 0.1))
	b_fw_12= tf.get_variable(name= 'bias_fw_12', shape= [32], initializer= tf.zeros_initializer())
	W_downsam1= tf.get_variable(name= 'weight_downsam1', shape= [1, 1, 1, 32], initializer= tf.truncated_normal_initializer(mean= 0, stddev= 0.1))
	b_downsam1= tf.get_variable(name= 'bias_downsam1', shape= [32], initializer= tf.zeros_initializer())
	W_fw_21= tf.get_variable(name= 'weight_fw_21', shape= [1, 3, 64, 64], initializer= tf.truncated_normal_initializer(mean= 0, stddev= 0.1))
	b_fw_21= tf.get_variable(name= 'bias_fw_21', shape= [64], initializer= tf.zeros_initializer())
	W_fw_22= tf.get_variable(name= 'weight_fw_22', shape= [1, 3, 64, 64], initializer= tf.truncated_normal_initializer(mean= 0, stddev= 0.1))
	b_fw_22= tf.get_variable(name= 'bias_fw_22', shape= [64], initializer= tf.zeros_initializer())
	W_downsam2= tf.get_variable(name= 'weight_downsam2', shape= [1, 1, 1, 64], initializer= tf.truncated_normal_initializer(mean= 0, stddev= 0.1))
	b_downsam2= tf.get_variable(name= 'bias_downsam2', shape= [64], initializer= tf.zeros_initializer())
	W_fw_31= tf.get_variable(name= 'weight_fw_31', shape= [1, 3, 128, 128], initializer= tf.truncated_normal_initializer(mean= 0, stddev= 0.1))
	b_fw_31= tf.get_variable(name= 'bias_fw_31', shape= [128], initializer= tf.zeros_initializer())
	W_fw_32= tf.get_variable(name= 'weight_fw_32', shape= [1, 3, 128, 128], initializer= tf.truncated_normal_initializer(mean= 0, stddev= 0.1))
	b_fw_32= tf.get_variable(name= 'bias_fw_32', shape= [128], initializer= tf.zeros_initializer())
	W_downsam3= tf.get_variable(name= 'weight_downsam3', shape= [1, 1, 1, 128], initializer= tf.truncated_normal_initializer(mean= 0, stddev= 0.1))
	b_downsam3= tf.get_variable(name= 'bias_downsam3', shape= [128], initializer= tf.zeros_initializer())
	W_fw_41= tf.get_variable(name= 'weight_fw_41', shape= [1, 3, 256, 128], initializer= tf.truncated_normal_initializer(mean= 0, stddev= 0.1))
	b_fw_41= tf.get_variable(name= 'bias_fw_41', shape= [128], initializer= tf.zeros_initializer())
	W_fw_42= tf.get_variable(name= 'weight_fw_42', shape= [1, 3, 128, 128], initializer= tf.truncated_normal_initializer(mean= 0, stddev= 0.1))
	b_fw_42= tf.get_variable(name= 'bias_fw_42', shape= [128], initializer= tf.zeros_initializer())
	W_bw_4to3= tf.get_variable(name= 'weight_bw_4to3', shape= [1, 5, 128, 128], initializer= tf.truncated_normal_initializer(mean= 0, stddev= 0.1))
	b_bw_4to3= tf.get_variable(name= 'bias_bw_4to3', shape= [128], initializer= tf.zeros_initializer())
	W_bw_31= tf.get_variable(name= 'weight_bw_31', shape= [1, 3, 256, 128], initializer= tf.truncated_normal_initializer(mean= 0, stddev= 0.1))
	b_bw_31= tf.get_variable(name= 'bias_bw_31', shape= [128], initializer= tf.zeros_initializer())
	W_bw_32= tf.get_variable(name= 'weight_bw_32', shape= [1, 3, 128, 128], initializer= tf.truncated_normal_initializer(mean= 0, stddev= 0.1))
	b_bw_32= tf.get_variable(name= 'bias_bw_32', shape= [128], initializer= tf.zeros_initializer())
	W_bw_3to2= tf.get_variable(name= 'weight_bw_3to2', shape= [1, 5, 64, 128], initializer= tf.truncated_normal_initializer(mean= 0, stddev= 0.1))
	b_bw_3to2= tf.get_variable(name= 'bias_bw_3to2', shape= [64], initializer= tf.zeros_initializer())
	W_bw_21= tf.get_variable(name= 'weight_bw_21', shape= [1, 3, 128, 64], initializer= tf.truncated_normal_initializer(mean= 0, stddev= 0.1))
	b_bw_21= tf.get_variable(name= 'bias_bw_21', shape= [64], initializer= tf.zeros_initializer())
	W_bw_22= tf.get_variable(name= 'weight_bw_22', shape= [1, 3, 64, 64], initializer= tf.truncated_normal_initializer(mean= 0, stddev= 0.1))
	b_bw_22= tf.get_variable(name= 'bias_bw_22', shape= [64], initializer= tf.zeros_initializer())
	W_bw_2to1= tf.get_variable(name= 'weight_bw_2to1', shape= [1, 5, 32, 64], initializer= tf.truncated_normal_initializer(mean= 0, stddev= 0.1))
	b_bw_2to1= tf.get_variable(name= 'bias_bw_2to1', shape= [32], initializer= tf.zeros_initializer())
	W_bw_11= tf.get_variable(name= 'weight_bw_11', shape= [1, 3, 64, 32], initializer= tf.truncated_normal_initializer(mean= 0, stddev= 0.1))
	b_bw_11= tf.get_variable(name= 'bias_bw_11', shape= [32], initializer= tf.zeros_initializer())
	W_bw_12= tf.get_variable(name= 'weight_bw_12', shape= [1, 3, 32, 32], initializer= tf.truncated_normal_initializer(mean= 0, stddev= 0.1))
	b_bw_12= tf.get_variable(name= 'bias_bw_12', shape= [32], initializer= tf.zeros_initializer())
	W_bw_out= tf.get_variable(name= 'weight_bw_out', shape= [1, 1, 32, 1], initializer= tf.truncated_normal_initializer(mean= 0, stddev= 0.1))
	b_bw_out= tf.get_variable(name= 'bias_bw_out', shape= [1], initializer= tf.zeros_initializer())
	trainable_vars= tf.trainable_variables()

with tf.name_scope('Tower1'):
	h11= tf.nn.relu(tf.nn.conv2d(x_t1, W_fw_11, strides= [1, 1, 1, 1], padding= 'SAME')+ b_fw_11)
	h12= tf.nn.tanh(tf.nn.conv2d(h11, W_fw_12, strides= [1, 1, 1, 1], padding= 'SAME')+ b_fw_12)
	h21_partial_1= tf.nn.conv2d(x_t1, W_downsam1, strides= [1, 1, 4, 1], padding= 'SAME')+b_downsam1
	h21_partial_2= tf.nn.max_pool(h12, ksize= [1 ,1, 4, 1], strides= [1, 1 ,4 ,1], padding= 'SAME')
	h21= tf.concat([h21_partial_1, h21_partial_2], 3)
	h22= tf.nn.relu(tf.nn.conv2d(h21, W_fw_21, strides= [1, 1, 1, 1], padding= 'SAME')+ b_fw_21)
	h23= tf.nn.tanh(tf.nn.conv2d(h22, W_fw_22, strides= [1, 1, 1, 1], padding= 'SAME')+ b_fw_22)
	h31_partial_1= tf.nn.conv2d(x_t1, W_downsam2, strides= [1, 1, 16, 1], padding= 'SAME')+b_downsam2
	h31_partial_2= tf.nn.max_pool(h23, ksize= [1 ,1, 4, 1], strides= [1, 1 ,4 ,1], padding= 'SAME')
	h31= tf.concat([h31_partial_1, h31_partial_2], 3)
	h32= tf.nn.relu(tf.nn.conv2d(h31, W_fw_31, strides= [1, 1, 1, 1], padding= 'SAME')+ b_fw_31)
	h33= tf.nn.tanh(tf.nn.conv2d(h32, W_fw_32, strides= [1, 1, 1, 1], padding= 'SAME')+ b_fw_32)
	h41_partial_1= tf.nn.conv2d(x_t1, W_downsam3, strides= [1, 1, 64, 1], padding= 'SAME')+b_downsam3
	h41_partial_2= tf.nn.max_pool(h33, ksize= [1 ,1, 4, 1], strides= [1, 1 ,4 ,1], padding= 'SAME')
	h41= tf.concat([h41_partial_1, h41_partial_2], 3)
	h42= tf.nn.relu(tf.nn.conv2d(h41, W_fw_41, strides= [1, 1, 1, 1], padding= 'SAME')+ b_fw_41)
	h43= tf.nn.tanh(tf.nn.conv2d(h42, W_fw_42, strides= [1, 1, 1, 1], padding= 'SAME')+ b_fw_42)
	h_bw_31_partial= tf.nn.tanh(tf.nn.conv2d_transpose(h43, W_bw_4to3, output_shape= h33.shape, strides= [1, 1, 4, 1], padding= 'SAME')+ b_bw_4to3)
	h_bw_31= tf.concat([h_bw_31_partial, h33], 3)
	h_bw_32= tf.nn.relu(tf.nn.conv2d(h_bw_31, W_bw_31, strides= [1, 1, 1, 1], padding= 'SAME')+ b_bw_31)
	h_bw_33= tf.nn.relu(tf.nn.conv2d(h_bw_32, W_bw_32, strides= [1, 1, 1, 1], padding= 'SAME')+ b_bw_32)
	h_bw_21_partial= tf.nn.tanh(tf.nn.conv2d_transpose(h_bw_33, W_bw_3to2, output_shape= h23.shape, strides= [1, 1, 4, 1], padding= 'SAME')+ b_bw_3to2)
	h_bw_21= tf.concat([h_bw_21_partial, h23], 3)
	h_bw_22= tf.nn.relu(tf.nn.conv2d(h_bw_21, W_bw_21, strides= [1, 1, 1, 1], padding= 'SAME')+ b_bw_21)
	h_bw_23= tf.nn.relu(tf.nn.conv2d(h_bw_22, W_bw_22, strides= [1, 1, 1, 1], padding= 'SAME')+ b_bw_22)
	h_bw_11_partial= tf.nn.tanh(tf.nn.conv2d_transpose(h_bw_23, W_bw_2to1, output_shape= h12.shape, strides= [1, 1, 4, 1], padding= 'SAME')+ b_bw_2to1)
	h_bw_11= tf.concat([h_bw_11_partial, h12], 3)
	h_bw_12= tf.nn.relu(tf.nn.conv2d(h_bw_11, W_bw_11, strides= [1, 1, 1, 1], padding= 'SAME')+ b_bw_11)
	h_bw_13= tf.nn.tanh(tf.nn.conv2d(h_bw_12, W_bw_12, strides= [1, 1, 1, 1], padding= 'SAME')+ b_bw_12)
	output= tf.nn.conv2d(h_bw_13, W_bw_out, strides= [1, 1, 1, 1], padding= 'SAME')+ b_bw_out	
	Training_Loss= tf.reduce_mean(tf.abs(output-y_t1))

with tf.name_scope('Test'):
	h11_test= tf.nn.relu(tf.nn.conv2d(x_ph_test, W_fw_11, strides= [1, 1, 1, 1], padding= 'SAME')+ b_fw_11)
	h12_test= tf.nn.tanh(tf.nn.conv2d(h11_test, W_fw_12, strides= [1, 1, 1, 1], padding= 'SAME')+ b_fw_12)
	h21_partial_1_test= tf.nn.conv2d(x_ph_test, W_downsam1, strides= [1, 1, 4, 1], padding= 'SAME')+b_downsam1
	h21_partial_2_test= tf.nn.max_pool(h12_test, ksize= [1 ,1, 4, 1], strides= [1, 1 ,4 ,1], padding= 'SAME')
	h21_test= tf.concat([h21_partial_1_test, h21_partial_2_test], 3)
	h22_test= tf.nn.relu(tf.nn.conv2d(h21_test, W_fw_21, strides= [1, 1, 1, 1], padding= 'SAME')+ b_fw_21)
	h23_test= tf.nn.tanh(tf.nn.conv2d(h22_test, W_fw_22, strides= [1, 1, 1, 1], padding= 'SAME')+ b_fw_22)
	h31_partial_1_test= tf.nn.conv2d(x_ph_test, W_downsam2, strides= [1, 1, 16, 1], padding= 'SAME')+b_downsam2
	h31_partial_2_test= tf.nn.max_pool(h23_test, ksize= [1 ,1, 4, 1], strides= [1, 1 ,4 ,1], padding= 'SAME')
	h31_test= tf.concat([h31_partial_1_test, h31_partial_2_test], 3)
	h32_test= tf.nn.relu(tf.nn.conv2d(h31_test, W_fw_31, strides= [1, 1, 1, 1], padding= 'SAME')+ b_fw_31)
	h33_test= tf.nn.tanh(tf.nn.conv2d(h32_test, W_fw_32, strides= [1, 1, 1, 1], padding= 'SAME')+ b_fw_32)
	h41_partial_1_test= tf.nn.conv2d(x_ph_test, W_downsam3, strides= [1, 1, 64, 1], padding= 'SAME')+b_downsam3
	h41_partial_2_test= tf.nn.max_pool(h33_test, ksize= [1 ,1, 4, 1], strides= [1, 1 ,4 ,1], padding= 'SAME')
	h41_test= tf.concat([h41_partial_1_test, h41_partial_2_test], 3)
	h42_test= tf.nn.relu(tf.nn.conv2d(h41_test, W_fw_41, strides= [1, 1, 1, 1], padding= 'SAME')+ b_fw_41)
	h43_test= tf.nn.tanh(tf.nn.conv2d(h42_test, W_fw_42, strides= [1, 1, 1, 1], padding= 'SAME')+ b_fw_42)
	h_bw_31_partial_test= tf.nn.tanh(tf.nn.conv2d_transpose(h43_test, W_bw_4to3, output_shape= h33_test.shape, strides= [1, 1, 4, 1], padding= 'SAME')+ b_bw_4to3)
	h_bw_31_test= tf.concat([h_bw_31_partial_test, h33_test], 3)
	h_bw_32_test= tf.nn.relu(tf.nn.conv2d(h_bw_31_test, W_bw_31, strides= [1, 1, 1, 1], padding= 'SAME')+ b_bw_31)
	h_bw_33_test= tf.nn.relu(tf.nn.conv2d(h_bw_32_test, W_bw_32, strides= [1, 1, 1, 1], padding= 'SAME')+ b_bw_32)
	h_bw_21_partial_test= tf.nn.tanh(tf.nn.conv2d_transpose(h_bw_33_test, W_bw_3to2, output_shape= h23_test.shape, strides= [1, 1, 4, 1], padding= 'SAME')+ b_bw_3to2)
	h_bw_21_test= tf.concat([h_bw_21_partial_test, h23_test], 3)
	h_bw_22_test= tf.nn.relu(tf.nn.conv2d(h_bw_21_test, W_bw_21, strides= [1, 1, 1, 1], padding= 'SAME')+ b_bw_21)
	h_bw_23_test= tf.nn.relu(tf.nn.conv2d(h_bw_22_test, W_bw_22, strides= [1, 1, 1, 1], padding= 'SAME')+ b_bw_22)
	h_bw_11_partial_test= tf.nn.tanh(tf.nn.conv2d_transpose(h_bw_23_test, W_bw_2to1, output_shape= h12_test.shape, strides= [1, 1, 4, 1], padding= 'SAME')+ b_bw_2to1)
	h_bw_11_test= tf.concat([h_bw_11_partial_test, h12_test], 3)
	h_bw_12_test= tf.nn.relu(tf.nn.conv2d(h_bw_11_test, W_bw_11, strides= [1, 1, 1, 1], padding= 'SAME')+ b_bw_11)
	h_bw_13_test= tf.nn.tanh(tf.nn.conv2d(h_bw_12_test, W_bw_12, strides= [1, 1, 1, 1], padding= 'SAME')+ b_bw_12)
	output_test= tf.nn.conv2d(h_bw_13_test, W_bw_out, strides= [1, 1, 1, 1], padding= 'SAME')+ b_bw_out
	Valid_Loss= tf.reduce_mean(tf.abs(output_test- y_ph_test))


tf.summary.scalar('Training_Loss', Training_Loss)
tf.summary.scalar('Valid_Loss', Valid_Loss)
train_step= tf.train.AdamOptimizer(learningRate).minimize(Training_Loss)

sess= tf.Session()
saver=tf.train.Saver()
writer=tf.summary.FileWriter('./summary', sess.graph)
merged=tf.summary.merge_all()


sess.run(tf.global_variables_initializer())

###################### training process #####################

with open(saveDir+ '/Loss.txt', 'w') as f_loss:
	f_loss.write('')

with open(saveDir+ '/Loss.txt', 'a') as f_loss:
	total_epoch= 2000
	for i in range(total_epoch):
		mini_1= random.sample(range(900), train_batch_size)
		x_feed1= x_train[mini_1]
		y_feed1= y_train[mini_1]
		mini_test= random.sample(range(100), test_batch_size)
		x_test_feed= x_test[mini_test]
		y_test_feed= y_test[mini_test]
		if i % (total_epoch/1000) ==0:
			trainResult=sess.run(merged, feed_dict={x_t1: x_feed1, y_t1: y_feed1, x_ph_test: x_test_feed, y_ph_test: y_test_feed})
			writer.add_summary(trainResult, i)
			f_loss.write(str(sess.run(Training_Loss, feed_dict={x_t1: x_feed1, y_t1: y_feed1, x_ph_test: x_test_feed, y_ph_test: y_test_feed})))
			f_loss.write(',')
			f_loss.write(str(sess.run(Valid_Loss, feed_dict={x_t1: x_feed1, y_t1: y_feed1, x_ph_test: x_test_feed, y_ph_test: y_test_feed})))
			f_loss.write('\n')
			print('Training Process:', i/total_epoch*100, '%')
			print('learningrate:', sess.run(learningRate))
			print('trainLoss:', sess.run(Training_Loss, feed_dict={x_t1: x_feed1, y_t1: y_feed1, x_ph_test: x_test_feed, y_ph_test: y_test_feed}))
			print('testLoss:', sess.run(Valid_Loss, feed_dict={x_t1: x_feed1, y_t1: y_feed1, x_ph_test: x_test_feed, y_ph_test: y_test_feed}))
		sess.run(train_step, feed_dict={x_t1: x_feed1, y_t1: y_feed1, x_ph_test: x_test_feed, y_ph_test: y_test_feed})

print('Training Finished')
# print(h11, h12, h21, h22, h23, h31, h32, h33, h41, h42, h43, h_bw_31_partial, h_bw_31, h_bw_32, h_bw_33, h_bw_21_partial, h_bw_21, h_bw_22, h_bw_23, h_bw_11_partial, h_bw_11, h_bw_12, h_bw_13)

########################### Parameters storage ############################
savePath=saver.save(sess, saveDir+ '/parameters.ckpt')
print('Model params saved in: ', savePath)

