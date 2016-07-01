from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import os
import sys
from IPython.display import display, Image
from six.moves import cPickle as pickle
import tensorflow as tf

#load data first
pickle_file = '/home/nfs/mjmu/haiming/data/visibility/' +  'ASOS_alone.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset= save['train_dataset']
    validate_dataset = save['validate_dataset']
    test_dataset = save['test_dataset']
    test_old = save['test_dataset_evan']
    del save
    
train_time = train_dataset['time']
train_data = train_dataset['data']
train_label = train_dataset['label']
validate_time = validate_dataset['time']
validate_data = validate_dataset['data']
validate_label = validate_dataset['label']
test_time = test_dataset['time']
test_data = test_dataset['data']
test_label = test_dataset['label']
test_old_data = test_old['data']
test_old_label = test_old['label']

#dataset normalize
mean = train_data.mean(axis = 0)
std = train_data.std(axis = 0)
print(mean.shape, std.shape)
train_data_n = (train_data - mean)/std
validate_data_n = (validate_data - mean)/std
test_old_data_n = (test_old_data - mean)/std

#train normal with gradient descent training
graph = tf.Graph()
with graph.as_default():
    tf_train_dataset = tf.constant(train_data_n)
	tf_train_label = tf.constant(train_label)
	tf_test_dataset = tf.constant(test_old_data_n)
	tf_test_label = tf.constant(test_old_label)
	weights = tf.Variable(tf.truncated_normal([train_data_n.shape[1], 1))
	biases = tf.Variable(tf.zeros([1]))
	
	def model(X, w, b):
	    return tf.matmul(X, w) + b
	
	predicted_label = model(tf_train_dataset, weights, biases)
	loss = tf.square(predicted_label - tf_test_label)
	op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)


num_steps = 801	
with tf.Session(graph = graph) as session:
    tf.initialize_all_variables().run()
	print('Initialized')
	for step in range(num_steps):
	    _, l, _ = session.run([predicted_label, loss, op])
		if (step % 100 == 0):
            print('Loss at step %d: %f' % (step, l))