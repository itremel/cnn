from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile
import re
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#set to True to use inception Module, Flase to use convolution and max pooling
use_inception = True

#set True to restore a previous Model
use_previous = False
#file path to previous Model
file_path = os.getcwd()+'/model.ckpt'

#batchsize for the test and training Data
batchsize = 50

#number of training steps for the Model
num_steps = 500

#creates a cnn with one convolution layer one max pooling layer and one fully connected layer
def cnn(x):
    
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        
    # Convolutional layer - maps 1 input image to 32 feature maps.
    with tf.name_scope('conv'):
        W_conv = weight_variable([5, 5, 1, 32])
        b_conv = bias_variable([32])
        h_conv = tf.nn.relu(conv2d(x_image, W_conv) + b_conv)
      
    # Pooling layer - downsamples by 2X.
    with tf.name_scope('pool'):
        h_pool = max_pool_2x2(h_conv)
        
    # Fully connected layer 1 -- after 1 round of downsampling, our 28x28 image
    # is down to 14x14x32 feature maps -- maps this to 1024 features.
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([14 * 14 * 32, 1024])
        b_fc1 = bias_variable([1024])

        h_pool_flat = tf.reshape(h_pool, [-1, 14*14*32])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)
        
    # Map the 1024 features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

        y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
    return y_conv

#creates a cnn with one Inception layer and one fully conected layer
def cnn_inception(x):
    
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', x_image, 10)
        
    #inception layer - performs a 1x1, 3x3, 5x5 convolution and 3x3 max pooling and concatenates the results
    with tf.name_scope('inception'):
        #follows input
        W_conv1_1x1_1 = weight_variable([1, 1, 1, 32])
        b_conv1_1x1_1 = bias_variable([32])
        h_conv1_1x1_1 = conv2d(x_image,W_conv1_1x1_1)+b_conv1_1x1_1
     
        #follows input
        W_conv1_1x1_2 = weight_variable([1, 1, 1, 16])
        b_conv1_1x1_2 = bias_variable([16])
        h_conv1_1x1_2 = tf.nn.relu(conv2d(x_image,W_conv1_1x1_2)+b_conv1_1x1_2)
     
        #follows input
        W_conv1_1x1_3 = weight_variable([1, 1, 1, 16])
        b_conv1_1x1_3 = bias_variable([16])
        h_conv1_1x1_3 = tf.nn.relu(conv2d(x_image,W_conv1_1x1_3)+b_conv1_1x1_3)
        
        #follows 1x1_2
        W_conv1_3x3 = weight_variable([3, 3, 16, 32])
        b_conv1_3x3 = bias_variable([32])
        h_conv1_3x3 = conv2d(h_conv1_1x1_2,W_conv1_3x3)+b_conv1_3x3
        
        #follows 1x1_3
        W_conv1_5x5 = weight_variable([5, 5, 16, 32])
        b_conv1_5x5 = bias_variable([32])
        h_conv1_5x5 = conv2d(h_conv1_1x1_3,W_conv1_5x5)+b_conv1_5x5
        
        #follows input
        h_maxpool1 = max_pool_3x3(x_image)
        
        #follows max pooling
        W_conv1_1x1_4= weight_variable([1, 1, 1, 32])
        b_conv1_1x1_4= bias_variable([32])
        h_conv1_1x1_4 = conv2d(h_maxpool1,W_conv1_1x1_4)+b_conv1_1x1_4
        
        #concatenate the feature maps
        h_inception = tf.nn.relu(tf.concat([h_conv1_1x1_1, h_conv1_3x3, h_conv1_5x5, h_conv1_1x1_4], 3))
        
    # Fully connected layer 1 - after inception layer
    with tf.name_scope('fc1'):
        W_fc1 = weight_variable([28 *28 * 4*32, 1024])
        b_fc1 = bias_variable([1024])

        h_inception_flat = tf.reshape(h_inception, [-1, 28*28*4*32])
        h_fc1 = tf.nn.relu(tf.matmul(h_inception_flat, W_fc1) + b_fc1)
        
    # Map the 1024 features to 10 classes, one for each digit
    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([1024, 10])
        b_fc2 = bias_variable([10])

        y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
    tf.summary.histogram('y_conv', y_conv)
    return y_conv
    
def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
                        
def max_pool_3x3(x):
  """max_pool_3x3 with stride size of 1"""
  return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                        strides=[1, 1, 1, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
  
# Reads an image from a file, decodes it into a dense tensor, crops, and resizes it to a fixed shape.
def _parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_image(image_string, channels=1)
  image_decoded.set_shape([384, 288, 1])
  image_cropped = tf.image.crop_to_bounding_box(image_decoded, 70, 125, 150, 150)
  image_resized = tf.image.resize_images(image_cropped, [28, 28])
  
  return image_resized, label
  
#creates a dataset from the hand_images and the corresponding labels
def create_dataset():
    regexLabelList = [[re.compile('image_A_.*' ),[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                      [re.compile('image_Ae_.*'),[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]],
                      [re.compile('image_M_.*' ),[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]],
                      [re.compile('image_O_.*' ),[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]],
                      [re.compile('image_Oe_.*'),[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]],
                      [re.compile('image_T_.*' ),[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]],
                      [re.compile('image_U_.*' ),[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]],
                      [re.compile('image_Ue_.*'),[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]],
                      [re.compile('image_V_.*' ),[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]],
                      [re.compile('image_Y_.*' ),[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]]


    imageFilenames = []
    imageLabels = []
    testimageFilenames = []
    testimageLabels = []
    
    counter = 0 

    for filename in os.listdir('hand_images/'):
        for regex, label in regexLabelList:
            if re.search(regex, filename) != None:
                if counter%4 != 0:
                    imageFilenames.append('hand_images/' + filename)
                    imageLabels.append(label)
                else:
                    testimageFilenames.append('hand_images/' + filename)
                    testimageLabels.append(label)
                counter += 1
                
    if counter == 0:
        print('No hand images found')
        return 

    filenames = tf.constant(imageFilenames)
    labels = tf.constant(imageLabels)
    testfilenames = tf.constant(testimageFilenames)
    testlabels = tf.constant(testimageLabels)

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(_parse_function)
    testdataset = tf.data.Dataset.from_tensor_slices((testfilenames, testlabels))
    testdataset = testdataset.map(_parse_function)

    return dataset, testdataset

def main():
    #create datasets and iterators
    dataset, testdataset = create_dataset()
    print("created dataset")
    
    batched_dataset = dataset.batch(batchsize)
    it = batched_dataset.make_initializable_iterator()
    next_batch = it.get_next()
    
    batched_test_data = testdataset.batch(batchsize)
    test_it = batched_test_data.make_initializable_iterator()
    next_test_batch = test_it.get_next()
    
    # Create the model
    
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])

    # Build the graph for the deep net
    if use_inception:
        print('using inception layer')
        y_conv = cnn_inception(x)
    else:
        print('not using inception layer')
        y_conv = cnn(x)
    
    #define loss and optimizer
    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        
    with tf.name_scope('test_prediction'):
        test_prediction = tf.nn.softmax(y_conv)
        
    #define accuracy for training and test data
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()
    graph_location = tempfile.mkdtemp()
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())
    
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        #use previos model
        if use_previous:
            saver.restore(sess, file_path)
            print('previous Model restored')
        
        #training
        print('starting training')
        sess.run(it.initializer)
        for i in range(num_steps):       
            train_accuracy = []
            
            try:
                batch = sess.run(next_batch)
                image = np.reshape(batch[0], [batchsize,784])
                label = np.reshape(batch[1],[batchsize,10])
                   
                train_step.run(feed_dict={x: image, y_: label})
                summary, acc = sess.run([merged, accuracy], feed_dict={x:image, y_:label})
                train_accuracy.append(acc)
                
                train_writer.add_summary(summary, i)

                if i % 50 == 0:
                    avg_accuracy = np.sum(train_accuracy)/len(train_accuracy)
                    print('step %d, average training accuracy %g' % (i, avg_accuracy))
            
            except tf.errors.OutOfRangeError:
                sess.run(it.initializer)
                i -= 1
                    
            
        #tests
        sess.run(test_it.initializer)      
        test_accuracy = []
        
        print('starting testruns')
        while True:
            try:                
                test_batch=sess.run(next_test_batch)
                image = np.reshape(test_batch[0], [batchsize,784])
                label = np.reshape(test_batch[1], [batchsize,10])
                
                test_accuracy.append(accuracy.eval(feed_dict={x: image, y_: label}))
                
            except tf.errors.OutOfRangeError:
                break
                
        avg_accuracy = np.sum(test_accuracy)/len(test_accuracy)
        print('average test accuracy %g' % avg_accuracy)
        
        saver.save(sess, file_path)
        print('Model saved')
            
if __name__ == '__main__':
    main()
