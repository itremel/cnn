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

##kopiert von miist_deep aber mit nur einem conv und pooling layer und ohne dropout
def cnn(x):
    
    # Reshape to use within a convolutional neural net.
    # Last dimension is for "features" - there is only one here, since images are
    # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        
    # Convolutional layer - maps one grayscale image to 32 feature maps.
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
    
def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


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

    for filename in os.listdir('hand_images/'):
        for regex, label in regexLabelList:
            if re.search(regex, filename) != None:
                imageFilenames.append('hand_images/' + filename)
                imageLabels.append(label)

    filenames = tf.constant(imageFilenames)
    labels = tf.constant(imageLabels)

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(_parse_function)
    #iterator = dataset.make_one_shot_iterator()
    #features, labelsS = iterator.get_next()
    #features = tf.reshape(features, [-1,784])
    #labelsS = tf.reshape(labelsS, [-1,784])

    return dataset

def main():
    dataset = create_dataset()
    print("created dataset")
    batchsize=10
    batched_dataset = dataset.batch(batchsize)
    it = batched_dataset.make_initializable_iterator()
    next_element = it.get_next()
    #iterator = batched_dataset.make_one_shot_iterator()
    #next_element = iterator.get_next()
    #with tf.Session() as sess:
        #print(sess.run(next_element)[1])
        #print(sess.run(next_element)[1])
        #print(sess.run(next_element)[1])
    #with tf.Session() as sess:
        #for _ in range(100):
            #sess.run(it.initializer)
        #while True:
            #try:
                #print(sess.run(next_element)[1])
                #print("\n")
            #except tf.errors.OutOfRangeError:
                #break
    #return;
    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    # Build the graph for the deep net
    y_conv = cnn(x)
    
    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    graph_location = tempfile.mkdtemp()
    print('Saving graph to: %s' % graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #print(sess.run(features))
        #image=np.array(sess.run(features))
        #label=np.array(sess.run(labels))
        #print(image)
        for i in range(5):
            sess.run(it.initializer)
            #batch = mnist.train.next_batch(50)
            #batch = data.batch(50)
            #iterator = dataset.make_one_shot_iterator()
            #next_element = iterator.get_next()
            while True:
                try:
                    #print(sess.run(next_element)[1])
                    element=sess.run(next_element)
                    image = np.reshape(element[0], [batchsize,784])
                    #print(image.shape)
                    label = np.reshape(element[1],[batchsize,10])
                    if i % 2 == 0:
                        train_accuracy = accuracy.eval(feed_dict={x: image, y_: label})
                        print('step %d, training accuracy %g' % (i, train_accuracy))
                    train_step.run(feed_dict={x: image, y_: label})
                except tf.errors.OutOfRangeError:
                    break
    #iterator = dataset.make_one_shot_iterator()
    #next_element = iterator.get_next()
    #while True:
        #try:
            #print(sess.run(next_element[0]))
            #element=sess.run(next_element)
            #image = np.reshape(element[0], [1000,784])
            #label = np.reshape(element[1],[1000,10])
            #print('test accuracy %g' % accuracy.eval(feed_dict={x: image, y_: label}))
        #except tf.errors.OutOfRangeError:
            #break
        max_value = tf.placeholder(tf.int64, shape=[])
        iterator = dataset.make_initializable_iterator()
        sess.run(iterator.initializer, feed_dict={max_value: 1000})

        pos = np.random.randint(400, high=1000)
        print("find random postion....")
        for _ in range(pos):
            sess.run(iterator.get_next())
        tensor = iterator.get_next()

        bla = sess.run(tensor)
        image = np.reshape(bla[0], [1,784])
        #print(image.shape)
        label = np.reshape(bla[1],[1,10])
        print('test accuracy %g' % accuracy.eval(feed_dict={x: image, y_: label}))
        print(bla[0].shape)
        print(bla[1])
        plt.imshow(np.reshape(bla[0], [28,28]))
        plt.show()

            
if __name__ == '__main__':
    main()
