import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import re
import os

# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def _parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_image(image_string, channels=1)
  image_decoded.set_shape([384, 288, 1])
  #image_cropped = tf.image.central_crop(image_decoded, 0.5)
  image_cropped = tf.image.crop_to_bounding_box(image_decoded, 70, 125, 150, 150)
  image_resized = tf.image.resize_images(image_cropped, [28, 28])
  
  #image = sess.run(image_resized)
  #plt.imshow(image_decoded)
  #plt.show()
  
  return image_resized, label
  
sess = tf.Session()

regexLabelList = [[re.compile('image_A_.*'),[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                  [re.compile('image_Ae_.*'),[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]],
                  [re.compile('image_M_.*'),[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]],
                  [re.compile('image_O_.*'),[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]],
                  [re.compile('image_Oe_.*'),[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]],
                  [re.compile('image_T_.*'),[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]],
                  [re.compile('image_U_.*'),[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]],
                  [re.compile('image_Ue_.*'),[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]],
                  [re.compile('image_V_.*'),[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]],
                  [re.compile('image_Y_.*'),[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]]


imageFilenames = []
imageLabels = []

for filename in os.listdir('hand_images/'):
    for regex, label in regexLabelList:
        if re.search(regex, filename) != None:
            imageFilenames.append('hand_images/' + filename)
            #print('hand_images/' + filename)
            imageLabels.append(label)
            #print(label)

for i in range(len(imageFilenames)):
    print(imageFilenames[i], imageLabels[i])

# A vector of filenames.
filenames = tf.constant(imageFilenames)

# labels[i] is the label for the image in filenames[i].
labels = tf.constant(imageLabels)

dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(_parse_function)

max_value = tf.placeholder(tf.int64, shape=[])
iterator = dataset.make_initializable_iterator()
sess.run(iterator.initializer, feed_dict={max_value: 1000})

pos = np.random.randint(0, high=1000)

for _ in range(pos):
    sess.run(iterator.get_next())
tensor = iterator.get_next()

bla = sess.run(tensor)

print(imageFilenames[pos])
print(bla[1])
plt.imshow(np.reshape(bla[0], [28,28]))
plt.show()

