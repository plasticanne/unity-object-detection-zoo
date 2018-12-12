import tensorflow as tf
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
path = os.path.join(os.getcwd(),"dataset","raccoon", "data","train.record")
record_iterator = tf.python_io.tf_record_iterator(path)
# print(list(record_iterator))
with tf.Session() as sess:
    for string_record in record_iterator:

        example = tf.train.Example()
        example.ParseFromString(string_record)


        height = example.features.feature['image/height'].int64_list.value
        print(height)
        width = example.features.feature['image/width'].int64_list.value
        print(width)
        filename = (example.features.feature['image/filename'].bytes_list.value)
        print(filename)
        xmin = (example.features.feature['image/object/bbox/xmin'].float_list.value)
        print(xmin)
        label = (example.features.feature['image/object/class/text'].bytes_list.value)
        print(label)
        image_encoded = (example.features.feature['image/encoded'].bytes_list.value)
        #image_1d = np.fromstring(image_encoded, dtype=np.uint8)
        #image = image_1d.reshape((height, width, 3))
        #cv2.imshow("image", image)
        #cv2.waitKey(0)
        #image = tf.image.decode_jpeg(image_encoded).eval()
        #plt.imshow(image)
        #plt.show()
        #break
        
