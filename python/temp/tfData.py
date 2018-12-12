import os
import numpy as np
import cv2
import tensorflow as tf
from utils import dataset_util
import ujson
import glob
import os.path
folder = "D:\\winpython\\py36\\work\\UnityEyes_Windows\\imgs"
flags = tf.app.flags
flags.DEFINE_string('output_path', 'train.tfrecords',
                    'Path to output TFRecord')
FLAGS = flags.FLAGS
height = 600
width = 800

def rect(path):
    target='iris_2d'
    with open(path, 'r') as f:
        json_data = ujson.load(f)
    landmarks=np.zeros([len(json_data[target]),2], dtype=np.int16)
    for i, val in enumerate(json_data[target]):
        #landmarks[i,:2]=np.asarray(eval(np.asarray(json_data[target])[i]))[:2]
        new=np.array([np.int_(eval(val)[0]),np.int_((height-eval(val)[1]))])
        landmarks[i,:2]=new
        
    #print(landmarks)
    # get rect from eye landmarks
    border = 0
    l_ul_x = min(landmarks[:,0])
    l_ul_y = min(landmarks[:,1])
    l_lr_x = max(landmarks[:,0])
    l_lr_y = max(landmarks[:,1])
    #print(l_ul_x, l_ul_y,l_lr_x,l_lr_y)
    long= max(l_lr_y-l_ul_y,l_lr_x-l_ul_x)
    paddingX=np.int_(np.sum(long-(l_lr_x-l_ul_x))/2)
    paddingY=np.int_(np.sum(long-(l_lr_y-l_ul_y))/2) 
    paddingX=0
    paddingY=0
    pt1 = np.sum(l_ul_x)-border-paddingX, np.sum(l_ul_y)-border-paddingY
    pt2 = np.sum(l_lr_x)+border+paddingX, np.sum(l_lr_y)+border+paddingY
    return pt1,pt2
def create_tf_example(name):
    print(name)
    # TODO(user): Populate the following variables from your example.
    path_jpg = os.path.join(folder, name+'.jpg')
    path_json = os.path.join(folder, name+'.json')
    filename = bytes(path_jpg, encoding='utf-8') # Filename of the image. Empty if image is not from file
    with tf.gfile.GFile(path_jpg, 'rb') as fid:
        encoded_jpg = bytes(fid.read())
    #encoded_jpg=bytes(cv2.imread(path_jpg))
    #encoded_jpg = tf.gfile.FastGFile(path_jpg, 'rb').read()
    encoded_image_data = encoded_jpg # Encoded image bytes
    image_format = b'jpg' # b'jpeg' or b'png'
    ul,lr=rect(path_json)
    xmins = [ul[0]] # List of normaized left x coordinates in bounding box (1 per box)
    xmaxs = [lr[0]] # List of normalized right x coordinates in bounding box
             # (1 per box)
    ymins = [ul[1]] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [lr[1]] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
    #print(xmins,ymins,xmaxs,ymaxs)
    classes_text = [b'iris'] # List of string class name of bounding box (1 per box)
    classes = [1] # List of integer class id of bounding box (1 per box)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def file_extension(path):
    return os.path.splitext(os.path.split(path)[1])[0]


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    # TODO(user): Write code to read in your dataset to examples variable
    examples=glob.glob(folder+'\\*.jpg')
    

    
    for example in examples:
        tf_example = create_tf_example(file_extension(example))
        writer.write(tf_example.SerializeToString())
    writer.close()    
    
    """
    tf_example = create_tf_example("10")
    writer.write(tf_example.SerializeToString())
    writer.close()
    """


if __name__ == '__main__':
  tf.app.run()
