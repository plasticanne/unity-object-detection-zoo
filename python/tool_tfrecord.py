#from __future__ import absolute_import, division, print_function
import os

import cv2
import pandas as pd
import tensorflow as tf
from PIL import Image
from tool_classes import read_label_map,get_class_item
from object_detection.utils import dataset_util

import argparse
import glob
import io
import numbers
from collections import OrderedDict, namedtuple


def class_text_to_int(classes_list,row_label):
    for item in classes_list:
        if item["name"] == row_label:
            return item["id"]
        else:
            None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]



def xml_to_PdDataFrame(path):
    import xml.etree.ElementTree as ET
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df
def read_tf_record_test(path):
    with tf.Session() as sess:
        record_iterator = tf.python_io.tf_record_iterator(path)
        example = tf.train.Example()
        for string_record in record_iterator:
            example.ParseFromString(string_record)
            dict=read_tf_record_info_each(example)
            for k, v in dict.items():
                print ('%s: %s'%(k,v))
            print ('\n')
            image_encoded=read_tf_record_image_each(example)
            image=image_decode_jpeg(image_encoded)
            b,g,r = cv2.split(image)
            image=cv2.merge([r,g,b])
            for i,c in enumerate( list(dict["classes_label"])):
                image=cv2.rectangle(image, (int(dict["xmin"][i]*dict["width"][0]),int(dict["ymin"][i]*dict["height"][0])),  (int(dict["xmax"][i]*dict["width"][0]),int(dict["ymax"][i]*dict["height"][0])), (0, 255, 0), 2)
            cv2.imshow("image", image)
            cv2.waitKey(0)
            #break
def image_decode_jpeg(image_encoded):
    #image_1d = np.fromstring(image_encoded, dtype=np.uint8)
    #image = image_1d.reshape((height, width, 3))
    return tf.image.decode_jpeg(image_encoded[0]).eval()
def read_tf_record(path,with_image=False):
    with tf.Session() as sess:
        record_iterator = tf.python_io.tf_record_iterator(path)
        example = tf.train.Example()
        dict_list=[]
        for string_record in record_iterator:
            example.ParseFromString(string_record)
            dict=read_tf_record_info_each(example)
            if with_image:
                dict["image_encoded"]=read_tf_record_image_each(example)
            dict_list.append(dict)
        return dict_list

def read_tf_record_info_each(example):
    dict={}
    dict["height"] = example.features.feature['image/height'].int64_list.value
    dict["width"] = example.features.feature['image/width'].int64_list.value 
    dict["filename"] = (example.features.feature['image/filename'].bytes_list.value) 
    dict["image_format"] = (example.features.feature['image/format'].bytes_list.value)
    dict["xmin"] = (example.features.feature['image/object/bbox/xmin'].float_list.value)
    dict["xmax"] = (example.features.feature['image/object/bbox/xmax'].float_list.value)
    dict["ymin"] = (example.features.feature['image/object/bbox/ymin'].float_list.value)
    dict["ymax"] = (example.features.feature['image/object/bbox/ymax'].float_list.value)
    dict["classes_text"] = (example.features.feature['image/object/class/text'].bytes_list.value)   
    dict["classes_label"] = (example.features.feature['image/object/class/label'].int64_list.value)
    dict["classes_index"] = (example.features.feature['image/object/class/index'].int64_list.value) 
    return dict
def read_tf_record_image_each(example):
    image_encoded = (example.features.feature['image/encoded'].bytes_list.value)
    return image_encoded
  
    
def create_tf_example(group, img_folder,classes_path):
    with tf.gfile.GFile(os.path.join(img_folder, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    classes_index = []
    classes_dict_list=read_label_map(classes_path)

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        j,item=get_class_item(classes_dict_list,row['class'],'name')
        classes.append(item["id"])
        classes_index.append(j)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/class/index': dataset_util.int64_list_feature(classes_index),
    }))
    return tf_example



if __name__ == '__main__':
    

    
    parser = argparse.ArgumentParser(description='Classes formet converter between "classes_lines","label_map","json".')
    parser.add_argument(
    '--in_formet', type=str,
    default="xml",
    help='input format: pandas dataframe --in_formet="csv" or --in_formet="xml" ')
    parser.add_argument(
    '--input_annotations', type=str,
    default="dataset/raccoon/annotations",
    help='input path , "csv" for .csv file, "xml" for annotations folder contain all .xml ')
    parser.add_argument(
    '--input_image', type=str,
    default="dataset/raccoon/images",
    help='input image path , images folder contain all .jpg ')
    parser.add_argument(
    '--input_classes', type=str,
    default="model_data/raccoon_labels_map.pbtxt",
    help='input classes .pbtxt file')
    parser.add_argument(
    '--output', type=str,
    default="dataset/raccoon/raccoon.record",
    help='output tf-recoed file')
    parser.add_argument(
    '--test', type=str,
    default="",
    help='test a tf-recoed file')
    parser.add_argument(
    '--num_shards', type=int,
    default=1,
    help='how many tf-recoed files will be generated')
    parser.add_argument(
    '--num_pre_shards', type=int,
    default=1000,
    help='how many datas in every tf-recoed files')

    FLAGS = parser.parse_args()
    if FLAGS.test =="":
        if FLAGS.in_formet =="csv":
            examples = pd.read_csv(FLAGS.input_annotations)
        elif FLAGS.in_formet =="xml":
            examples = xml_to_PdDataFrame(FLAGS.input_annotations)

        grouped = split(examples, 'filename')
        if FLAGS.num_shards==1:
            with tf.python_io.TFRecordWriter(FLAGS.output) as f:
                for group in grouped:
                    tf_example = create_tf_example(group, FLAGS.input_image,FLAGS.input_classes)
                    f.write(tf_example.SerializeToString())
        elif FLAGS.num_shards>1 and FLAGS.num_pre_shards>=1:
            for i in range(FLAGS.num_shards):
                file_name = FLAGS.output+'-{}-of-{}'.format(i,FLAGS.num_shards)
                with tf.python_io.TFRecordWriter(file_name) as f:
                    for j in range(FLAGS.num_pre_shards):
                        tf_example = create_tf_example(grouped[i*FLAGS.num_pre_shards+j], FLAGS.input_image,FLAGS.input_classes)
                        f.write(tf_example.SerializeToString())
        else:
            print('--num_shards must >= 1')

            """import contextlib2
            from object_detection.dataset_tools import tf_record_creation_util
            with contextlib2.ExitStack() as tf_record_close_stack:
                output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
                    tf_record_close_stack, FLAGS.output, FLAGS.num_shards)
            for index,group in enumerate(grouped):
                tf_example = create_tf_example(group, FLAGS.input_image,FLAGS.input_classes)
                output_shard_index = index % FLAGS.num_shards
                output_tfrecords[output_shard_index].write(tf_example.SerializeToString())"""
                
    else: 
        read_tf_record_test(os.path.join(os.getcwd(),FLAGS.test))
