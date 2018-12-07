# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from tensorflow.python.platform import gfile
from utils import *





class TF_ZOO(object):

    def __init__(self, classes_num):
        #self.class_names = self._get_class(classes_path)
        self.classes_num=classes_num 

    def load_model_by_tf_interface(self,pb_path,num_classes,score_threshold):

        input_1 = tf.placeholder(shape=[None, None, 3], name="input_image_data", dtype=tf.uint8)
        new_img_dims = tf.expand_dims(input_1, 0)
        # Load a (frozen) Tensorflow model into memory
    
        with tf.gfile.GFile(pb_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='', input_map={"image_tensor:0": new_img_dims})
        self._generate_graph(score_threshold,num_classes)

    def _generate_graph(self,score_threshold,num_classes):
        # Get handles to input and output tensors
        tensor_num= tf.get_default_graph().get_tensor_by_name('num_detections:0')
        tensor_scores= tf.get_default_graph().get_tensor_by_name('detection_scores:0')
        tensor_boxes= tf.get_default_graph().get_tensor_by_name('detection_boxes:0')
        tensor_classes= tf.get_default_graph().get_tensor_by_name('detection_classes:0')

        #print(tensor_dict)
        num_detections= tf.cast(tensor_num[0],tf.int32)
        detection_scores=tensor_scores[0]
        detection_boxes = tensor_boxes[0]
        detection_classes = tf.cast(tensor_classes[0],tf.uint8)
        mask=detection_scores >= score_threshold
        scores_ =tf.boolean_mask(detection_scores,mask)
        boxes_ =tf.boolean_mask(detection_boxes,mask)
        classes_ =tf.boolean_mask(detection_classes,mask)
        num_=tf.shape(scores_)
        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_nodes={}
        output_nodes['num'] = tf.identity(num_, name="out_num")
        output_nodes['classes'] = tf.identity(classes_, name="out_classes") 
        output_nodes['boxes'] = tf.identity(boxes_, name="out_boxes") 
        output_nodes['scores'] =tf.identity(scores_, name="out_scores") 

    






if __name__ == '__main__':
    # loading model from:
    # 0: freezed tf interface pb
    # 1: freezed unity interface pb
    model_load_from = 0
    # args
    MODEL_tf_path = 'model_data/ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb'
    MODEL_unity_path = 'model_data/ssdlite_mobilenet_v2_coco_2018_05_09/frozen_unity_inference_graph.pb'
    #CLASSES_path = 'object_detection/data/mscoco_label_map.pbtxt'
    CLASSES_path='model_data/coco_classes90.json'
    CLASSES_num=90
    # classify score threshold, value will be fixed to output freezed
    MODEL_score_threshold = 0.1
   

    # doing detection:
    # 0: no action
    # 1: img
    # 2: video
    do_detect = 1
    # args
    IMG_path = 'demo/car_cat.jpg'
    VIDEO_path = 'demo/Raccoon.mp4'
    OUTPUT_video = ""
    DRAW_score_threshold = 0.1  # score filter for draw boxes
    # (height,width) 
    FORCE_image_resize = (128, 128)

    # interface convert output:
    # 0: no action
    # 1: tf-->unity freezed interface pb
   
    do_output_freezed_unity_interface = 0
    # args
    OUTPUT_pb_path = "model_data/ssdlite_mobilenet_v2_coco_2018_05_09"
    OUTPUT_pb_file = "frozen_unity_inference_graph.pb"


detection_graph = tf.Graph()
with detection_graph.as_default():
    with tf.Session() as sess:
        if model_load_from == 0:
            model=TF_ZOO(CLASSES_num)
            model.load_model_by_tf_interface(MODEL_tf_path,CLASSES_num,MODEL_score_threshold)
        elif model_load_from == 1:
            load_unity_interface_frozen_pb(MODEL_unity_path,CLASSES_num)
        with tf.Session() as sess:
            get_input,get_output=get_nodes(sess)
            if model_load_from == 0 and do_output_freezed_unity_interface == 1:
                write_unity_interface_frozen_pb( sess,OUTPUT_pb_path,OUTPUT_pb_file,get_input,get_output)
            else:
                print("no output pb")
            if do_detect == 1:
                detect_image(sess,IMG_path,CLASSES_path,get_input,get_output,DRAW_score_threshold,FORCE_image_resize,False)
            elif do_detect == 2:
                detect_video(sess, VIDEO_path,CLASSES_path,get_input,get_output,DRAW_score_threshold,FORCE_image_resize,False,OUTPUT_video)
            else:
                print("no detect")






