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

        input_1 = tf.placeholder(shape=[None, None, 3], name="input_image", dtype=tf.uint8)
        new_img_dims = tf.expand_dims(input_1, 0)
        # Load a (frozen) Tensorflow model into memory
    
        with tf.gfile.GFile(pb_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='', input_map={"image_tensor:0": new_img_dims})
        self._generate_graph(score_threshold,num_classes)

    def _generate_graph(self,score_threshold,num_classes):
        # Get handles to input and output tensors
        #print( tf.get_default_graph().get_tensor_by_name('image_tensor:0'))
        tensor_num= tf.get_default_graph().get_tensor_by_name('num_detections:0')
        
        tensor_scores= tf.get_default_graph().get_tensor_by_name('detection_scores:0')
        #print(tensor_scores)
        tensor_boxes= tf.get_default_graph().get_tensor_by_name('detection_boxes:0')
        tensor_classes= tf.get_default_graph().get_tensor_by_name('detection_classes:0')

        #print(tensor_dict)
        num_detections= tf.cast(tensor_num[0],tf.int32)
        detection_scores=tensor_scores[0]
        detection_boxes = tensor_boxes[0]
        detection_classes = tf.cast(tensor_classes[0],tf.int32)
        #mask=tf.constant(detection_scores >= score_threshold,dtype=tf.bool,shape=detection_classes.get_shape())
        
        mask=detection_scores >= score_threshold
        #print(mask)
        scores_ =tf.boolean_mask(detection_scores,mask)
        boxes_ =tf.boolean_mask(detection_boxes,mask)
        classes_ =tf.boolean_mask(detection_classes,mask)
        num_=tf.shape(scores_)
        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_nodes={}
        output_nodes['num'] = tf.identity(num_, name="output_num")
        output_nodes['classes'] = tf.identity(classes_, name="output_classes") 
        output_nodes['boxes'] = tf.identity(boxes_, name="output_boxes") 
        output_nodes['scores'] =tf.identity(scores_, name="output_scores") 

    






if __name__ == '__main__':
    # loading model from:
    model_load_from = 1
    # 0: freezed tf interface pb
    MODEL_tf_path = 'ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb'
    # classify score threshold, value will be fixed to output freezed
    MODEL_score_threshold = 0.1
    # 1: freezed unity interface pb
    MODEL_unity_path = 'ssdlite_mobilenet_v2_coco_2018_05_09/freezed_coco_zoo.pb'
    
    # args
    CLASSES_path = 'object_detection/data/mscoco_label_map.pbtxt'
    CLASSES_num=1
    
   

    # doing detection:
    do_detect = 1
    # 0: no action
    # 1: img
    IMG_path = 'demo/boys.jpg'
    # 2: video
    VIDEO_path = 'demo/Raccoon.mp4'
    OUTPUT_video = ""
    # args  
    DRAW_score_threshold = 0.5  # score filter for draw boxes
    FORCE_image_resize = (416, 416) # (height,width) 

    # interface convert output:
    do_output_freezed_unity_interface = 0
    # 0: no action
    # 1: tf-->unity freezed interface pb
    OUTPUT_pb_path = "output"
    OUTPUT_pb_file = "freezed_coco_zoo.pb"
 


detection_graph = tf.Graph()
with detection_graph.as_default():
    with tf.Session() as sess:
        if model_load_from == 0:
            model=TF_ZOO(CLASSES_num)
            model.load_model_by_tf_interface(MODEL_tf_path,CLASSES_num,MODEL_score_threshold)
        elif model_load_from == 1:
            load_unity_interface_frozen_pb(MODEL_unity_path)
        with tf.Session() as sess:
            get_input,get_output=get_nodes(sess)
            if model_load_from == 0 and do_output_freezed_unity_interface == 1:
                write_unity_interface_frozen_pb( sess,OUTPUT_pb_path,OUTPUT_pb_file,get_input,get_output)
            else:
                print("no output pb")
            if do_detect == 1:
                detect_image(sess,IMG_path,CLASSES_num,CLASSES_path,'id',get_input,get_output,DRAW_score_threshold,FORCE_image_resize,False)
            elif do_detect == 2:
                detect_video(sess, VIDEO_path,CLASSES_num,CLASSES_path,'id',get_input,get_output,DRAW_score_threshold,FORCE_image_resize,False,OUTPUT_video)
            else:
                print("no detect")






