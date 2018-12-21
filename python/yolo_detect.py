# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Input
from keras.models import load_model
from keras.utils import multi_gpu_model
from yolo3.model import tiny_yolo_body, yolo_body
from utils_y import yolo_eval
from utils import *


class YOLO(object):

    def __init__(self, classes_num, anchors_path):
        
        self.classes_num=classes_num
        self.anchors = self._get_anchors(anchors_path)
        

    def _get_anchors(self, anchors_path_raw):
        anchors_path = os.path.expanduser(anchors_path_raw)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def load_model_by_h5(self, model_h5_path, model_score_threshold, iou_threshold, gpu_num):
        model_path = os.path.expanduser(model_h5_path)
        assert model_path.endswith(
            '.h5'), 'Keras model or weights must be a .h5 file.'
        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        is_tiny_version = num_anchors == 6  # default setting

        try:
            yolo_model = load_model(model_path, compile=False)
        except:
            yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors//2, self.classes_num) \
            if is_tiny_version else yolo_body(Input(shape=(None, None, 3)), num_anchors//3, self.classes_num)
            # make sure model, anchors and classes match
            yolo_model.load_weights(model_h5_path)
        else:
            assert yolo_model.layers[-1].output_shape[-1] == \
            num_anchors/len(yolo_model.output) * (self.classes_num + 5), \
            'Mismatch between model and given anchor and class sizes'

        if gpu_num >= 2:
            yolo_model = multi_gpu_model(yolo_model, gpus=gpu_num)
        self._generate_graph(yolo_model, self.classes_num,
                             model_score_threshold, iou_threshold)
       

    def _generate_graph(self, model_body, num_classes, model_score_threshold, iou_threshold):
        # Generate output tensor targets for filtered bounding boxes.
        #self.input_0 = K.placeholder(
        #    shape=(2), name="return_box_shape", dtype="int32")
        self.input_1 = tf.placeholder(
            shape=(None, None, 3), name="input_image",dtype="uint8")
        new_img = tf.cast(self.input_1, tf.float32) /255.
        new_img_dims = tf.expand_dims(new_img, 0)
        out = model_body(new_img_dims)
    
        boxes, scores, classes,num = yolo_eval(out,
                                           self.anchors,
                                           num_classes,
                                           #self.input_0,
                                           score_threshold=model_score_threshold,
                                           iou_threshold=iou_threshold)
        self.output_nodes={}
        self.output_nodes['boxes'] = tf.identity(boxes, name="output_boxes")
        self.output_nodes['scores'] = tf.identity(scores, name="output_scores")
        self.output_nodes['classes'] = tf.identity(classes, name="output_classes")
        self.output_nodes['num'] = tf.identity(num, name="output_num")
        

    


if __name__ == '__main__':
    # loading model from:
    model_load_from = 1
    # 0: h5
    MODEL_h5_path = 'model_data/yolov3.h5'
    MODEL_score_threshold = 0.1 # classify score threshold, value will be fixed to output freezed
    IOU_threshold = 0.1  # yolo iou box filter, value will be fixed to output freezed
    GPU_num = 1  # video cards count , cpu version or gpu version with counts will fixed after convert to pb
    # 1: freezed unity interface pb
    MODEL_pb_path = 'output/freezed_coco_yolo.pb'
    # args
    ANCHORS_path = 'model_data/yolov3_anchors.txt'
    CLASSES_path='model_data/coco_labels_map.pbtxt'
    CLASSES_num = 80
   


    # doing detection:
    do_detect = 1
    # 0: no action
    # 1: img
    IMG_path = 'demo/boys.jpg'
    # 2: video
    VIDEO_path = 'demo/Raccoon.mp4'
    OUTPUT_video = ""
    # args   
    DRAW_score_threshold = 0.1  # score filter for draw boxes
    FORCE_image_resize = (416, 416) # (height,width) 'Multiples of 32 required' , resize input to model

    # keras h5 convert to freezed graph output:
    do_output_freezed_unity_interface = 0
    # 0: no action
    # 1: h5-->unity_interface freezed pb
    OUTPUT_pb_path = "output"
    OUTPUT_pb_file = "freezed_coco_yolo.pb"
   
    


detection_graph = tf.Graph()
with detection_graph.as_default():
    if model_load_from == 0:
        model=YOLO(CLASSES_num, ANCHORS_path)
        model.load_model_by_h5(MODEL_h5_path, MODEL_score_threshold, IOU_threshold, GPU_num)
    elif model_load_from == 1:
        load_unity_interface_frozen_pb(MODEL_pb_path)
    with K.get_session() as sess:
        get_input,get_output=get_nodes(sess)
        if model_load_from == 0 and do_output_freezed_unity_interface == 1:
            write_unity_interface_frozen_pb( sess,OUTPUT_pb_path,OUTPUT_pb_file,get_input,get_output)
        else:
            print("no output pb")
        if do_detect == 1:
            detect_image(sess,IMG_path,CLASSES_num,CLASSES_path,'index',get_input,get_output,DRAW_score_threshold,FORCE_image_resize,True)
        elif do_detect == 2:
            detect_video(sess, VIDEO_path,CLASSES_num,CLASSES_path,'index',get_input,get_output,DRAW_score_threshold,FORCE_image_resize,True, OUTPUT_video)
        else:
            print("no detect")
      
        
