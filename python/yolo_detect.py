# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from keras.utils import multi_gpu_model
from yolo3.model import tiny_yolo_body, yolo_body, yolo_eval2
from utils import *


class YOLO(object):

    def __init__(self, classes_num, anchors_path):
        
        self.classes_num=classes_num
        self.anchors = self._get_anchors(anchors_path)
        
        #self.sess = session

    

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


        yolo_model = load_model(model_path, compile=False)
        assert yolo_model.layers[-1].output_shape[-1] == \
            num_anchors/len(yolo_model.output) * (self.classes_num + 5), \
            'Mismatch between model and given anchor and class sizes'


        if gpu_num >= 2:
            yolo_model = multi_gpu_model(yolo_model, gpus=gpu_num)
        self._generate_graph(yolo_model, self.classes_num,
                             model_score_threshold, iou_threshold)
       

    def load_model_by_buider(self, weight_h5_path, model_score_threshold, iou_threshold, gpu_num):

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
       

        is_tiny_version = num_anchors == 6  # default setting

        self.yolo_model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors//2, self.classes_num) \
            if is_tiny_version else yolo_body(Input(shape=(None, None, 3)), num_anchors//3, self.classes_num)
        # make sure model, anchors and classes match
        self.yolo_model.load_weights(weight_h5_path)

        if gpu_num >= 2:
            yolo_model = multi_gpu_model(yolo_model, gpus=gpu_num)
        self._generate_graph(yolo_model, self.classes_num,
                             model_score_threshold, iou_threshold)

    def _generate_graph(self, model_body, num_classes, model_score_threshold, iou_threshold):
        # Generate output tensor targets for filtered bounding boxes.
        #self.input_0 = K.placeholder(
        #    shape=(2), name="return_box_shape", dtype="int32")
        self.input_1 = tf.placeholder(
            shape=(None, None, 3), name="input_image_data",dtype="uint8")
        new_img = tf.cast(self.input_1, tf.float32) /255.
        new_img_dims = tf.expand_dims(new_img, 0)
        out = model_body(new_img_dims)
    
        boxes, scores, classes,num = yolo_eval2(out,
                                           self.anchors,
                                           num_classes,
                                           #self.input_0,
                                           score_threshold=model_score_threshold,
                                           iou_threshold=iou_threshold)
        self.output_nodes={}
        self.output_nodes['boxes'] = tf.identity(boxes, name="out_boxes")
        self.output_nodes['scores'] = tf.identity(scores, name="out_scores")
        self.output_nodes['classes'] = tf.identity(classes, name="out_classes")
        self.output_nodes['num'] = tf.identity(num, name="out_num")
        

    


if __name__ == '__main__':
    # loading model from:
    # 0: h5
    # 1: freezed unity interface pb
    # 2: blider & h5 weights
    model_load_from = 0
    # args
    MODEL_h5_path = 'model_data/yolo.h5'
    MODEL_pb_path = 'model_data/freezed_coco_yolo.pb'
    ANCHORS_path = 'model_data/yolo_anchors.txt'
    CLASSES_path='model_data/coco_classes80.json'
    CLASSES_num = 80
    MODEL_weight_h5_path = ""
    # classify score threshold, value will be fixed to output freezed
    MODEL_score_threshold = 0.1
    IOU_threshold = 0.1  # yolo iou box filter, value will be fixed to output freezed
    GPU_num = 1  # video cards count , cpu version or gpu version with counts will fixed after convert to pb graph

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
    # (height,width) 'Multiples of 32 required' , resize input to model
    FORCE_image_resize = (128, 128)

    # keras h5 convert to freezed graph output:
    # 0: no action
    # 1: h5-->unity_interface freezed pb
    do_output_freezed_unity_interface = 0
    # args
    OUTPUT_pb_path = "./model_data"
    OUTPUT_pb_file = "freezed_coco_yolo.pb"


detection_graph = tf.Graph()
with detection_graph.as_default():
    if model_load_from == 0:
        model=YOLO(CLASSES_num, ANCHORS_path)
        model.load_model_by_h5(MODEL_h5_path, MODEL_score_threshold, IOU_threshold, GPU_num)
    elif model_load_from == 1:
        load_unity_interface_frozen_pb(MODEL_pb_path,CLASSES_num)
    elif model_load_from == 2:
        model=YOLO(CLASSES_num, ANCHORS_path)
        model.load_model_by_buider(MODEL_weight_h5_path)
    with K.get_session() as sess:
        get_input,get_output=get_nodes(sess)
        if model_load_from == 0 and do_output_freezed_unity_interface == 1:
            write_unity_interface_frozen_pb( sess,OUTPUT_pb_path,OUTPUT_pb_file,get_input,get_output)
        else:
            print("no output pb")
        if do_detect == 1:
            detect_image(sess,IMG_path,CLASSES_path,get_input,get_output,DRAW_score_threshold,FORCE_image_resize,True)
        elif do_detect == 2:
            detect_video(sess, VIDEO_path,CLASSES_path,get_input,get_output,DRAW_score_threshold,FORCE_image_resize,True, OUTPUT_video)
        else:
            print("no detect")
      
        