# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import os

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
import cv2
from keras import backend as K
from keras.layers import Input, Lambda
from keras.models import Model, Sequential, load_model
from keras.utils import multi_gpu_model
from tensorflow.image import ResizeMethod
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from tensorflow.python.tools import optimize_for_inference_lib

import colorsys
from timeit import default_timer as timer
from yolo3 import utils
from yolo3.model import tiny_yolo_body, yolo_body, yolo_eval,yolo_eval2


class YOLO(object):

    def __init__(self, classes_num, anchors_path, session):
        #self.class_names = self._get_class(classes_path)
        self.classes_num=classes_num
        self.anchors = self._get_anchors(anchors_path)
        #self._generate_colors()
        self.sess = session

    

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

        print('{} model, anchors, and classes loaded.'.format(model_path))

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

        print('{} model, anchors, and classes loaded.'.format(weight_h5_path))
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
    
        boxes, scores, classes,num = yolo_eval2(out,
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
        
    def load_model_by_pb(self, model_pb_path):
        model_path = os.path.expanduser(model_pb_path)
        # Load model, or construct model and load weights.

        with gfile.FastGFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            yolo_model = tf.import_graph_def(graph_def, name='')

        print('{} model, anchors, and classes loaded.'.format(model_path))
        

    def write_pb(self, output_pb_path, output_pb_file):
        self.input_nodes = [self.get_input["input_1"].name.split(":")[0]]
        self.output_nodes = [self.get_output["boxes"].name.split(":")[0], self.get_output["scores"].name.split(":")[
            0], self.get_output["classes"].name.split(":")[0]]
        print("input nodes:", self.input_nodes)
        print("output nodes:", self.output_nodes)
        constant_graph = graph_util.convert_variables_to_constants(
            self.sess, tf.get_default_graph().as_graph_def(), self.output_nodes)
        optimize_Graph = optimize_for_inference_lib.optimize_for_inference(
            constant_graph,
            self.input_nodes,  # an array of the input node(s)
            self.output_nodes,  # an array of output nodes
            tf.float32.as_datatype_enum)
        optimize_for_inference_lib.ensure_graph_is_valid(optimize_Graph)
        with tf.gfile.GFile(os.path.join(output_pb_path, output_pb_file), "wb") as f:
            f.write(constant_graph.SerializeToString())

    def load_model_by_meta(self, model_meta_folder):
        checkpoint = tf.train.get_checkpoint_state(
            model_meta_folder).model_checkpoint_path
        saver = tf.train.import_meta_graph(
            checkpoint + '.meta', clear_devices=True)
        saver.restore(self.sess, checkpoint)
        yolo_model = tf.import_graph_def(self.sess.graph_def, name='')

        print('{} model, anchors, and classes loaded.'.format(model_meta_folder))
      

    def write_meta(self, meta_output_folder, meta_output_file_name):
        saver = tf.train.Saver()
        saver.save(self.sess, os.path.join(
            meta_output_folder, meta_output_file_name+".ckpt"))
        tf.train.write_graph(self.sess.graph_def,
                             meta_output_folder, meta_output_file_name+'.pb')

    def get_nodes(self):
        #num_anchors = len(self.anchors)
        # is_tiny_version = num_anchors==6 # default setting
        #self.input_0 = self.sess.graph.get_tensor_by_name("return_box_shape:0")
        self.get_output={}
        self.get_input={}
        self.get_input["input_1"] = self.sess.graph.get_tensor_by_name("input_image:0")
        self.get_output["boxes"] = self.sess.graph.get_tensor_by_name("output_boxes:0")
        self.get_output["scores"] = self.sess.graph.get_tensor_by_name("output_scores:0")
        self.get_output["classes"] = self.sess.graph.get_tensor_by_name("output_classes:0")
        self.get_output["num"] = self.sess.graph.get_tensor_by_name("output_num:0")
    def load_image_into_numpy_array(self,image):
        (im_width, im_height) = image.size
        return np.array(image, dtype='float32').astype(np.uint8)
        #return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
    def detect(self, image, force_image_resize):
        image_data = pil_image_resize(force_image_resize, image)
        image_data=self.load_image_into_numpy_array(image_data)
        print("resize %s to %s" %
              ((image.size[1], image.size[0]), force_image_resize))
        
        start = timer()
        output_dict = self.sess.run(
            self.get_output,
            feed_dict={
                #self.input_0: [image.size[1], image.size[0]],
                self.get_input["input_1"]: image_data
            })
        #print(out_boxes, out_scores, out_classes)
        end = timer()
        print("detect time %s s" % (end - start))
        print(output_dict)
        output_dict["boxes"]=self.padding_boxes_reversize(output_dict["boxes"],force_image_resize,image.size)
        return output_dict
    def padding_boxes_reversize(self,boxes,in_shape,out_shape):
        long_side = max( out_shape)
        
        w_scale=long_side/in_shape[1]
        h_scale=long_side/in_shape[0]
        w_offset=(long_side-out_shape[0])/2.
        h_offset=(long_side-out_shape[1])/2.

        for box in boxes:
            
            box[0] = box[0]*h_scale*in_shape[1] -h_offset
            box[1] = box[1]*w_scale*in_shape[0] -w_offset
            box[2] = box[2]*h_scale*in_shape[1] -h_offset
            box[3] = box[3]*w_scale*in_shape[0] -w_offset
     
        return boxes.astype('int32')
def get_class(classes_path_raw):
        classes_path = os.path.expanduser(classes_path_raw)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names           

def generate_colors(class_names):
        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(class_names), 1., 1.)
                      for x in range(len(class_names))]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        # Shuffle colors to decorrelate adjacent classes.
        np.random.shuffle(colors)
        np.random.seed(None)  # Reset seed to default.
        return colors

def draw(image,class_names,colors, draw_score_threshold, out_boxes, out_scores, out_classes):

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):

            predicted_class = class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            if out_scores[i] >= draw_score_threshold:
                for i in range(thickness):
                    draw.rectangle(
                        [left + i, top + i, right - i, bottom - i],
                        outline=colors[c])
                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=colors[c])
                draw.text(text_origin, label, fill=(0, 0, 0), font=font)
                del draw
        return image


def tf_image_resize(target_size, image):
    boxed_image = tf_letterbox_image(target_size, image)
    return boxed_image


def tf_letterbox_image(size, image):
    '''resize image with unchanged aspect ratio using padding'''
    new_image = tf.image.resize_image_with_pad(
        image,
        target_height=size[1],
        target_width=size[0],
        method=ResizeMethod.BICUBIC
    )
    
    return new_image


def pil_image_resize(target_size, image):
    if target_size != (None, None):  # (height,width)
        assert target_size[0] % 32 == 0, 'Multiples of 32 required'
        assert target_size[1] % 32 == 0, 'Multiples of 32 required'
        new_image = utils.letterbox_image(image, tuple(reversed(target_size)))
    else:
        new_image_size = (image.width - (image.width % 32),
                          image.height - (image.height % 32))
        new_image = utils.letterbox_image(image, new_image_size)
    return new_image
def cv2_letterbox_image(img_path, size):
    '''resize image with unchanged aspect ratio using padding'''
    
    im = cv2.imread(img_path)
    old_size = im.shape[:2] # old_size is in (height, width) format
    ratio_w = float(size[1])/old_size[1]
    ratio_h = float(size[0])/old_size[0]
    ratio=min(ratio_h,ratio_w)
    new_size = tuple([int(x*ratio) for x in old_size])
    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0]))
    delta_w = size[1] - new_size[1]
    delta_h = size[0] - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    color = [0, 0, 0]
    new_image = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
    value=color)
    return new_image

def detect_video(yolo, video_path,class_path, draw_score_threshold, force_image_resize, output_path=""):
    vid = cv2.VideoCapture(video_path)
    class_names =get_class(class_path)
    colors=generate_colors(class_names)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(
            video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        output_dict = yolo.detect(
            image, force_image_resize)
        out_boxes=output_dict["boxes"]
        out_scores=output_dict["scores"]
        out_classes=output_dict["classes"]
        image = draw(
            image,class_names,colors, draw_score_threshold, out_boxes, out_scores, out_classes)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def detect_image(yolo, img_path,class_path, draw_score_threshold, force_image_resize):
    image = Image.open(img_path)
    output_dict = yolo.detect(
        image, force_image_resize)
    out_boxes=output_dict["boxes"]
    out_scores=output_dict["scores"]
    out_classes=output_dict["classes"]
    class_names =get_class(class_path)
    colors=generate_colors(class_names)
    image = draw(image,class_names,colors, draw_score_threshold,
                            out_boxes, out_scores, out_classes)
    image.show()


if __name__ == '__main__':
    # loading model from:
    # 0: h5
    # 1: freezed unity interface pb
    # 2: unity interface meta
    # 3: blider & h5 weights
    model_load_from = 0
    # args
    MODEL_h5_path = 'model_data/yolo.h5'
    MODEL_pb_path = 'model_data/freezed_coco_yolo.pb'
    ANCHORS_path = 'model_data/yolo_anchors.txt'
    CLASSES_path = 'model_data/coco_classes.txt'
    CLASSES_num = 80
    MODEL_meta_folder = ""
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
    FORCE_image_resize = (416, 416)

    # keras h5 convert to freezed graph output:
    # 0: no action
    # 1: h5-->freezed pb
    # 2: h5-->meta
    do_output_freezed_unity_interface = 0
    # args
    OUTPUT_pb_path = "./model_data"
    OUTPUT_pb_file = "freezed_coco_yolo.pb"
    OUTPUT_meta_folder = ""
    OUTPUT_meta_file_name = ""

    

    K.clear_session()
    with K.get_session() as sess:
        yolo = YOLO(CLASSES_num, ANCHORS_path, sess)

        if model_load_from == 0:
            yolo.load_model_by_h5(
                MODEL_h5_path, MODEL_score_threshold, IOU_threshold, GPU_num)
        elif model_load_from == 1:
            yolo.load_model_by_pb(MODEL_pb_path)
        elif model_load_from == 2:
            yolo.load_model_by_meta(MODEL_meta_folder)
        elif model_load_from == 3:
            yolo.load_model_by_buider(MODEL_weight_h5_path)
        yolo.get_nodes()
        if model_load_from == 0:
            if do_output_freezed_unity_interface == 1:
                yolo.write_pb(OUTPUT_pb_path, OUTPUT_pb_file)
            elif do_output_freezed_unity_interface == 2:
                yolo.write_meta(OUTPUT_meta_folder, OUTPUT_meta_file_name)
        else:
            if do_output_freezed_unity_interface != 0:
                print("for output, model must loading from .h5")

        if do_detect == 1:
            detect_image(yolo, IMG_path,CLASSES_path, DRAW_score_threshold,
                         FORCE_image_resize)
        elif do_detect == 2:
            detect_video(yolo, VIDEO_path,CLASSES_path, DRAW_score_threshold,
                         FORCE_image_resize, OUTPUT_video)
