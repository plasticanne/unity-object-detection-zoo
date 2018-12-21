import os
import sys
import cv2
import numpy as np
import six.moves.urllib as urllib
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from sqlalchemy.sql.sqltypes import BOOLEAN
from timeit import default_timer as timer
import tarfile
from collections import defaultdict
from distutils.version import StrictVersion
from io import StringIO
from object_detection.utils import label_map_util, ops as utils_ops, visualization_utils as vis_util
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from tensorflow.python.tools import optimize_for_inference_lib



if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
    raise ImportError(
        'Please upgrade your TensorFlow installation to v1.9.* or later!')





def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)



def load_model_by_tf_interface(pb_path,num_classes,score_threshold):

    input_1 = tf.placeholder(shape=[None, None, 3], name="input_image", dtype=tf.uint8)
    new_img_dims = tf.expand_dims(input_1, 0)
    # Load a (frozen) Tensorflow model into memory
    
    with tf.gfile.GFile(pb_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='', input_map={"image_tensor:0": new_img_dims})
    _generate_graph(score_threshold,num_classes)

def _generate_graph(score_threshold,num_classes):
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
    output_nodes['num'] = tf.identity(num_, name="output_num")
    output_nodes['classes'] = tf.identity(classes_, name="output_classes") 
    output_nodes['boxes'] = tf.identity(boxes_, name="output_boxes") 
    output_nodes['scores'] =tf.identity(scores_, name="output_scores") 

def load_model_by_unity_interface(pb_path,num_classes):
    with tf.gfile.GFile(pb_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')


    # Run inference
def get_nodes(sess):
    get_output={}
    get_input={}
    get_input["input_1"] = sess.graph.get_tensor_by_name("input_image:0")
    get_output["boxes"] = sess.graph.get_tensor_by_name("output_boxes:0")
    get_output["scores"] = sess.graph.get_tensor_by_name("output_scores:0")
    get_output["classes"] = sess.graph.get_tensor_by_name("output_classes:0")
    get_output["num"] = sess.graph.get_tensor_by_name("output_num:0")
    return get_input,get_output
def boxes_reversize(boxes,in_shape,out_shape):
    for box in boxes:
        box[0] = box[0]*in_shape[1]
        box[1] = box[1]*in_shape[0]
        box[2] = box[2]*in_shape[1]
        box[3] = box[3]*in_shape[0]
def detect(sess,image,get_input,get_output,force_image_resize):

    image_data = load_image_into_numpy_array(image)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    start = timer()
    output_dict = sess.run(get_output,
                           feed_dict={get_input["input_1"]: image_data})
    end = timer()
    print("detect time %s s" % (end - start))
    boxes_reversize(output_dict["boxes"],image.size,image.size)

    print(output_dict)
    return output_dict

def write_pb( sess,output_pb_dir, output_pb_file,get_input,get_output):
        
        input_nodes = [get_input["input_1"].name.split(":")[0]]
        output_nodes = [get_output["boxes"].name.split(":")[0], get_output["scores"].name.split(":")[
            0], get_output["classes"].name.split(":")[0],get_output["num"].name.split(":")[0]]
        print("input nodes:", input_nodes)
        print("output nodes:", output_nodes)
        constant_graph = graph_util.convert_variables_to_constants(
            sess, sess.graph.as_graph_def(), output_nodes)
        optimize_Graph = optimize_for_inference_lib.optimize_for_inference(
            constant_graph,
            input_nodes,  # an array of the input node(s)
            output_nodes,  # an array of output nodes
            tf.float32.as_datatype_enum)
        optimize_for_inference_lib.ensure_graph_is_valid(optimize_Graph)
        with tf.gfile.GFile(os.path.join(output_pb_dir, output_pb_file), "wb") as f:
            f.write(constant_graph.SerializeToString())




def generate_colors(class_names):
    import colorsys
    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x / len(class_names), 1., 1.)
                      for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),colors))
    np.random.seed(10101)  # Fixed seed for consistent colors across runs.
    # Shuffle colors to decorrelate adjacent classes.
    np.random.shuffle(colors)
    np.random.seed(None)  # Reset seed to default.
    return colors
def get_class(classes_path_raw):
        return label_map_util.create_category_index_from_labelmap(classes_path_raw, use_display_name=True)
def draw( image,class_names,colors, draw_score_threshold, out_boxes, out_scores, out_classes):
        
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
def detect_image(sess,img_path,class_path,get_input,get_output,force_image_resize):
    image = Image.open(img_path)
    output_dict = detect(sess,image,get_input,get_output,force_image_resize)
    out_boxes=output_dict["boxes"]
    out_scores=output_dict["scores"]
    out_classes=output_dict["classes"]
    class_names =get_class(class_path)
    colors=generate_colors(class_names)
    image = draw(image,class_names,colors, 0.5,out_boxes, out_scores, out_classes)
    image.show()
def detect_video(sess, video_path,class_path,get_input,get_output, draw_score_threshold, force_image_resize, output_path=""):
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
        output_dict = detect(sess,
            image,get_input,get_output, force_image_resize)
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






if __name__ == '__main__':
    # loading model from:
    # 0: freezed tf interface pb
    # 1: freezed unity interface pb
    model_load_from = 1
    # args
    MODEL_tf_path = 'model_data/ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb'
    MODEL_unity_path = 'model_data/ssdlite_mobilenet_v2_coco_2018_05_09/frozen_unity_inference_graph.pb'
    CLASSES_path = 'object_detection/data/mscoco_label_map.pbtxt'
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
    FORCE_image_resize = (416, 416)

    # interface convert output:
    # 0: no action
    # 1: tf-->unity freezed interface pb
   
    do_output_freezed_unity_interface = 0
    # args
    OUTPUT_pb_path = "model_data/ssdlite_mobilenet_v2_coco_2018_05_09"
    OUTPUT_pb_file = "frozen_unity_inference_graph.pb"





    detection_graph = tf.Graph()
    with detection_graph.as_default():
        if model_load_from == 0:
            load_model_by_tf_interface(MODEL_tf_path,CLASSES_num,MODEL_score_threshold)
        elif model_load_from == 1:
            load_model_by_unity_interface(MODEL_unity_path,CLASSES_num)
        with tf.Session() as sess:
            get_input,get_output=get_nodes(sess)
            if model_load_from == 0:
                if do_output_freezed_unity_interface == 1:
                    write_pb( sess,OUTPUT_pb_path,OUTPUT_pb_file,get_input,get_output)
            else:
                if do_output_freezed_unity_interface != 0:
                    print("for output, model must loading from tf interface")
            if do_detect == 1:
                detect_image(sess,IMG_path,CLASSES_path,get_input,get_output,FORCE_image_resize)
            elif do_detect == 2:
                detect_video(sess,IMG_path,CLASSES_path,get_input,get_output,FORCE_image_resize,OUTPUT_video)




