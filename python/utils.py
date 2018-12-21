import os

import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from tool_classes import read_label_map,get_class_item
from keras import backend as K
from object_detection.utils import label_map_util
from tensorflow.image import ResizeMethod
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from tensorflow.python.tools import optimize_for_inference_lib
from yolo3 import utils

import colorsys
from distutils.version import StrictVersion
from timeit import default_timer as timer


if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
    raise ImportError(
        'Please upgrade your TensorFlow installation to v1.9.* or later!')


def load_image_into_numpy_array(image):
    #(im_width, im_height) = image.size
    # return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
    return np.array(image, dtype='float32').astype(np.uint8)


def get_nodes(sess):
    get_output = {}
    get_input = {}
    get_input["input_1"] = sess.graph.get_tensor_by_name("input_image:0")
    get_output["boxes"] = sess.graph.get_tensor_by_name("output_boxes:0")
    get_output["scores"] = sess.graph.get_tensor_by_name("output_scores:0")
    get_output["classes"] = sess.graph.get_tensor_by_name("output_classes:0")
    get_output["num"] = sess.graph.get_tensor_by_name("output_num:0")
    return get_input, get_output


def boxes_reversize(boxes,in_shape,out_shape):
    for box in boxes:
        box[0] = box[0]*in_shape[1]
        box[1] = box[1]*in_shape[0]
        box[2] = box[2]*in_shape[1]
        box[3] = box[3]*in_shape[0]


def detect(sess, image, get_input, get_output):
    image = load_image_into_numpy_array(image)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    #print (image[64,64,:])
    start = timer()
    output_dict = sess.run(get_output,
                           feed_dict={get_input["input_1"]: image})
    end = timer()
    print("detect time %s s" % (end - start))
    print(output_dict)
    return output_dict


def padding_boxes_reversize(boxes, in_shape, out_shape):
    long_side = max(out_shape)

    w_scale = long_side/in_shape[1]
    h_scale = long_side/in_shape[0]
    w_offset = (long_side-out_shape[0])/2.
    h_offset = (long_side-out_shape[1])/2.

    for box in boxes:

        box[0] = box[0]*h_scale*in_shape[1] - h_offset
        box[1] = box[1]*w_scale*in_shape[0] - w_offset
        box[2] = box[2]*h_scale*in_shape[1] - h_offset
        box[3] = box[3]*w_scale*in_shape[0] - w_offset

    return boxes.astype('int32')


def write_unity_interface_frozen_pb(sess, output_pb_dir, output_pb_file, get_input, get_output):
    input_nodes = [get_input["input_1"].name.split(":")[0]]
    output_nodes = [get_output["boxes"].name.split(":")[0], get_output["scores"].name.split(":")[
        0], get_output["classes"].name.split(":")[0], get_output["num"].name.split(":")[0]]
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
        print("output unity interface frozen graph at "+os.path.join(output_pb_dir, output_pb_file))


def load_unity_interface_frozen_pb(pb_path):
    with tf.gfile.GFile(pb_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')


def load_unity_interface_meta(sess, model_meta_folder):
    checkpoint = tf.train.get_checkpoint_state(
        model_meta_folder).model_checkpoint_path
    saver = tf.train.import_meta_graph(
        checkpoint + '.meta', clear_devices=True)
    saver.restore(sess, checkpoint)
    yolo_model = tf.import_graph_def(sess.graph_def, name='')


def write_unity_interface_meta(sess, meta_output_folder, meta_output_file_name):
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(
        meta_output_folder, meta_output_file_name+".ckpt"))
    tf.train.write_graph(sess.graph_def,
                         meta_output_folder, meta_output_file_name+'.pb')


def generate_colors(class_num):
    import colorsys
    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x / class_num, 1., 1.)
                  for x in range(class_num)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    np.random.seed(10101)  # Fixed seed for consistent colors across runs.
    # Shuffle colors to decorrelate adjacent classes.
    np.random.shuffle(colors)
    np.random.seed(None)  # Reset seed to default.
    return colors


def get_class(classes_path_raw):
    # return label_map_util.create_category_index_from_labelmap(classes_path_raw, use_display_name=True)
    return read_label_map(classes_path_raw)


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


def check_multiples_32(target_size, image):
    if target_size != (None, None):  # (height,width)
        assert target_size[0] % 32 == 0, 'Multiples of 32 required'
        assert target_size[1] % 32 == 0, 'Multiples of 32 required'
        return target_size
    else:
        new_image_size = (image.width - (image.width % 32),
                          image.height - (image.height % 32))
        return new_image_size


def pil_image_resize(target_size, image):
    print("resize %s to %s" %
          ((image.size[1], image.size[0]), target_size))
    return utils.letterbox_image(image, tuple(reversed(target_size)))


def cv2_letterbox_image(im, size):
    '''resize image with unchanged aspect ratio using padding'''

    #im = cv2.imread(img_path)
    old_size = im.shape[:2]  # old_size is in (height, width) format
    ratio_w = float(size[1])/old_size[1]
    ratio_h = float(size[0])/old_size[0]
    ratio = min(ratio_h, ratio_w)
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


def _draw(image, class_names,class_names_type, colors, draw_score_threshold, out_boxes, out_scores, out_classes):

    font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                              size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300
    
    for i, c in enumerate(out_classes):
        
        index,predicted_class = get_class_item(class_names,c,class_names_type)
        box = out_boxes[i]
        score = out_scores[i]

        label = '{} {:.2f}'.format(predicted_class["name"], score)
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
                    outline=colors[c-1])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=colors[c-1])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw
    return image


def detect_image(sess, img_path,class_num, class_path,class_names_type, get_input, get_output, draw_score_threshold, force_image_resize, requiredshape32=False):
    image = Image.open(img_path)
    if requiredshape32:
        force_image_resize = check_multiples_32(force_image_resize, image)
    image_resize = pil_image_resize(force_image_resize, image)
    output_dict = detect(sess, image_resize, get_input, get_output)
    out_boxes = padding_boxes_reversize(
        output_dict["boxes"], force_image_resize, image.size)
    out_scores = output_dict["scores"]
    out_classes = output_dict["classes"]
    class_names = get_class(class_path)
    colors = generate_colors(class_num)
    image = _draw(image, class_names,class_names_type, colors, draw_score_threshold,
                  out_boxes, out_scores, out_classes)
    image.show()


def detect_video(sess, video_path,class_num, class_path,class_names_type, get_input, get_output, draw_score_threshold, force_image_resize, requiredshape32=False, output_path=""):
    vid = cv2.VideoCapture(video_path)
    class_names = get_class(class_path)
    colors = generate_colors(class_num)
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
        if requiredshape32:
            force_image_resize = check_multiples_32(force_image_resize, image)
        image_resize = pil_image_resize(force_image_resize, image)
        output_dict = detect(sess, image_resize, get_input, get_output)
        out_boxes = padding_boxes_reversize(
            output_dict["boxes"], force_image_resize, image.size)
        out_scores = output_dict["scores"]
        out_classes = output_dict["classes"]
        image = _draw(
            image, class_names,class_names_type, colors, draw_score_threshold, out_boxes, out_scores, out_classes)
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
