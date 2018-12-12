
import numpy as np
import tensorflow as tf
from keras import backend as K
from yolo3.model import yolo_head
def yolo_correct_boxes(box_xy, box_wh):
    '''Get corrected boxes'''
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    #input_shape = K.cast(input_shape, K.dtype(box_yx))
   
    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    #boxes *= K.concatenate([input_shape, input_shape])
    return boxes

def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape):
    '''Process Conv layer output'''
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats,
        anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores

def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5):
    """
    Evaluate YOLO model on given input and return filtered boxes.
    :param yolo_outputs:
    :param anchors: [9,2]
    :param num_classes:
    :param image_shape: see yolo_boxes_and_scores()
    :param max_boxes: a scalar integer who present the maximum number of boxes to be selected by non max suppression
    :param score_threshold: score_threshold=.6
    :param iou_threshold: iou_threshold=.5
    :return:
    """
   
    num_layers = len(yolo_outputs)
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]] # default setting
    # shape None*13*13*3
    # input_shape = tf.shape(yolo_outputs[0])[1:3] * 32  # scale1 13*32=416 [416,416]
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []
    for l in range(num_layers): #tiny for 2 or yolo for 3
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
                                                    anchors[anchor_mask[l]],
                                                    num_classes,input_shape
                                                    )
        boxes.append(_boxes)# list(3 array): [3, None*13*13*3, 4]
        box_scores.append(_box_scores)# list(3 array): [3, None*13*13*3, 80]
    boxes = K.concatenate(boxes, axis=0)  # [3 *None*13*13*3, 4]
    box_scores = K.concatenate(box_scores, axis=0) # [3 *None*13*13*3, 80]

    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # TODO: use keras backend instead of tf.
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0) # [N, 4] with N: number of objects
    scores_ = K.concatenate(scores_, axis=0) # [N,]
    classes_ = K.concatenate(classes_, axis=0) # [N,]
    num_= tf.shape(classes_)
    return boxes_, scores_, classes_,num_