
import numpy as np
import tensorflow as tf
from keras import backend as K
from yolo3.model import yolo_head
from PIL import Image
import io
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from keras.layers import Input, Lambda
from keras.models import Model
from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
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
def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

def get_random_data_tf(parsed_features,input_shape, random=True, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True):
    '''random preprocessing for real-time data augmentation'''
    
    iw=int(parsed_features['image/width'][0])
    ih=int(parsed_features['image/height'][0])
    h, w = input_shape
    encoded_jpg_io = io.BytesIO(bytearray(parsed_features["image/encoded"][0][0] ))
    image = Image.open(encoded_jpg_io)
    #image = Image.fromarray(parsed_features["image/encoded"][:1],'RGB')
    #image.show()
    #box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
    
    box=np.vstack(
    (np.around(parsed_features["image/object/bbox/xmin"].values*parsed_features['image/width']),
    np.around(parsed_features["image/object/bbox/ymin"].values*parsed_features['image/height']),
    np.around(parsed_features["image/object/bbox/xmax"].values*parsed_features['image/width']),
    np.around(parsed_features["image/object/bbox/ymax"].values*parsed_features['image/height']),
    parsed_features["image/object/class/index"].values)).astype(np.int32)
    box=box.T
    
    if not random:
        # resize image padding
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        dx = (w-nw)//2
        dy = (h-nh)//2
        image_data=0
        if proc_img:
            image = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)/255.
        # correct boxes
        box_data = np.zeros((max_boxes,5))
        if len(box)>0:
            np.random.shuffle(box)
            if len(box)>max_boxes: box = box[:max_boxes]
            box[:, [0,2]] = box[:, [0,2]]*scale + dx
            box[:, [1,3]] = box[:, [1,3]]*scale + dy
            box_data[:len(box)] = box
        
        
        return image_data, box_data

    # resize image
    new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
    scale = rand(.25, 2)
    scale=0.5
    if new_ar < 1:
        nh = int(scale*h)
        nw = int(nh*new_ar)
    else:
        nw = int(scale*w)
        nh = int(nw/new_ar)
    image = image.resize((nw,nh), Image.BICUBIC)

    # place image
    dx = int(rand(0, w-nw))
    dy = int(rand(0, h-nh))
    new_image = Image.new('RGB', (w,h), (128,128,128))
    new_image.paste(image, (dx, dy))
    image = new_image
    # flip image or not
    flip = rand()<.5
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # distort image
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
    val = rand(1, val) if rand()<.5 else 1/rand(1, val)
    x = rgb_to_hsv(np.array(image)/255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0]>1] -= 1
    x[..., 0][x[..., 0]<0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x>1] = 1
    x[x<0] = 0
    image_data = hsv_to_rgb(x) # numpy array, 0 to 1
    
    # correct boxes
    box_data = np.zeros((max_boxes,5))
    
    if len(box)>0:
        np.random.shuffle(box)
        
        box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
        box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
        if flip: box[:, [0,2]] = w - box[:, [2,0]]
        box[:, 0:2][box[:, 0:2]<0] = 0
        box[:, 2][box[:, 2]>w] = w
        box[:, 3][box[:, 3]>h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w>1., box_h>1.)] # discard invalid box
        if len(box)>max_boxes: box = box[:max_boxes]
        box_data[:len(box)] = box
    
    return image_data, box_data

def create_model(weight_path,input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2):
    '''create the training model'''
    
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weight_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weight_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def create_tiny_model(weight_path,input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2):
    '''create the training model, for Tiny YOLOv3'''
    
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
        num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weight_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weight_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers)-2)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

def create_model_bottleneck(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/yolo_weights.h5'):
    '''create the training model'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    # get output of second last layers and create bottleneck model of it
    out1=model_body.layers[246].output
    out2=model_body.layers[247].output
    out3=model_body.layers[248].output
    bottleneck_model = Model([model_body.input, *y_true], [out1, out2, out3])

    # create last layer model of last layers from yolo model
    in0 = Input(shape=bottleneck_model.output[0].shape[1:].as_list()) 
    in1 = Input(shape=bottleneck_model.output[1].shape[1:].as_list())
    in2 = Input(shape=bottleneck_model.output[2].shape[1:].as_list())
    last_out0=model_body.layers[249](in0)
    last_out1=model_body.layers[250](in1)
    last_out2=model_body.layers[251](in2)
    model_last=Model(inputs=[in0, in1, in2], outputs=[last_out0, last_out1, last_out2])
    model_loss_last =Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_last.output, *y_true])
    last_layer_model = Model([in0,in1,in2, *y_true], model_loss_last)

    
    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model, bottleneck_model, last_layer_model