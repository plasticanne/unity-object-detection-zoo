
"""
Retrain the YOLO model for your own dataset.
"""
import os
import cv2
import numpy as np
import keras.backend as K
import tensorflow as tf
from PIL import Image
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data
from utils import pil_image_resize

# dataset from
dataset_from=0
# 0: tf-records
RECORDS = ['dataset/raccoon/a.record']
# 1: orginal txt file 
ANNOTATIONS ="model_data/raccoon_annotations.txt"

# args
WEIGHT_from='model_data/raccoon.h5'
OUTPUT_train_dir = 'logs/777'
NUM_classes=1
ANCHORS_path = 'model_data/raccoon_anchors.txt'
INPUT_shape = (64,64) # multiple of 32, (height,width)
BATCH_size = 2
VALID_ratio = 0.1

# Train with frozen layers first, to get a stable loss.
# Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
PRE_epochs=50
PRE_initial_epoch=0
# Unfreeze and continue training, to fine-tune.
# Train longer if the result is not good.
EPOCHS=500
INITIAL_epoch=50 





def _main():
        
    anchors = get_anchors(ANCHORS_path)
    is_tiny_version = len(anchors)==6 # default setting
    
    if is_tiny_version:
        model = create_tiny_model(INPUT_shape , anchors, NUM_classes,
            freeze_body=2)
    else:
        model = create_model(INPUT_shape , anchors, NUM_classes,
            freeze_body=2) # make sure you know what you freeze
            
    if not os.path.isdir(OUTPUT_train_dir): os.mkdir(OUTPUT_train_dir)
    logging = TensorBoard(log_dir=os.path.join(OUTPUT_train_dir,'TensorBoard'))
    checkpoint = ModelCheckpoint(os.path.join(OUTPUT_train_dir , 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'),
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
        
    
    if dataset_from==1:
        with open(ANNOTATIONS) as f:
            lines = f.readlines()
        np.random.seed(10101)
        np.random.shuffle(lines)
        np.random.seed(None)
        len_valid = int(len(lines)*VALID_ratio)
        len_train = len(lines) - len_valid
        iter_train   = lines[:len_train]
        pre_iter_train   = lines[len_train:]
        iter_valid   = lines[:len_train]
        pre_iter_valid   = lines[len_train:]
        generator=data_generator_wrapper

    else:
        train_ratio=1.0-VALID_ratio
        dataset = tf.data.TFRecordDataset(RECORDS, 'rb').map(parse_exmp)
        dataset=dataset.map(lambda x: split_train_valid(x, train_rate=train_ratio))
        dataset_train = dataset.filter(lambda x: filter_per_split(x, train=True))
        len_train=len_dataset(dataset_train)
        pre_dataset_train=dataset_train.repeat(PRE_epochs).shuffle(len_train*PRE_epochs).batch(1)
        dataset_train=dataset_train.repeat(EPOCHS).shuffle(len_train*EPOCHS).batch(1)

        dataset_valid = dataset.filter(lambda x: filter_per_split(x, train=False))
        len_valid=len_dataset(dataset_valid)
        pre_dataset_valid =dataset_valid.repeat(PRE_epochs).shuffle(len_train*PRE_epochs).batch(1)
        dataset_valid =dataset_valid.repeat(EPOCHS).shuffle(len_train*EPOCHS).batch(1)
        iter_train   = dataset_train.make_one_shot_iterator().get_next()
        pre_iter_train   = pre_dataset_train.make_one_shot_iterator().get_next()
        iter_valid   = dataset_valid.make_one_shot_iterator().get_next()
        pre_iter_valid   = pre_dataset_valid.make_one_shot_iterator().get_next() 
        generator=data_prepared


    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if True:
        model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred})

        print('Train on {} samples, valid on {} samples, with batch size {}.'.format(len_train, len_valid, BATCH_size))
        write_board(K.get_session())
        model.fit_generator(
            generator(pre_iter_train, BATCH_size, INPUT_shape , anchors, NUM_classes),
            steps_per_epoch=max(1, len_train//BATCH_size),
            validation_data=generator(pre_iter_valid, BATCH_size, INPUT_shape , anchors, NUM_classes),
            validation_steps=max(1, len_valid//BATCH_size),
            epochs=PRE_epochs,
            initial_epoch=PRE_initial_epoch,
            callbacks=[logging, checkpoint])
        model.save_weights(os.path.join(OUTPUT_train_dir, 'trained_weights_stage_1.h5'))

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
        print('Unfreeze all of the layers.')

        print('Train on {} samples, valid on {} samples, with batch size {}.'.format(len_train, len_valid, BATCH_size))

        model.fit_generator(
            generator(iter_train, BATCH_size, INPUT_shape , anchors, NUM_classes),
            steps_per_epoch=max(1, len_train//BATCH_size),
            validation_data=generator(iter_valid, BATCH_size, INPUT_shape , anchors, NUM_classes),
            validation_steps=max(1, len_valid//BATCH_size),
            epochs=EPOCHS,
            initial_epoch=INITIAL_epoch,
            callbacks=[logging, checkpoint, 
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1), 
            EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1)])
        model.save_weights(os.path.join(OUTPUT_train_dir, 'trained_weights_final.h5'))

def write_board(sess):
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(os.path.join(OUTPUT_train_dir,'TensorBoard'), graph = sess.graph)


def len_dataset(dataset):
    iter=dataset.make_one_shot_iterator().get_next()
    with tf.Session() as sess:
        i=0
        try:
            while True:
                sess.run(iter)
                i+=1
        except tf.errors.OutOfRangeError:
            return i
def parse_exmp(serial_exmp):
       
        feats = tf.parse_single_example(serial_exmp, features={
            #'image/filename':tf.FixedLenFeature([], tf.string),
            'image/encoded':tf.FixedLenFeature([1], tf.string),
            'image/height':tf.FixedLenFeature([1], tf.int64),
            'image/width':tf.FixedLenFeature([1], tf.int64),
            'image/object/class/label':tf.VarLenFeature( tf.int64),
            'image/object/class/index':tf.VarLenFeature( tf.int64),
            'image/object/bbox/xmin':tf.VarLenFeature( tf.float32),
            'image/object/bbox/xmax':tf.VarLenFeature( tf.float32),
            'image/object/bbox/ymin':tf.VarLenFeature( tf.float32),
            'image/object/bbox/ymax':tf.VarLenFeature( tf.float32)
            })
       
        return feats
def split_train_valid(parsed_features, train_rate=0.8, seed=10101):
        """ Randomly classify samples into training or testing split """
        parsed_features['is_train'] = tf.gather(tf.random_uniform([1], seed=seed) < train_rate, 0)
        return parsed_features
def filter_per_split(parsed_features, train=True):
        """ Filter samples depending on their split """
        return parsed_features['is_train'] if train else ~parsed_features['is_train']

def data_prepared(iter, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    if  batch_size<=0: return None
    while True:
        parsed_features=K.get_session().run(iter)
        image_data = []
        box_data = []
        for b in range(batch_size):
            #if i==0:
            #    np.random.shuffle(annotation_lines)
            image, box = get_random_data2(parsed_features, input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            #i = (i+1) % n
        
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
    
        yield [image_data, *y_true], np.zeros(batch_size)
def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a
def get_random_data2(parsed_features,input_shape, random=True, max_boxes=20, jitter=.3, hue=.1, sat=1.5, val=1.5, proc_img=True):
    from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
    '''random preprocessing for real-time data augmentation'''
    #line = annotation_line.split()
    #image = Image.open(line[0])
    image=Image.fromarray(parsed_features["image/encoded"],'RGB')
    iw=int(parsed_features['image/width'][0])
    ih=int(parsed_features['image/height'][0])
    h, w = input_shape
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
    #print(box_data)
    return image_data, box_data

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2):
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
        model_body.load_weights(WEIGHT_from, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(WEIGHT_from))
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

def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
        num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(WEIGHT_from, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(WEIGHT_from))
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

def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
            
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)

if __name__ == '__main__':
    K.clear_session() 
    _main()
import gc; gc.collect()
