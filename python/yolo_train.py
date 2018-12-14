
"""
Retrain the YOLO model for your own dataset.
"""

import os
import numpy as np
import keras.backend as K
import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from yolo3.model import preprocess_true_boxes
from yolo3.utils import get_random_data
from utils import pil_image_resize
from utils_y import get_random_data_tf,create_tiny_model,create_model
# dataset from
dataset_from=0
# 0: tf-records
RECORDS = ['dataset/raccoon/a.record']
RECORDS_test = ['dataset/raccoon/a.record']
# 1: orginal txt file 
ANNOTATIONS ="model_data/raccoon_annotations.txt"
ANNOTATIONS_test ="model_data/raccoon_annotations.txt"

# args
WEIGHT_from='model_data/raccoon.h5'
#WEIGHT_from='logs/777/trained_weights_final.h5'
OUTPUT_train_dir = 'logs/777'
NUM_classes=1
ANCHORS_path = 'model_data/raccoon_anchors.txt'
INPUT_shape =(64,64) # multiple of 32, (height,width)
BATCH_size = 2
VALID_ratio = 0.1

# Train with frozen layers first, to get a stable loss.
# Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
do_pre_train=True
PRE_epochs=50
# Unfreeze and continue training, to fine-tune.
# Train longer if the result is not good.
do_train=True
EPOCHS=500
# random test from dataset
do_test=True




def main():
    graph_train = tf.Graph()
    with graph_train.as_default():
        with tf.Session().as_default() as sess:
            anchors = get_anchors(ANCHORS_path)
            is_tiny_version = len(anchors)==6 # default setting
            
            if is_tiny_version:
                model = create_tiny_model(WEIGHT_from,INPUT_shape , anchors, NUM_classes,
                    freeze_body=2)
            else:
                model = create_model(WEIGHT_from,INPUT_shape , anchors, NUM_classes,
                    freeze_body=2) # make sure you know what you freeze
            if not os.path.isdir(OUTPUT_train_dir): os.mkdir(OUTPUT_train_dir)
        

            if dataset_from==1:
                with open(ANNOTATIONS) as f:
                    lines = f.readlines()
                np.random.seed(10101)
                np.random.shuffle(lines)
                np.random.seed(None)
                len_valid = int(len(lines)*VALID_ratio)
                len_train = len(lines) - len_valid
                iter_train   = lines[:len_train]
                iter_valid   = lines[len_train:]
                

            else:
                train_ratio=1.0-VALID_ratio
                dataset = tf.data.TFRecordDataset(RECORDS)
                dataset_train,len_train,dataset_valid,len_valid=split_by_sum(sess,dataset,train_ratio)
        
                dataset_train=dataset_train.map(parse_exmp).shuffle(len_train).batch(1).repeat()
                dataset_valid =dataset_valid.map(parse_exmp).shuffle(len_valid).batch(1).repeat()

                iter_train   = dataset_train.make_one_shot_iterator().get_next()
                iter_valid   = dataset_valid.make_one_shot_iterator().get_next()
                
                """for a in range(1):
                    print(sess.run(iter_valid)["image/filename"])"""
                """for i in generator(sess,iter_train, BATCH_size, INPUT_shape , anchors, NUM_classes):
                    print('x')"""
            # Train with frozen layers first, to get a stable loss.
            # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
            logging = TensorBoard(log_dir=os.path.join(OUTPUT_train_dir,'TensorBoard'))
            checkpoint = ModelCheckpoint(os.path.join(OUTPUT_train_dir , 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'),
                monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
            output_1=os.path.join(OUTPUT_train_dir, 'trained_weights_stage_1.h5')
            output_final=os.path.join(OUTPUT_train_dir, 'trained_weights_final.h5')
            if do_pre_train:
                model.compile(optimizer=Adam(lr=1e-3), loss={'yolo_loss': lambda y_true, y_pred: y_pred})# use custom yolo_loss Lambda layer.

                print('Train on {} samples, valid on {} samples, with batch size {}.'.format(len_train, len_valid, BATCH_size))
                model.fit_generator(
                    data_generator_wrapper(sess,iter_train, BATCH_size, INPUT_shape , anchors, NUM_classes),
                    steps_per_epoch=max(1, len_train//BATCH_size),
                    validation_data=data_generator_wrapper(sess,iter_valid, BATCH_size, INPUT_shape , anchors, NUM_classes),
                    validation_steps=max(1, len_valid//BATCH_size),
                    epochs=PRE_epochs,
                    initial_epoch=0,
                    callbacks=[logging, checkpoint])
                model.save_weights(output_1)
            
            # Unfreeze and continue training, to fine-tune.
            # Train longer if the result is not good.
            
            if do_train:
                for i in range(len(model.layers)):
                    model.layers[i].trainable = True
                model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
                print('Unfreeze all of the layers.')

                print('Train on {} samples, valid on {} samples, with batch size {}.'.format(len_train, len_valid, BATCH_size))

                model.fit_generator(
                    data_generator_wrapper(sess,iter_train, BATCH_size, INPUT_shape , anchors, NUM_classes),
                    steps_per_epoch=max(1, len_train//BATCH_size),
                    validation_data=data_generator_wrapper(sess,iter_valid, BATCH_size, INPUT_shape , anchors, NUM_classes),
                    validation_steps=max(1, len_valid//BATCH_size),
                    epochs=EPOCHS,
                    initial_epoch=PRE_epochs,
                    callbacks=[logging, checkpoint, 
                    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1), 
                    EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1)])
                model.save_weights(output_final)
                #model.save(os.path.join(OUTPUT_train_dir, 'trained_final.h5'))
    graph_test = tf.Graph()
    with graph_test.as_default():
        with tf.Session().as_default() as sess:      
            if do_test:
                if dataset_from==1:
                    with open(ANNOTATIONS_test) as f:
                        lines = f.readlines()
                    np.random.seed(10101)
                    np.random.shuffle(lines)
                    np.random.seed(None)
                    iter_dataset_test   = lines
                    
                else:
                    dataset_test = tf.data.TFRecordDataset(RECORDS_test).map(parse_exmp)
                    dataset_test=dataset_test.shuffle(len_dataset(sess,dataset_test)).batch(1).repeat()
                    iter_dataset_test   = dataset_test.make_one_shot_iterator().get_next()
                    
                if is_tiny_version:
                    model_test = create_tiny_model(output_final,INPUT_shape , anchors,NUM_classes,freeze_body=0) 
                else:
                    model_test = create_model(output_final,INPUT_shape , anchors, NUM_classes,freeze_body=0)  # make sure you know what you freeze              #model_test=model
                
                model_test.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
                eval_score=model_test.evaluate_generator(data_generator_wrapper(sess,iter_dataset_test, BATCH_size, INPUT_shape , anchors, NUM_classes),1)
                print("eval_score: %s"%(eval_score))
                pred_score=model_test.predict_generator(data_generator_wrapper(sess,iter_dataset_test, BATCH_size, INPUT_shape , anchors, NUM_classes),1)
                print("pred_score: %s"%(pred_score))
                
def write_board(sess):
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(os.path.join(OUTPUT_train_dir,'TensorBoard'), graph = sess.graph)


def len_dataset(sess,dataset):
        iter=dataset.make_one_shot_iterator().get_next()
        i=0
        try:
            while True:
                sess.run(iter)
                i+=1
        except tf.errors.OutOfRangeError:
            return i
def parse_exmp(serial_exmp):
        feats = tf.parse_single_example(serial_exmp, features={
            'image/filename':tf.FixedLenFeature([], tf.string),
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

def split_by_seed(sess,dataset,train_rate):
    def split_train_valid(parsed_features, train_rate=0.9, seed=10101):
        """ Randomly classify samples into training or testing split """
        parsed_features['is_train'] = tf.gather(tf.random_uniform([1], seed=seed) < train_rate, 0)
        return parsed_features
    def filter_per_split(parsed_features, train=True):
        """ Filter samples depending on their split """
        return parsed_features['is_train'] if train else ~parsed_features['is_train']
    dataset=dataset.map(lambda x: split_train_valid(x, train_rate=train_rate))
    dataset_train = dataset.filter(lambda x: filter_per_split(x, train=True))
    len_train=len_dataset(sess,dataset_train)
    dataset_valid = dataset.filter(lambda x: filter_per_split(x, train=False))
    len_valid=len_dataset(sess,dataset_valid)
    return dataset_train,len_train,dataset_valid,len_valid
def split_by_sum(sess,dataset,train_rate=0.9):
    
    len_full=len_dataset(sess,dataset)
    #len_full=1
    len_train=int(len_full*train_rate)
    #len_train=1
    len_valid=len_full-len_train
    #len_valid=1
    dataset = dataset.shuffle(len_full)
    dataset_train=dataset.take(len_train).cache()
    dataset_valid=dataset.skip(len_train).cache()

    return dataset_train,len_train,dataset_valid,len_valid

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def data_generator_tf(sess,iter, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    if  batch_size<=0: return None
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            parsed_features=sess.run(iter)
            #if i==0:
            #    np.random.shuffle(annotation_lines)
            image, box = get_random_data_tf(parsed_features, input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            #i = (i+1) % n
            #print(parsed_features["image/filename"])
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)


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

def data_generator_wrapper(sess, iter, batch_size, input_shape, anchors, num_classes):
    if batch_size<=0: return None
    if dataset_from ==0:
        return data_generator_tf(iter, batch_size, input_shape, anchors, num_classes)
    if dataset_from ==1:
        n = len(iter)
        if n==0 :return None
        return data_generator(iter, batch_size, input_shape, anchors, num_classes)

if __name__ == '__main__':
    main()
    

