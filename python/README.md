# dependents
    python==3.5.6
    opencv>=3.1.0
    keras==2.2.0
    tensorflow==1.9.0
    protobuf
    numpy
    pandas
    tensorflow slim (included)
    tensorflow object_detection (included)
# install
```
protoc object_detection/protos/*.proto --python_out=.
```

# use orginal model
### *For Yolo V3*
- [download](https://github.com/tensorflow/models/tree/master/research/object_detection) the .cfg .weight
- convert the .cfg .weight to .h5
```
python yolo_convert_to_h5.py yolov3.cfg yolov3.weights model_data/yolov3.h5
```
### *For TF Model Zoo*
- [download](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
- "frozen_inference_graph.pb" is what we want
# do detection
### *For Yolo V3*
- edit yolo_detect.py
```
    # loading model from:
    model_load_from = 0
    # 0: h5
    MODEL_h5_path = 'model_data/yolov3.h5'
    MODEL_score_threshold = 0.1 # classify score threshold, value will be fixed to output freezed
    IOU_threshold = 0.1  # yolo iou box filter, value will be fixed to output freezed
    GPU_num = 1  # video cards count , cpu version or gpu version with counts will fixed after convert to pb
    # 1: freezed unity interface pb
    MODEL_pb_path = '' 
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
    VIDEO_path = ''
    OUTPUT_video = ""
    # args   
    DRAW_score_threshold = 0.5  # score filter for draw boxes
    FORCE_image_resize = (416, 416) # (height,width) 'Multiples of 32 required' , resize input to model

    # keras h5 convert to freezed graph output:
    do_output_freezed_unity_interface = 0
    # 0: no action
    # 1: h5-->unity_interface freezed pb
    OUTPUT_pb_path = ""
    OUTPUT_pb_file = ""
```
```
python yolo_detect.py
```
### *For TF Model Zoo*
- edit tf_zoo_detect.py
```
    # loading model from:
    model_load_from = 0
    # 0: freezed tf interface pb
    MODEL_tf_path = 'ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb'
    # classify score threshold, value will be fixed to output freezed
    MODEL_score_threshold = 0.1
    # 1: freezed unity interface pb
    MODEL_unity_path = ''
    #MODEL_unity_path = ''
    
    # args
    CLASSES_path = 'object_detection/data/mscoco_label_map.pbtxt'
    #CLASSES_path = 'model_data/raccoon_labels_map.pbtxt'
    #CLASSES_path='model_data/coco_classes90.json'
    CLASSES_num=1
    
    # doing detection:
    do_detect = 1
    # 0: no action
    # 1: img
    IMG_path = 'demo/boys.jpg'
    # 2: video
    VIDEO_path = ''
    OUTPUT_video = ""
    # args  
    DRAW_score_threshold = 0.5  # score filter for draw boxes
    FORCE_image_resize = (300, 300) # (height,width) 

    # interface convert output:
    do_output_freezed_unity_interface = 0
    # 0: no action
    # 1: tf-->unity freezed interface pb
    OUTPUT_pb_path = ""
    OUTPUT_pb_file = ""
```
```
python tf_zoo_detect.py
```
# prepare dataset
For example ,use  [raccoon dataset](https://github.com/datitran/raccoon_dataset), the structure is same to the PASCAL VOC dataset 
```
python tool_tfrecord.py --in_formet='xml' --input_annotations='dataset/raccoon/annotations' --input_image='dataset/raccoon/images' --input_classes='model_data/raccoon_labels_map.pbtxt' --output='dataset/raccoon/raccoon.record'
```
note yolo use the labels "index" and tf zoo use the labels "id" for training. It's is different key in tf-record.
# retrain model with own dataset
### *For Yolo V3*
- edit yolo_kmeans.py ,for get anchors.txt 
```
    # data from
    data_from=0
    # 0: tf-records
    RECORDS = ['dataset/raccoon/raccoon.record']
    # 1: orginal txt file
    ANNOTATIONS = ""

    # args
    LEN_anchors = 9 #normal is 9, tiny is 6
    OUTPUT_anchors='model_data/raccoon_anchors.txt'
```
```
python yolo_kmeans.py
```
- edit the cfg file and convert new .h5

[more details](https://github.com/thtrieu/darkflow#training-on-your-own-dataset)

- edit yolo_train.py
```
    # dataset from
    dataset_from=0
    # 0: tf-records
    RECORDS = ['dataset/raccoon/raccoon.record']
    RECORDS_test = ['dataset/raccoon/raccoon.record']
    # 1: orginal txt file 
    ANNOTATIONS =""
    ANNOTATIONS_test =""

    # args
    WEIGHT_from='dataset/raccoon/raccoon.h5'
    OUTPUT_train_dir = 'logs/raccoon'
    NUM_classes=1
    ANCHORS_path = 'model_data/raccoon_anchors.txt'
    INPUT_shape =(416,416) # multiple of 32, (height,width)
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
```
```
python yolo_train.py
```
### *For TF Model Zoo*
- edit the config

[more details](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md)
- edit tf_zoo_train.py
```
    FLAGS.checkpoint_dir="logs/raccoon"  #output logs
    FLAGS.pipeline_config_path="dataset/raccoon/ssdlite_mobilenet_v2_raccoon.config" 
```
```
python tf_zoo_train.py
```
- edit tf_zoo_convert_to_tf_interface.py for convert model to freezed tf interface pb
```
    FLAGS.trained_checkpoint_prefix = "logs/raccoon/model.ckpt-20000" #last one
    FLAGS.pipeline_config_path = "logs/raccoon/pipeline.config" 
    FLAGS.output_directory = "logs/raccoon/saved_model"
```
```
python tf_zoo_convert_to_tf_interface.py
```

# convert .pb for unity
- convert coco_labels_map.pbtxt to json
```
python tool_classes.py --in_format='label_map' --input='model_data/coco_labels_map.pbtxt' --out_format='json' --output='model_data/coco_labels.json'
```
### *For Yolo V3*
- edit yolo_detect.py
```
    # loading model from:
    model_load_from = 0
    ...
    # doing detection:
    do_detect = 0
    ...
    # keras h5 convert to freezed graph output:
    do_output_freezed_unity_interface = 1
    # 0: no action
    # 1: h5-->unity_interface freezed pb
    OUTPUT_pb_path = "./output"
    OUTPUT_pb_file = "freezed_coco_yolo.pb"
```
```
python yolo_detect.py
```
### *For TF Model Zoo*
- edit tf_zoo_detect.py
```
    # loading model from:
    model_load_from = 0
    ...    
    # doing detection:
    do_detect = 0
    ...
    # interface convert output:
    do_output_freezed_unity_interface = 1
    # 0: no action
    # 1: tf-->unity freezed interface pb
    OUTPUT_pb_path = "./output"
    OUTPUT_pb_file = "freezed_coco_tf_zoo.pb"
```
```
python tf_zoo_detect.py
```

# references
[TF-Object Detection](https://github.com/tensorflow/models/tree/master/research/object_detection)

[keras-yolo3](https://github.com/qqwweee/keras-yolo3)

[TF-sim](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim)

[TF-detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

[darknet-YOLOv3](https://pjreddie.com/darknet/yolo/)