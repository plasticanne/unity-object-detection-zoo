dependents:
    python==3.5.6
    opencv>=3.1.0
    keras==2.2.0
    tensorflow==1.9.0
    protobuf
    numpy
    pandas
    
install
protoc object_detection/protos/*.proto --python_out=.

references:
    [TF-Object Detection](https://github.com/tensorflow/models/tree/master/research/object_detection)
    [keras-yolo3](https://github.com/qqwweee/keras-yolo3)
    [TF-slim](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim)
    [TF-detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
    [darknet-YOLOv3](https://pjreddie.com/darknet/yolo/)