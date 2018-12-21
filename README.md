This project is a simple integration of interface with [Yolo V3](https://pjreddie.com/darknet/yolo/) and [TF-Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).

And those models can be used in [Unity](https://unity3d.com/).


## Workflow

**[Python part](python)**
- Download orginal model.
- Retrain model or just use it.
- Convert the model to unity interface frozen model .pb file. 


**[Unity part](unity/object%20detection)**
- Rename the .pb file to .bytes.
- Enjoy it.