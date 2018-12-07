# -*- coding: utf-8 -*-
import os
from keras import backend as K
from keras.layers import Input
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from tensorflow.python.tools import optimize_for_inference_lib
import tensorflow as tf
import numpy as np
from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from tensorflow.python.tools import freeze_graph


class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolo.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "num_classes": 80,
        "output_dir":"model_data",
        "output_name":"yolo_coco_graph"
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()


    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = self.num_classes
        is_tiny_version = num_anchors==6 # default setting
   
        self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
        self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        print('{} model, anchors, and classes loaded.'.format(model_path))
           
        output_node_names = [node.op.name for node in self.yolo_model.outputs]
        input_node_names = [node.op.name for node in self.yolo_model.inputs]
        print("output_node_names: "+str(output_node_names))
        print("input_node_names: "+str(input_node_names))
        
        # save model and checkpoint
        saver = tf.train.Saver()
        save_path=saver.save(self.sess,os.path.join(self.output_dir,self.output_name+".ckpt"))
        tf.train.write_graph(self.sess.graph_def, self.output_dir, self.output_name+'.pb')
        print("Model saved in path: %s" % save_path)
        
        # save model and checkpoint
        constant_graph = graph_util.convert_variables_to_constants(self.sess, self.sess.graph_def, output_node_names)      
        optimize_Graph = optimize_for_inference_lib.optimize_for_inference(
              constant_graph,
              input_node_names, # an array of the input node(s)
              output_node_names, # an array of output nodes
              tf.float32.as_datatype_enum)
        optimize_for_inference_lib.ensure_graph_is_valid(optimize_Graph)

        with tf.gfile.GFile(os.path.join(self.output_dir,"freezed_"+self.output_name+".pb"), "wb") as f:
            f.write(optimize_Graph.SerializeToString())

        
        #graph_io.write_graph(optimize_Graph, self.output_dir, "freezed_"+self.output_name+".bytes", as_text=False)
        #graph_io.write_graph(constant_graph, outputdir, output+".prototxt", as_text=True)
        print('saved the freezed graph (ready for inference) at: ', self.output_name)
    def close_session(self):
        self.sess.close()


if __name__ == '__main__':
    K.clear_session() 
    yolo=YOLO()
    yolo.generate()
    yolo.close_session()



