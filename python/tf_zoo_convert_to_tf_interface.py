"""
run object_detection/export_inference_graph.py
"""

import sys
from os.path import abspath, dirname, join
sys.path.insert(0, join(abspath(dirname(__file__)), 'slim'))
import tensorflow as tf
from google.protobuf import text_format
from object_detection import exporter
from object_detection.protos import pipeline_pb2
from object_detection.export_inference_graph import FLAGS, main



FLAGS.input_type = "image_tensor"
FLAGS.trained_checkpoint_prefix = "logs/training-gpu/model.ckpt-20000"
FLAGS.pipeline_config_path = "logs/training-gpu/pipeline.config"
FLAGS.output_directory = "logs/training-gpu/saved_model"




def main(_):
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.gfile.GFile(FLAGS.pipeline_config_path, 'r') as f:
        text_format.Merge(f.read(), pipeline_config)
    text_format.Merge(FLAGS.config_override, pipeline_config)
    input_shape = [1, None, None, 3]
    exporter.export_inference_graph(
        FLAGS.input_type, pipeline_config, FLAGS.trained_checkpoint_prefix,
        FLAGS.output_directory, input_shape=input_shape,
        write_inference_graph=FLAGS.write_inference_graph)


if __name__ == '__main__':
    tf.app.run()
