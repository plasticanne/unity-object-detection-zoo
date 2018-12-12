"""
run object_detection/legacy/eval.py
"""
import sys
from os.path import abspath, dirname, join
import tensorflow as tf

sys.path.insert(0, join(abspath(dirname(__file__)), 'slim'))
from object_detection.legacy.eval import FLAGS,main

FLAGS.logtostderr=True
FLAGS.train_dir="logs/training-gpu/"
FLAGS.pipeline_config_path="logs/training-gpu/pipeline.config"
FLAGS.eval_dir="logs/training-gpu/TensorBoard"


if __name__ == '__main__':
  tf.app.run()
