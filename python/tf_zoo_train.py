"""
run object_detection/legacy/train.py
"""

import sys
from os.path import abspath, dirname, join
import tensorflow as tf

sys.path.insert(0, join(abspath(dirname(__file__)), 'slim'))
from object_detection.legacy.train import FLAGS,main

FLAGS.logtostderr=True
FLAGS.checkpoint_dir="logs/training-gpu/"
FLAGS.pipeline_config_path="logs/ssdlite_mobilenet_v2_raccoon.config"



if __name__ == '__main__':
  tf.app.run()