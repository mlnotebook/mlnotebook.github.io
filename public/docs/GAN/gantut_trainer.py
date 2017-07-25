#!/usr/bin/python

import os
import numpy as  np
import tensorflow as tf

from gantut_gan import DCGAN



#DEFINE THE FLAGS FOR RUNNING SCRIPT FROM THE TERMINAL
# ARG1 = NAME OF THE FLAG
# ARG2 = DEFAULT VALUE
# ARG3 = DESCRIPTION
flags = tf.app.flags
flags.DEFINE_integer("epoch", 20, "Number of epochs to train [20]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate for adam optimiser [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term for adam optimiser [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of training images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The batch-size (number of images to train at once) [64]")
flags.DEFINE_integer("image_size", 64, "The size of the images [n x n] [64]")
flags.DEFINE_string("dataset", "lfw-aligned-64", "Dataset directory.")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
FLAGS = flags.FLAGS

#CREATE SOME FOLDERS FOR THE DATA
if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)
    
# GET ALL OF THE OPTIONS FOR TENSORFLOW RUNTIME 
config = tf.ConfigProto(intra_op_parallelism_threads=8)


with tf.Session(config=config) as sess:
    #INITIALISE THE GAN BY CREATING A NEW INSTANCE OF THE DCGAN CLASS
    dcgan = DCGAN(sess, image_size=FLAGS.image_size, batch_size=FLAGS.batch_size,
                  is_crop=False, checkpoint_dir=FLAGS.checkpoint_dir)

    #TRAIN THE GAN
    dcgan.train(FLAGS)