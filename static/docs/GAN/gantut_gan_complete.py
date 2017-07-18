from __future__ import division
import os
import time
import math
import itertools
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

#IMPORT OUR IMAGE AND DATA FUNCTIONS
from gantut_datafuncs import *
from gantut_imgfuncs import *

#A LITTLE HOUSEKEEPING TO CATCH ERRORS IN THE DATA
SUPPORTED_EXTENSIONS = ["png", "jpg", "jpeg"]

""" Returns the list of all SUPPORTED image files in the directory
"""
def dataset_files(root):   
    return list(itertools.chain.from_iterable(
    glob(os.path.join(root, "*.{}".format(ext))) for ext in SUPPORTED_EXTENSIONS))
    
#DEFINE THE GAN CLASS
"""this class instantiates the GAN. It has methods:

build_model:    initialises the model variables and the graph for the GAN
generator:      the reverse CNN for the generator
discriminator:  the CNN for the discriminator
train:          trains the network (defines the optimisers etc.)
save:           saves the current configuration
load:           loads a saved configuration

RETURNS
- a DCGAN object with the above methods
"""
class DCGAN(object):    
    def __init__(self, sess, image_size=64, is_crop=False, batch_size=64, sample_size=64, z_dim=100,
                 gf_dim=64, df_dim=64, gfc_dim=1024, dfc_dim=1024, c_dim=3, checkpoint_dir=None, lam=0.1):

        """
        Args:
            sess: TensorFlow session
            image_size: the width (and height) of the images
            is_crop: whether to crop the images or leave them
            batch_size: size of the batch (to be specified before training)
            sample_size: number of z samples (should be equal to batch_size)
            z_dim: number of samples to take for ecach z
            gf_dim: dimension of gen filters in first conv layer
            df_dim: dimenstion of discrim filters in first conv layer
            gfc_dim: dimension of gen units for fully-connected layer
            dfc_gim: dimension of discrim units for fully-connected layer
            c_dim: dimension of image color
            checkpoint_dir: where to store the TensorFlow checkpoints
            lam: small constant weight for the sum of contextual and perceptual loss
        """
        
        #image_size must be power of 2 and 8+
        assert(image_size & (image_size - 1) == 0 and image_size >= 8)
        
        self.sess = sess
        self.is_crop = is_crop
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.image_shape = [image_size, image_size, c_dim]
        
        self.z_dim = z_dim
        self.gf_dim = gf_dim
        self.df_dim = df_dim        
        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim
        
        self.lam = lam
        self.c_dim = c_dim
        
        #batchnorm (from funcs.py)
        self.d_bns = [batch_norm(name='d_bn{}'.format(i,)) for i in range(4)]
        
        log_size = int(math.log(image_size) / math.log(2))
        self.g_bns = [batch_norm(name='g_bn{}'.format(i,)) for i in range(log_size)]
        
        self.checkpoint_dir = checkpoint_dir
        self.build_model()
        
        self.model_name="DCGAN.model"
        
    def build_model(self):
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.images = tf.placeholder(
            tf.float32, [None] + self.image_shape, name='real_images')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.z_sum = tf.summary.histogram("z", self.z)

        self.G = self.generator(self.z)
        self.D, self.D_logits = self.discriminator(self.images)

        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_)
        self.G_sum = tf.summary.image("G", self.G)

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits,
                                                    labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_,
                                                    labels=tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_,
                                                    labels=tf.ones_like(self.D_)))

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver(max_to_keep=1)      
        
    def generator(self, z):
        with tf.variable_scope("generator") as scope:
            self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*8*4*4, 'g_h0_lin', with_w=True)
        
            hs = [None]
            hs[0] = tf.reshape(self.z_, [-1, 4, 4, self.gf_dim * 8])
            hs[0] = tf.nn.relu(self.g_bns[0](hs[0], self.is_training))
            
            i=1             #iteration number
            depth_mul = 8   #depth decreases as spatial component increases
            size=8          #size increases as depth decreases
            
            while size < self.image_size:
                hs.append(None)
                name='g_h{}'.format(i)
                hs[i], _, _ = conv2d_transpose(hs[i-1], [self.batch_size, size, size, self.gf_dim*depth_mul],
                                                name=name, with_w=True)
                hs[i] = tf.nn.relu(self.g_bns[i](hs[i], self.is_training))
                
                i += 1
                depth_mul //= 2
                size *= 2
                
            hs.append(None)
            name = 'g_h{}'.format(i)
            hs[i], _, _ = conv2d_transpose(hs[i-1], [self.batch_size, size, size, 3], name=name, with_w=True)
            
            return tf.nn.tanh(hs[i])        
        
    def discriminator(self, image, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            
            h0 = lrelu(conv2d(image, self.df_dim, name='d_h00_conv'))
            h1 = lrelu(self.d_bns[0](conv2d(h0, self.df_dim*2, name='d_h1_conv'), self.is_training))
            h2 = lrelu(self.d_bns[1](conv2d(h1, self.df_dim*4, name='d_h2_conv'), self.is_training))
            h3 = lrelu(self.d_bns[2](conv2d(h2, self.df_dim*8, name='d_h3_conv'), self.is_training))
            h4 = linear(tf.reshape(h3, [-1, 8192]), 1, 'd_h4_lin')
            
            return tf.nn.sigmoid(h4), h4        
    
    def train(self, config):
        data = dataset_files(config.dataset)
        np.random.shuffle(data)
        assert(len(data) > 0)
        
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.g_loss, var_list=self.g_vars)
        
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()
            
            self.g_sum = tf.summary.merge([self.z_sum, self.d__sum, self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.summary.merge([self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)
        
        sample_z = np.random.uniform(-1, 1, size=(self.sample_size, self.z_dim))

        sample_files = data[0:self.sample_size]        
        sample = [get_image(sample_file, self.image_size, is_crop=self.is_crop) for sample_file in sample_files]
        sample_images = np.array(sample).astype(np.float32)
        
        counter = 1
        start_time = time.time()
        
        if self.load(self.checkpoint_dir):
            print(""" An existing model was found - delete the directory or specify a new one with --checkpoint_dir """)
        else:
            print(""" No model found - initializing a new one""")
        
        for epoch in xrange(config.epoch):
            data = dataset_files(config.dataset)
            batch_idxs = min(len(data), config.train_size) // self.batch_size
            
            for idx in xrange(0, batch_idxs):
                batch_files = data[idx*config.batch_size:(idx+1)*config.batch_size]
                batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop) for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)
                
                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)
                
                #update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum],
                                               feed_dict={self.images: batch_images, self.z: batch_z, self.is_training: True})
                self.writer.add_summary(summary_str, counter)
                
                #update G network
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={self.z: batch_z, self.is_training: True})
                self.writer.add_summary(summary_str, counter)
                
                #run g_optim twice to make sure that d_loss does not go to zero (not in the paper)
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={self.z: batch_z, self.is_training: True})
                self.writer.add_summary(summary_str, counter)
                
                errD_fake = self.d_loss_fake.eval({self.z: batch_z, self.is_training: False})
                errD_real = self.d_loss_real.eval({self.images: batch_images, self.is_training: False})
                errG = self.g_loss.eval({self.z: batch_z, self.is_training: False})
                
                counter += 1
                print("Epoch [{:2d}] [{:4d}/{:4d}] time: {:4.4f}, d_loss: {:.8f}".format(
                        epoch, idx, batch_idxs, time.time() - start_time, errD_fake + errD_real, errG))
                
                if np.mod(counter, 100) == 1:
                    samples, d_loss, g_loss = self.sess.run([self.G, self.d_loss, self.g_loss], 
                                                            feed_dict={self.z: sample_z, self.images: sample_images, self.is_training: False})
                    save_images(samples, [8,8], './samples/train_{:02d}-{:04d}.png'.format(epoch, idx))
                    print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))
                    
                if np.mod(counter, 500) == 2:
                    self.save(config.checkpoint_dir, counter)
        
    def save(self, checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            
        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name), global_step=step)
        
        
    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            return True
        else:
            return False
        
    
    