import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

 
#Hyperparameters
numK = 16               #number of kernels in each conv layer
sizeConvK = 3           #size of the kernels in each conv layer [n x n]
sizePoolK = 2           #size of the kernels in each pool layer [m x m]
inputSize = 28          #size of the input image
numChannels = 1         #number of channels to the input image grayscale=1, RGB=3
keepProb = 0.4      
learning_rate = 0.001

def doConv(inputs):
    convOut = tf.layers.conv2d(inputs=inputs, filters=numK, kernel_size=[sizeConvK, sizeConvK], \
        padding="same", activation=tf.nn.relu)    
    return convOut
    
def doPool(inputs):
    poolOut = tf.layers.max_pooling2d(inputs=inputs, pool_size=[sizePoolK, sizePoolK], strides=2)
    return poolOut

def convNet(inputs, labels, mode):
    #reshape the input from a vector to a 2D image
    input_layer = tf.reshape(inputs, [-1, inputSize, inputSize, numChannels])   
    
    #perform convolution and pooling
    conv1 = doConv(input_layer) 
    pool1 = doPool(conv1)      
    
    conv2 = doConv(pool1)
    pool2 = doPool(conv2)

    #flatted the result back to a vector for the FC layer
    flatPool = tf.reshape(pool2, [-1, 7 * 7 * numK])    
    dense = tf.layers.dense(inputs=flatPool, units=1024, activation=tf.nn.relu)

    #Uncomment these two lines to include batch normalisation and dropout
    #dense = tf.contrib.layers.batch_norm(dense, decay=0.99, is_training= mode==learn.ModeKeys.TRAIN)
    #dense = tf.layers.dropout(inputs=dense, rate=keepProb, training = mode==learn.ModeKeys.TRAIN)
    
    #Get the output in the form of one-hot labels with x units
    logits = tf.layers.dense(inputs=dense, units=10) 
    
    loss = None
    train_op = None
    #At the end of the network, check how well we did     
    if mode != learn.ModeKeys.INFER:
        #create one-hot tabels from the training-labels
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
        #check how close the output is to the training-labels
        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
    
    #After checking the loss, use it to train the network weights   
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(loss=loss, global_step=tf.contrib.framework.get_global_step(), \
            learning_rate=learning_rate, optimizer="SGD")
        
    predictions = {"classes": tf.argmax(input=logits, axis=1), "probabilities": tf.nn.softmax(logits, name="softmax_tensor")}
    
    return model_fn_lib.ModelFnOps(mode=mode, predictions=predictions, loss=loss, train_op=train_op)
        
def main(unused_argv):
    # Load training and eval data
    mnist = learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    
    mnistClassifier = learn.Estimator(model_fn=convNet,   model_dir="/tmp/mln_MNIST2")
    
    tensors2log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors2log, every_n_iter=100)
    
    mnistClassifier.fit(x=train_data, y=train_labels, batch_size=100, steps=1000, monitors=[logging_hook])

    metrics = {"accuracy": learn.MetricSpec(metric_fn=tf.metrics.accuracy, prediction_key="classes")}
    
    eval_results = mnistClassifier.evaluate(x=eval_data, y=eval_labels, metrics=metrics)
    print(eval_results)
   
   
if __name__ == "__main__":
    tf.app.run()