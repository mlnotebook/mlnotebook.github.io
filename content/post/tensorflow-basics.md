+++
date = "2017-07-03T09:44:24+01:00"
title = "Convolutional Neural Networks - TensorFlow (Basics)"
description = "Using TensorFlow to build a CNN"
topics = ["tutorial"]
tags = ["CNN", "tensorflow", "neural network"]
social=true
featured_image="/img/featCNN2.png"
+++

We've looked at the principles behind how a CNN works, but how do we actually implement this in Python? This tutorial will look at the basic idea behind Google's TensorFlow: an efficient way to build a CNN using purpose-build Python libraries.

<!--more-->

<div style="text-align:center;"><img width=30% title="TensorFlow" src="/img/CNN/TF_logo.png"></div>

<h2 id="intro">  Introduction </h2>

Building a CNN from scratch in Python is perfectly possible, but very memory intensive. It can also lead to very long pieces of code. Several libraries have been developed by the community to solve this problem by wrapping the most common parts of CNNs into special methods called from their own libraries. Theano, Keras and PyTorch are notable libraries being used today that are all opensource. However, since TensorFlow was released and Google announced their machine-learning-specific hardware, the Tensor Processing Unit (TPU), TensorFlow has quickly become a much-used tool in the field. If any applications being built today are intended for use on mobile devices, TensorFlow is the way to go as the mobile TPU in the upcoming Google phones will be able to perform inference from machine learning models in the User's hand. Of course, being a relative newcomer and updates still very much controlled by Google, TensorFlow may not have the huge body of support that has built up with Theano, say.

Nevertheless, TensorFlow is powerful and quick to setup so long as you know how: read on to find out. Much of this tutorial is based around the documentation provided by Google, but gives a lot more information that many be useful to less experienced users.

<h2 id="install"> Installation </h2>

TensorFlow is just another set of Python libraries distributed by Google via the website: <a href="https://www.tensorflow.org/install" title="TensorFlow Installation">https://www.tensorflow.org/install</a>. There's the option to install the version for use on GPUs but that's not necessary for this tutorial, we'll be using the MNIST dataset which is not too memory instensive.

Go ahead and install the TensorFlow libraries. I would say that even though they suggest using TF in a virtual environment, we will be coding up our CNN in a Python script so don't worry about that if you're not comfortable with it.

One of the most frustrating things you will find with TF is that much of the documentation on various websites is already out-of-date. Some of the commands have been re-written or renamed since the support was put in place. Even some of Google's own tutorials are now old and require tweaking. Currently, the code written here will work on all versions, but may throw some 'depreication' warnings.

<h2 id="structure"> TensorFlow Structure </h2>

The idea of 'flow' is central to TF's organisation. The actual CNN is written as a 'graph'. A graph is simply a list of the differnet layers in your network each with their own input and output. Whatever data we input at the top will 'flow' through the graph and output some values. The values we will also deal with using TensorFlow which will automatically take care of the updating of any internal weights via whatever optimization method and loss function we prefer.

The graph is called by some initial functions in the script that create the classifier, run the training and output whatever evlauation metrics we like.

Before writing any functions, lets import the necessary includes and tell TF to limit any program logging:

```python
import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
```
 We've included multiple TF lines to save on the typing later.
 
<h3 id="graph"> The Graph </h3>

Let's get straight to it and start to build our graph. We will keep it simple:

* 2 convolutional layers learning 16 filters (or kernels) of [3 x 3]
* 2 max-pooling layers that half the size of the image using [2 x 2] kernel
* A fully connected layer at the end.


```python
#Hyperparameters
numK = 16               #number of kernels in each conv layer
sizeConvK = 3           #size of the kernels in each conv layer [n x n]
sizePoolK = 2           #size of the kernels in each pool layer [m x m]
inputSize = 28          #size of the input image
numChannels = 1         #number of channels to the input image grayscale=1, RGB=3

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
```

So what's going on here? First we've defined some parameters for the CNN such as kernel sizes, the height of the input image (assuming it's square) and the number of channels for the image. The number of channels is `1` for both Black and White with intensity values of either 0 or 1, and grayscale images with intensities in the range [0 255]. Colour images have `3` channels, Red, Green and Blue.

You'll notice that we've barely used TF so far: we use it to reshape the data. This is important, when we run our script, TF will take our raw data and turn it into its own data type i.e. a `tensor`. That means our normal `numpy` operations won't work on them so we should use the in-built `tf.reshape` function which works in the same was as the one in numpy - it takes the input data and an output shape as arguments.

But why are we reshaping at all? Well, the data that is input into the network will be in the form of vectors. The image will have been saved along with lots of other images as single lines of a larger file. This is the case with the MNIST dataset and is common in machine learning. So we need to put it back into image-form so that we can perform convolutions.

"Where are those random 7s and the -1 from?"... good question. In this example, we are going to be using the MNIST dataset whose images are 28 x 28. If we put this through 2 pooling layers we will half (14 x 14) and half again (7 x 7) the width. Thus the layer needs to know what it is expecting the output to look like based upon the input which will be a 7 x 7 x `numK` tensor, one 7 x 7 for each kernel. Keep in mind that we will be running the network with more than one input image at a time, so in reality when we get to this stage, there will be `n` images here which all have 7 x 7 x `numK` values associated with them. The -1 simply tells TensorFlow to take *all* of these images and do the same to each. It's short hand for "do this for the whole batch".

There's also a `tf.layers.dense` method at the end here. This is one of TF's in-built layer types that is very handy. We just tell it what to take as input, how many units we want it to have and what non-linearity we would prefer at the end. Instead of typing this all separately, it's combined into a single line. Neat!

But what about the `conv` and `pool` layers? Well, to keep the code nice and tidy, I like to write the convolution and pooling layers in separate functions. This means that if I want to add more `conv` or `pool` layers, I can just write them in underneath the current ones and the code will still look clean (not that the functions are very long). Here they are:

```python
def doConv(inputs):
    convOut = tf.layers.conv2d(inputs=inputs, filters=numK, kernel_size=[sizeConvK, sizeConvK], \
    	padding="SAME", activation=tf.nn.relu)    
    return convOut
    
def doPool(inputs):
    poolOut = tf.layers.max_pooling2d(inputs=inputs, pool_size=[sizePoolK, sizePoolK], strides=2)
    return poolOut
```

Again, both the `conv` and `pool` layers are simple one-liners. They both take in some input data and need to know the size of the kernel you want them to use (which we defined earlier on). The `conv` layer needs to know how many `filters` to learn too. Alongside this, we need to take care of any mis-match between the image size and the size of the kernels to ensure that we're not changing the size of the image when we get the output. This is easily done in TF by setting the `padding` attribute to `"SAME"`. We've got our non-linearity at the end here too. We've hard-coded that the pooling layer will have `strides=2` and will therefore half in size at each pooling layer.

Now we have the main part of our network coded-up. But it wont do very much unless we ask TF to give us some outputs and compare them to some training data.

As the MNIST data is used for image-classification problems, we'll be trying to get the network to output probabilities that the image it is given belongs to a specific class i.e. a number 0-9. The MNIST dataset provides the numbers 0-9 which, if we provided this to the network, would start to output guesses of decimal values 0.143, 4.765, 8.112 or whatever. We need to change this data so that each class can have its own specific box which the network can assign a probability. We use the idea of 'one-hot' labels for this. For example, class 3 becomes [0 0 0 1 0 0 0 0 0 0] and class 9 becomes [0 0 0 0 0 0 0 0 0 1]. This way we're not asking the network to predict the number associated with each class but rather how likely is the test-image to be in this class.

TF has a very handy function for changing class labels into 'one-hot' labels. Let's continue coding our graph in the `convNet` function.

```python
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
```

`logits` here is the output of the network which corresponds to the 10 classes of the training labels. The next two sections check whether we should be training the weights right now, or checking how well we've done. First we check our progress: we use `tf.one_hot` to create the one-hot labels from the numeric training labels given to the network in `labels`. We've performed a `tf.cast` operation to make sure that the data is of the correct type before doing the conversion.

Our loss-function is an important part of a CNN (or any machine learning algorithm). There are many different loss functions already built-in with TensorFlow from simple `absolute_difference` to more complex functions like our `softmax_cross_entropy`. We won't delve into how this is calculated, just know that we can pick any loss function. More advanced users can write their own loss-functions. The loss function takes in the output of the network `logits` and compares it to our `onehot_labels`.

When this is done, we ask TF to perform some updating or 'optimization' of the network based on the loss that we just calculated. the `train_op` in TF is the name given in support documents to the function that performs any background changes to the fundamentals of the network or updates values. Our `train_op` here is a simple loss-optimiser that tries to find the minimum loss for our data. As with all machine learning algorithms, the parameters of this optimiser are subject to much research. Using a pre-built optimiser such as those included with TF will ensure that your network performs efficiently and trains as quickly as possible. The `learning_rate` can be set as a variable at the beginning of our script along with the other parameters. We tend to stick with `0.001` to begin with and move in orders of magnitude if we need to e.g. `0.01` or `0.0001`. Just like the loss functions, there are a number of optimisers to use, some will take longer than others if they are more complex. For our purposes on the MNIST dataset, simple stochastic gradient descent (`SGD`) will suffice.

Notice that we are just giving TF some instructions: take my network, calculate the loss and do some optimisation based on that loss.

We are going to want to show what the network has learned, so we output the current predictions by definiing a dictionary of data. The raw logits information and the associated probabilities (found by taking the softmax of the logits tensor).

```
predictions ={"classes": tf.argmax(input=logits, axis=1), "probabilities": tf.nn.softmax(logits, name="softmax_tensor")}
 ```

We can finish off our graph by making sure it returns the data:

```
return model_fn_lib.ModelFnOps(mode=mode, predictions=predictions, loss=loss, train_op=train_op)
```

`ModelFnOps` class is returned that contains the current mode of the network (training or inference), the current predictions, loss and the `train_op` that we use to train the network.

<h3 id="setup">Setting up the Script</h3>

Now that the graph has been constructed, we need to call it and tell TF to do the training. First, lets take a moment to load the data the we will be using. The MNIST dataset has its own loading method within TF (handy!). Let's define the main body of our script:

```python
def main(unused_argv):
    # Load training and eval data
    mnist = learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
```
Next, we create the classifier that will hold the network and all of its data. We have to tell it what our graph is called under `model_fn` and where we would like our output stored.

**Note:** If you use the `/tmp` directory in Linux you will probably find that the model will no longer be there if you restart your computer. If you intend to reload and use your model later on, be sure to save it in a more conventient place.

```python
    mnistClassifier = learn.Estimator(model_fn=convNet,   model_dir="/tmp/mln_MNIST")
```

We will want to get some information out of our network that tells us about the training performance. For example, we can create a dictionary that will hold the probabilities from the key that we named 'softmax\_tensor' in the graph. How often we save this information is controlled with the `every_n_iter` attricute. We add this to the `tf.train.LoggingTensorHook`. 

```python
    tensors2log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors2log, every_n_iter=100)
 ```
 
 Finally! Let's get TF to actually train the network. We call the `.fit` method of the classifier that we created earlier. We pass it the training data and the labels along with the batch size (i.e. how much of the training data we want to use in each iteration). Bare in mind that even though the MNIST images are very small, there are 60,000 of them and this may not do well for your RAM. We also need to say what the maximum number of iterations we'd like TF to perform is and also add on that we want to `monitor` the training by outputting the data we've requested in `logging_hook`.
 
 ```python 
    mnistClassifier.fit(x=train_data, y=train_labels, batch_size=100, steps=1000, monitors=[logging_hook])
```
When the training is complete, we'd like TF to take some test-data and tell us how well the network performs. So we create a special metrics dictionary that TF will populate by calling the `.evaluate` method of the classifier.

```python
    metrics = {"accuracy": learn.MetricSpec(metric_fn=tf.metrics.accuracy, prediction_key="classes")}
    
    eval_results = mnistClassifier.evaluate(x=eval_data, y=eval_labels, metrics=metrics)
    print(eval_results)
```

In this case, we've chosen to find the accuracy of the classifier by using the `tf.metrics.accuracy` value for the `metric_fn`. We also need to tell the evaluator that it's the 'classes' key we're looking at in the graph. This is then passed to the evaluator along with the test data.

<h3 id="running">Running the Network</h3>

Adding the final main function to the script and making sure we've done all the necessary includes, we can run the program. The full script can be found [here](/docs/tfCNNMNIST.py "TFCNNMNIST.py").

In the current configuration, running the network for 1000 epochs gave me an output of:

```python
{'loss': 1.9025836, 'global_step': 1000, 'accuracy': 0.64929998}
```

Definitely not a great accuracy for the MNIST dataset! We could just run this for longer and would likely see an increase in accuracy, Instead, lets make some of the easy tweaks to our network that we've described before: dropout and batch normalisation.

In our graph, we want to add:

```python
    dense = tf.contrib.layers.batch_norm(dense, decay=0.99, is_training= mode==learn.ModeKeys.TRAIN)
    dense = tf.layers.dropout(inputs=dense, rate=keepProb, training = mode==learn.ModeKeys.TRAIN)
```

This layer [has many different attirbutes](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/batch_norm "tf.contrib.layers.batch_norm"). It's functionality is taken from [the paper by Loffe and Szegedy (2015)](https://arxiv.org/abs/1502.03167 "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift").

Dropout layer's `keepProb` is defined in the Hyperparameter pramble to the script. Another value that can be changed to improve the performance of the network. Both of these lines are in the final script [available here](/docs/tfCNNMNIST.py "tffCNNMNIST.py"), just uncomment them.

If we re-run the script, it will automatically load the most recent state of the network (clever TensorFlow!) but... it will fail because the checkpoint does not include the two new layers in its graph. So we must either delete our `/tmp/mln_MNIST` folder, or give the classifier a new `model_dir`.

Doing this and rerunning for the same 1000 epochs, I get an instant 140% increase in accuracy:

```python
{'loss': 0.29391664, 'global_step': 1000, 'accuracy': 0.91680002}
```

Simply changing the optimiser to use the "Adam" rather than "SGD" optimiser yields:

```python
{'loss': 0.040745325, 'global_step': 1000, 'accuracy': 0.98500001}
```

And running for slightly longer (20,000 iterations);

```python
{'loss': 0.046967514, 'global_step': 20000, 'accuracy': 0.99129999}
```

<h2 id="conclusion"> Conclusion </h2>

TensorFlow takes away the tedium of having to write out the full code for each individual layer and is able to perform optimisation and evaluation with minimal effort.

If you look around online, you will see many methods for using TF that will get you similar results. I actually prefer some methods that are a little more explicit. The tutorial on Google for example has some room to allow us to including more logging features.

In future posts, we will look more into logging and TensorBoard, but for now, happy coding!

