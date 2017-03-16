+++
description = "Coding up a Simple Neural Network in Python"
topics = ["tutorial"]
date = "2017-03-15T09:55:00Z"
title = "A Simple Neural Network - With Numpy in Python"
tags = ["neural network", "python", "back propagation", "numpy", "transfer function"]
social=true
featured_image="/img/brain4.png"
+++

Part 4 of our tutorial series on Simple Neural Networks. We're ready to write our Python script! Having gone through the maths, vectorisation and activation functions, we're now ready to put it all together and write it up. By the end of this tutorial, you will have a working NN in Python, using only numpy, which can be used to learn the output of logic gates (e.g. XOR)
<!--more-->

<div id="toctop"></div>

1. [Introduction][11]
2. [Transfer Function][12]
2. [Back Propagation Class][13]
	1. [Initialisation][14]
	2. [Forward Pass][15]
	3. [Back Propagation][16]
4. [Testing][17]
5. [Iterating][18]

[11]:{{< relref "#intro" >}}
[12]:{{< relref "#transferfunction" >}}
[13]:{{< relref "#backpropclass" >}}
[14]:{{< relref "#initialisation" >}}
[15]:{{< relref "#forwardpass" >}}
[16]:{{< relref "#backprop" >}}
[17]:{{< relref "#testing" >}}
[18]:{{< relref "#iterating" >}}

<h3 id="intro"> Introduction </h3>

[To contents][100]

We've [ploughed through the maths][1], then [some more][2], now we're finally here! This tutorial will run through the coding up of a simple neural network (NN) in Python. We're not going to use any fancy packages (though they obviously have their advantages in tools, speed, efficiency...) we're only going to use numpy!

By the end of this tutorial, we will have built an algorithm which will create a neural network with as many layers (and nodes) as we want. It will be trained by taking in multiple training examples and running the back propagation algorithm many times.

Here are the things we're going to need to code:

* The transfer functions
* The forward pass
* The back propagation algorithm
* The update function

To keep things nice and contained, the forward pass and back propagation algorithms should be coded into a class. We're going to expect that we can build a NN by creating an instance of this class which has some internal functions (forward pass, delta calculation, back propagation, weight updates).

First things first... lets import numpy:

{{< highlight python >}}
import numpy as np
{{</ highlight>}}

Now let's go ahead and get the first bit done:

<h2 id="transferfunction"> Transfer Function </h2>

[To contents][100]

To begin with, we'll focus on getting the network working with just one transfer function: the sigmoid function. As we discussed in a [previous post][3] this is very easy to code up because of its simple derivative:

<div >$$
f\left(x_{i} \right) = \frac{1}{1 + e^{  - x_{i}  }} \ \ \ \
f^{\prime}\left( x_{i} \right) = \sigma(x_{i}) \left( 1 -  \sigma(x_{i}) \right)
$$</div>

```python
def sigmoid(x, Derivative=False):
	if not Derivative:
		return 1 / (1 + np.exp (-x))
	else:
		out = sigmoid(x)
		return out * (1 - out)
```

This is a succinct expression which actually calls itself in order to get a value to use in its derivative. We've used numpy's exponential function to create the sigmoid function and created an `out` variable to hold this in the derivative. Whenever we want to use this function, we can supply the parameter `True` to get the derivative, We can omit this, or enter `False` to just get the output of the sigmoid. This is the same function I used to get the graphs in the [post on transfer functions][3].

<h2 id="backpropclass"> Back Propagation Class </h2>

[To contents][100]

I'm fairly new to building my own classes in Python, but for this tutorial, I really relied on the videos of [Ryan on YouTube][4]. Some of his hacks were very useful so I've taken some of those on board, but i've made a lot of the variables more self-explanatory.

First we're going to get the skeleton of the class setup. This means that whenever we create a new variable with the class of `backPropNN`, it will be able to access all of the functions and variables within itself.

It looks like this:

```python
class backPropNN:
    """Class defining a NN using Back Propagation"""
    
    # Class Members (internal variables that are accessed with backPropNN.member) 
    numLayers = 0
    shape = None
    weights = []
    
    # Class Methods (internal functions that can be called)
    
    def __init__(self):
        """Initialise the NN - setup the layers and initial weights"""
        
    # Forward Pass method
    def FP(self):
    	"""Get the input data and run it through the NN"""
    	 
    # TrainEpoch method
    def backProp(self):
        """Get the error, deltas and back propagate to update the weights"""
```
We've not added any detail to the functions (or methods) yet, but we know there needs to be an `__init__` method for any class, plus we're going to want to be able to do a forward pass and then back propagate the error.

We've also added a few class members, variables which can be called from an instance of the `backPropNN` class. `numLayers` is just that, a count of the number of layers in the network, initialised to `0`.  The `shape` of the network will return the size of each layer of the network in an array and the `weights` will return an array of the weights across the network.

<h3 id="initialisation"> Initialisation </h3>

[To contents][100]

We're going to make the user supply an input variablewhich is the size of the layers in the network i.e. the number of nodes in each layer: `numNodes`. This will be an array which is the length of the number of layers (including the input and output layers) where each element is the number of nodes in that layer.

```python
def __init__(self, numNodes):
	"""Initialise the NN - setup the layers and initial weights"""

	# Layer information
	self.numLayers = len(numNodes) - 1
	self.shape = numNodes
```
We've told our network to ignore the input layer when counting the number of layers (common practice) and that the shape of the network should be returned as the input array `numNodes`.

Lets also initialise the weights. We will take the approach of initialising all of the weights to small, random numbers. To keep the code succinct, we'll use a neat function`zip`. `zip` is a function which takes two vectors and pairs up the elements in corresponding locations (like a zip). For example:

```python
A = [1, 2, 3]
B = [4, 5, 6]

zip(A,B)
[(1,4), (2,5), (3,6)]
```

Why might this be useful? Well, when we talk about weights we're talking about the connections between layers. Lets say we have `numNodes=(2, 2, 1)` i.e. a 2 layer network with 2 inputs, 1 output and 2 nodes in the hidden layer. Then we need to let the algorithm know that we expect two input nodes to send weights to 2 hidden nodes. Then 2 hidden nodes to send weights to 1 output node, or `[(2,2), (2,1)]`. Note that overall we will have 4 weights from the input to the hidden layer, and 2 weights from the hidden to the output layer.

What is our `A` and `B` in the code above that will give us `[(2,2), (2,1)]`? It's this:

```python
numNodes = (2,2,1)
A = numNodes[:-1]
B = numNodes[1:]

A
(2,2)
B
(2,1)
zip(A,B)
[(2,2), (2,1)]
```

Great! So each pair represents the nodes between which we need initialise some weights. In fact, the shape of each pair `(2,2)` is the clue to how many weights we are going to need between each layer e.g. between the input and hidden layers we are going to need `(2 x 2) =4` weights.

so `for` each pair `in zip(A,B)` (hint hint) we need to `append` some weights into that empty weight matrix we initialised earlier. 

```python
# Initialise the weight arrays
for (l1,l2) in zip(numNodes[:-1],numNodes[1:]):
    self.weights.append(np.random.normal(scale=0.1,size=(l2,l1+1)))
```

`self.weights` as we're appending to the class member initialised earlier. We're using the numpy random number generator from a `normal` distribution. The `scale` just tells numpy to choose numbers around the 0.1 kind of mark and that we want a matrix of results which is the size of the tuple `(l2,l1+1)`. Huh, `+1`? Don't think we're getting away without including the _bias_ term! We want a random starting point even for the weight connecting the bias node (`=1`) to the next layer. Ok, but why this way and not `(l1+1,l2)`? Well, we're looking for `l2` connections from each of the `l1+1` nodes in the previous layer - think of it as (number of observations x number of features). We're creating a matrix of weights which goes across the nodes and down the weights from each node, or as we've seen in our maths tutorial:

<div>$$
W_{ij} = \begin{pmatrix} w_{11} & w_{21} & w_{31} \\ w_{12} &w_{22} & w_{32} \end{pmatrix}, \ \ \ \

W_{jk} = \begin{pmatrix} w_{11} & w_{21} & w_{31} \end{pmatrix}
$$</div>

Between the first two layers, and second 2 layers respectively with node 3 being the bias node.

Before we move on, lets also put in some placeholders in `__init__` for the input and output values to each layer:

```python
self._layerInput = []
self._layerOutput = []
```

<h3 id="forwardpass"> Forward Pass </h3>

[To contents][100]

We've now initialised out network enough to be able to focus on the forward pass (FP).

Our `FP` function needs to have the input data. It needs to know how many training examples it's going to have to go through, and it will need to reassign the inputs and outputs at each layer, so lets clean those at the beginning:

```python
def FP(self,input):

	numExamples = input.shape[0]

	# Clean away the values from the previous layer
	self._layerInput = []
	self._layerOutput = []
```

So lets propagate. We already have a matrix of (randomly initialised) weights. We just need to know what the input is to each of the layers. We'll separate this into the first hidden layer, and subsequent hidden layers.

For the first hidden layer we will write:

```python
layerInput = self.weights[0].dot(np.vstack([input.T, np.ones([1, numExamples])]))
```
Let's break this down:

Our training example inputs need to match the weights that we've already created. We expect that our examples will come in rows of an array with columns acting as features, something like `[(0,0), (0,1),(1,1),(1,0)]`. We can use numpy's `vstack` to put each of these examples one on top of the other.

Each of the input examples is a matrix which will be multiplied by the weight matrix to get the input to the current layer:

<div>$$
\mathbf{x_{J}} = \mathbf{W_{IJ} \vec{\mathcal{O}}_{I}}
$$</div>

where $\mathbf{x\_{J}}$ are the inputs to the layer $J$ and  $\mathbf{\vec{\mathcal{O}}\_{I}}$ is the output from the precious layer (the input examples in this case).

So given a set of $n$ input examples we `vstack` them so we just have `(n x numInputNodes)`. We want to transpose this, `(numInputNodes x n)` such that we can multiply by the weight matrix which is `(numOutputNodes x numInputNodes)`. This gives an input to the layer which is `(numOutputNodes x n)` as we expect.

**Note** we're actually going to do the transposition first before doing the `vstack` - this does exactly the same thing, but it also allows us to more easily add the bias nodes in to each input.

Bias! Lets not forget this: we add a bias node which always has the value `1` to each input (including the input layer). So our actual method is:

1. Transpose the inputs `input.T`
2. Add a row of ones to the bottom (one bias node for each input) `[input.T, np.ones([1,numExamples])]`
3. `vstack` this to compact the array `np.vstack(...)`
4. Multipy with the weights connecting from the previous to the current layer `self.weights[0].dot(...)`

But what about the subsequent hidden layers? We're not using the input examples in these layers, we are using the output from the previous layer `[self._layerOutput[-1]]` (multiplied by the weights).

```python
for index in range(self.numLayers):
#Get input to the layer
if index ==0:
        layerInput = self.weights[0].dot(np.vstack([input.T, np.ones([1, numExamples])]))
else:
        layerInput = self.weights[index].dot(np.vstack([self._layerOutput[-1],np.ones([1,numExamples])]))
```

Make sure to save this output, but also to now calculate the output of the current layer i.e.:

<div>$$
\mathbf{ \vec{ \mathcal{O}}_{J}} = \sigma(\mathbf{x_{J}})
$$</div>

```python
self._layerInput.append(layerInput)
self._layerOutput.append(sigmoid(layerInput))
```

Finally, make sure that we're returning the data from our output layer the same way that we got it:

```python
return self._layerOutput[-1].T
```

<h3 id="backprop">Back Propagation</h3>

[To contents][100]

We've successfully sent the data from the input layer to the output layer using some initially randomised weights **and** we've included the bias term (a kind of threshold on the activation functions). Our vectorised equations from the previous post will now come into play:

<div>$$
\begin{align}

\mathbf{\vec{\delta}_{K}} &= \sigma^{\prime}\left( \mathbf{W_{JK}}\mathbf{\vec{\mathcal{O}}_{J}} \right) * \left( \mathbf{\vec{\mathcal{O}}_{K}} -  \mathbf{T_{K}}\right) \\[0.5em]

\mathbf{ \vec{ \delta }_{J}} &= \sigma^{\prime} \left( \mathbf{ W_{IJ} \mathcal{O}_{I} } \right) * \mathbf{ W^{\intercal}_{JK}} \mathbf{ \vec{\delta}_{K}}

\end{align}
$$</div>

<div>$$
\begin{align}

\mathbf{W_{JK}} + \Delta \mathbf{W_{JK}} &\rightarrow \mathbf{W_{JK}}, \ \ \ \Delta \mathbf{W_{JK}} = -\eta \mathbf{ \vec{ \delta }_{K}} \mathbf{ \vec { \mathcal{O} }_{J}} \\[0.5em]

\vec{\theta}  + \Delta \vec{\theta}  &\rightarrow \vec{\theta}, \ \ \ \Delta \vec{\theta} = -\eta \mathbf{ \vec{ \delta }_{K}} 

\end{align}
$$</div>

With $*$ representing an elementwise multiplication between the matrices.

First, lets initialise some variables and get the error on the output of the output layer. We assume that the target values have been formatted in the same way as the input values i.e. they are a row-vector per input example. In our forward propagation method, the outputs are stored as column-vectors, thus the targets have to be transposed. We will need to supply the input data, the target data and  $\eta$, the learning rate, which we will set at some small number for default. So we start back propagation by first initialising a placeholder for the deltas and getting the number of training examples before running them through the `FP` method:

```python
def backProp(self, input, target, trainingRate = 0.2):
"""Get the error, deltas and back propagate to update the weights"""

delta = []
numExamples = input.shape[0]

# Do the forward pass
self.FP(input)

output_delta = self._layerOutput[index] - target.T
error = np.sum(output_delta**2)
```
We know from previous posts that the error is squared to get rid of the negatives. From this we compute the deltas for the output layer:

```python
delta.append(output_delta * sigmoid(self._layerInput[index], True))
```
We now have the error but need to know what direction to alter the weights in, thus the gradient of the inputs to the layer need to be known. So, we get the gradient of the activation function at the input to the layer and get the product with the error. Notice we've supplied `True` to the sigmoid function to get its derivative.

This is the delta for the output layer. So this calculation is only done when we're considering the index at the end of the network. We should be careful that when telling the algorithm that this is the "last layer" we take account of the zero-indexing in Python i.e. the last layer is `self.numLayers - 1` i.e. in a network with 2 layers, `layer[2]` does not exist.

We also need to get the deltas of the intermediate hidden layers. To do this, (according to our equations above) we have to 'pull back' the delta from the output layer first. More accurately, for any hidden layer, we pull back the delta from the _next_ layer, which may well be another hidden layer. These deltas from the _next_ layer are multiplied by the weights from the _next_ layer `[index + 1]`, before getting the product with the sigmoid derivative evaluated at the _current_ layer.

**Note**: this is _back_ propagation. We have to start at the end and work back to the beginning. We use the `reversed` keyword in our loop to ensure that the algorithm considers the layers in reverse order.

Combining this into one method:

```python
# Calculate the deltas
for index in reversed(range(self.numLayers)):
    if index == self.numLayers - 1:
        # If the output layer, then compare to the target values
        output_delta = self._layerOutput[index] - target.T
        error = np.sum(output_delta**2)
        delta.append(output_delta * sigmoid(self._layerInput[index], True))
    else:
        # If a hidden layer. compare to the following layer's delta
        delta_pullback = self.weights[index + 1].T.dot(delta[-1])
        delta.append(delta_pullback[:-1,:] * sigmoid(self._layerInput[index], True))
```

Pick this piece of code apart. This is an important snippet as it calculates all of the deltas for all of the nodes in the network. Be sure that we understand:

1. This is a `reversed` loop because we want to deal with the last layer first
2. The delta of the output layer is the residual between the output and target multiplied with the gradient (derivative) of the activation function _at the current layer_.
3. The delta of a hidden layer first needs the product of the _subsequent_ layer's delta with the _subsequent_ layer's weights. This is then multiplied with the gradient of the activation function evaluated at the _current_ layer.

Double check that this matches up with the equations above too! We can double check the matrix multiplication. For the output layer:

`output_delta` = (numOutputNodes x 1) - (1 x numOutputNodes).T = (numOutputNodes x 1)
`error` = (numOutputNodes x 1) **2 = (numOutputNodes x 1) 
`delta` = (numOutputNodes x 1) * sigmoid( (numOutputNodes x 1) ) = (numOutputNodes  x 1)

For the hidden layers (take the one previous to the output as example):

`delta_pullback` = (numOutputNodes x numHiddenNodes).T.dot(numOutputNodes x 1) = (numHiddenNodes x 1)
`delta` = (numHiddenNodes x 1) * sigmoid ( (numHuddenNodes x 1) ) = (numHiddenNodes x 1)

Hurray! We have the delta at each node in our network. We can use them to update the weights for each layer in the network. Remember, to update the weights between layer $J$ and $K$ we need to use the output of layer $J$ and the deltas of layer $K$. This means we need to keep a track of the index of the layer we're currently working on ($J$) and the index of the delta layer ($K$) - not forgetting about the zero-indexing in Python:

```python
for index in range(self.numLayers):
    delta_index = self.numLayers - 1 - index
```
Let's first get the outputs from each layer:

```python
    if index == 0:
        layerOutput = np.vstack([input.T, np.ones([1, numExamples])])
    else:
        layerOutput = np.vstack([self._layerOutput[index - 1], np.ones([1,self._layerOutput[index -1].shape[1]])])
```
The output of the input layer is just the input examples (which we've `vstack`-ed again and the output from the other layers we take from calculation in the forward pass (making sure to add the bias term on the end).

For the current `index` (layer) lets use this `layerOutput` to get the change in weight. We will use a few neat tricks to make this succinct:

```python
	thisWeightDelta = np.sum(\
	    layerOutput[None,:,:].transpose(2,0,1) * delta[delta_index][None,:,:].transpose(2,1,0) \
	    , axis = 0)
```

Break it down. We're looking for $\mathbf{ \vec{ \delta }\_{K}} \mathbf{ \vec { \mathcal{O} }\_{J}} $ so it's the delta at `delta_index`, the next layer along.

We want to be able to deal with all of the input training examples simultaneously. This requires a bit of fancy slicing and transposing of the matrices. Take a look: by calling `vstack` we made all of the input data and bias terms live in the same matrix of a numpy array. When we slice this arraywith the `[None,:,:]` argument, it tells Python to take all (`:`) the data in the rows and columns and shift it to the 1st and 2nd dimensions and leave the first dimension empty (`None`). We do this to create the three dimensions which we can now transpose into. Calling `transpose(2,0,1)` instructs Python to move around the dimensions of the data (e.g. its rows... or examples). This creates an array where each example now lives in its own plane. The same is done for the deltas of the subsequent layer, but being careful to transpost them in the opposite direction so that the matrix multiplication can occur. The `axis= 0` is supplied to make sure that the inputs are multiplied by the correct dimension of the delta matrix.

This looks incredibly complicated. It an be broken down into a for-loop over the input examples, but this reduces the efficiency of the network. Taking advantage of the numpy array like this keeps our calculations fast. In reality, if you're struggling with this particular part, just copy and paste it, forget about it and be happy with yourself for understanding the maths behind back propagation, even if this random bit of Python is perplexing.

Anyway. Lets take this set of weight deltas and put back the $\eta$. We'll call this the `learningRate`. It's called a lot of things, but this seems to be the most common. We'll update the weights by making sure to include the `-` from the $-\eta$.

```python
	weightDelta = trainingRate * thisWeightDelta
	self.weights[index] -= weightDelta
```
the `-=` is Python slang for: take the current value and subtract the value of `weightDelta`.

To finish up, we want our back propagation to return the current error in the network, so:

```python
return error
```

<h2 id="testing"> A Toy Example</h2>

[To contents][100]

Believe it or not, that's it! The fundamentals of forward and back propagation have now been implemented in Python. If you want to double check your code, have a look at my completed .py [here][5]

Let's test it!

```python
Input = np.array([[0,0],[1,1],[0,1],[1,0]])
Target = np.array([[0.0],[0.0],[1.0],[1.0]])

NN = backPropNN((2,2,1))

Error = NN.backProp(Input, Target)
Output = NN.FP(Input)

print 'Input \tOutput \t\tTarget'
for i in range(Input.shape[0]):
    print '{0}\t {1} \t{2}'.format(Input[i], Output[i], Target[i])
```

This will provide 4 input examples and the expected targets. We create an instance of the network called `NN` with 2 layers (2 nodes in the hidden and 1 node in the output layer). We make `NN` do `backProp` with the input and target data and then get the output from the final layer by running out input through the network with a `FP`. The printout is self explantory. Give it a try!

```
Input 	Output 		Target
[0 0]	 [ 0.51624448] 	[ 0.]
[1 1]	 [ 0.51688469] 	[ 0.]
[0 1]	 [ 0.51727559] 	[ 1.]
[1 0]	 [ 0.51585529] 	[ 1.]
```

We can see that the network has taken our inputs, and we have some outputs too. They're not great, and all seem to live around the same value. This is because we initialised the weights across the network to a similarly small random value. We need to repeat the `FP` and `backProp` process many times in order to keep updating the weights.

<h2 id="iterating"> Iterating </h2>

[To contents][100]

Iteration is very straight forward. We just tell our algorithm to repeat a maximum of `maxIterations` times or until the `Error` is below `minError` (whichever comes first). As the weights are stored internally within `NN` every time we call the `backProp` method, it uses the latest, internally stored weights and doesn't start again - the weights are only initialised once upon creation of `NN`.

```python
maxIterations = 100000
minError = 1e-5

for i in range(maxIterations + 1):
    Error = NN.backProp(Input, Target)
    if i % 2500 == 0:
        print("Iteration {0}\tError: {1:0.6f}".format(i,Error))
    if Error <= minError:
        print("Minimum error reached at iteration {0}".format(i))
        break
```

Here's the end of my output from the first run:

```
Iteration 100000	Error: 0.000291
Input 	Output 		Target
[0 0]	 [ 0.00780385] 	[ 0.]
[1 1]	 [ 0.00992829] 	[ 0.]
[0 1]	 [ 0.99189799] 	[ 1.]
[1 0]	 [ 0.99189943] 	[ 1.]
```

Much better! The error is very small and the outputs are very close to the correct value. However, they're note completely right. We can do better, by implementing different activation functions which we will do in the next tutorial.

**Please** let me know if anything is unclear, or there are mistakes. Let me know how you get on!

[100]:{{< relref "#toctop" >}}

[1]: /post/neuralnetwork
[2]: /post/nn-more-maths
[3]: /post/transfer-functions
[4]: https://www.youtube.com/playlist?list=PLRyu4ecIE9tibdzuhJr94uQeKnOFkkbq6
[5]: /docs/simpleNN.py