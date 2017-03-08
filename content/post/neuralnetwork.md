+++
title = "Simple Neural Network - Mathematics"
description = "Understanding the maths of Neural Networks"
topics = ["tutorials"]
tags = ["neural network","back propagation", "machine learning"]
date = "2017-03-06T17:04:53Z"
social=true
+++

<div id="toctop"></div>

Tutorials on neural networks (NN) can be found all over the internet. Though many of them are the same, each is written (or recorded) slightly differently. This means that I always feel like I learn something new or get a better understanding of things with every tutorial I see. I'd like to make this tutorial as clear as I can, so sometimes the maths may be simplistic, but hopefully it'll give you a good unserstanding of what's going on. **Please** let me know if any of the notation is incorrect or there are any mistakes - either comment or use the contact page on the left.

1. [Neural Network Architecture][1]
2. [Transfer Function][2]
3. [Feed-forward][3]
4. [Error][4]
5. [Back Propagation - the Gradients][5]
6. [Bias][6]
7. [Back Propagaton - the Algorithm][7]

[1]:{{< relref "#nnarchitecture" >}}
[2]:{{< relref "#transferFunction" >}}
[3]:{{< relref "#feedforward" >}}
[4]:{{< relref "#error" >}}
[5]:{{< relref "#backPropagationGrads" >}}
[6]:{{< relref "#bias" >}}
[7]:{{< relref "#backPropagationAlgorithm" >}}

<h2 id="nnarchitecture">1. Neural Network Architecture </h2>

[To contents][100]

By now, you may well have come across diagrams which look very similar to the one below. It shows some input node, connected to some output node via an intermediate node in what is called a 'hidden layer' - 'hidden' because in the use of NN only the input and output is of concern to the user, the 'under-the-hood' stuff may not be interesting to them. In real, high-performing NN there are usually more hidden layers.

<div class="container">
	<div class="figure_container">
		<img title="Simple NN" style="width:80%" class="figure_img" src="/img/simpleNN/simpleNN.png">
		<div class="figure_caption">
			<font color="blue">Figure 1</font>: A simple 2-layer NN with 2 features in the input layer, 3 nodes in the hidden layer and two nodes in the output layer.
		</div>
	</div>
</div>

When we train our network, the nodes in the hidden layer each perform a calculation using the values from the input nodes. The output of this is passed on to the nodes of the next layer. When the output hits the final layer, the 'output layer', the results are compared to the real, known outputs and some tweaking of the network is done to make the output more similar to the real results. This is done with an algorithm called _back propagation_. Before we get there, lets take a closer look at these calculations being done by the nodes.

<h2 id="transferFunction">2. Transfer Function </h2>

[To contents][100]

At each node in the hidden and output layers of the NN, an _activation_ or _transfer_ function is executed. This function takes in the output of the previous node, and multiplies it by some _weight_. These weights are the lines which connect the nodes. The weights that come out of one node can all be different, that is they will _activate_ different neurons. There can be many forms of the transfer function, we will first look at the _sigmoid_ transfer function as it seems traditional.

<div class="container">
	<div class="figure_container">
		<img title="The sigmoid function" class="figure_img" src="/img/simpleNN/sigmoid.png">
		<div class="figure_caption">
			<font color="blue">Figure 2</font>: The sigmoid function.
		</div>
	</div>
</div>

As you can see from the figure, the sigmoid function takes any real-valued input and maps it to a real number in the range $(0 \ 1)$ - i.e. between, but not equal to, 0 and 1. We can think of this almost like saying 'if the value we have maps to an output near 1, this node fires, if it maps to an output near 0, the node does not fire'. The equation for this sigmoid function is:

<div id="eqsigmoidFunction">$$
\sigma ( x ) = \frac{1}{1 + e^{-x}}
$$</div>

We need to have the derivative of this transfer function so that we can perform back propagation later on. This is the process where by the connections in the network are updated to tune the performance of the NN. We'll talk about this in more detail later, but let's find the derivative now.

<div>
$$
\begin{align*}
\frac{d}{dx}\sigma ( x ) &= \frac{d}{dx} \left( 1 + e^{ -x }\right)^{-1}\\
&=  -1 \times -e^{-x} \times \left(1 + e^{-x}\right)^{-2}= \frac{ e^{-x} }{ \left(1 + e^{-x}\right)^{2} } \\
&= \frac{\left(1 + e^{-x}\right) - 1}{\left(1 + e^{-x}\right)^{2}} 
= \frac{\left(1 + e^{-x}\right) }{\left(1 + e^{-x}\right)^{2}} - \frac{1}{\left(1 + e^{-x}\right)^{2}} 
= \frac{1}{\left(1 + e^{-x}\right)} - \left( \frac{1}{\left(1 + e^{-x}\right)} \right)^{2} \\[0.5em]
&= \sigma ( x ) - \sigma ( x ) ^ {2}
\end{align*}
$$</div>

Therefore, we can write the derivative of the sigmoid function as:

<div id="eqdsigmoid">$$
\sigma^{\prime}( x ) = \sigma (x ) \left( 1 - \sigma ( x ) \right)
$$</div>

The sigmoid function has the nice property that its derivative is very simple: a bonus when we want to hard-code this into our NN later on. Now that we have our activation or transfer function selected, what do we do with it?

<h2 id="feedforward">3. Feed-forward </h2>

[To contents][100]

During a feed-forward pass, the network takes in the input values and gives us some output values. To see how this is done, let's first consider a 2-layer neural network like the one in Figure 1. Here we are going to refer to:

* $i$ - the $i^{\text{th}}$ node of the input layer $I$
* $j$ - the $j^{\text{th}}$ node of the hidden layer $J$
* $k$ - the $k^{\text{th}}$ node of the input layer $K$

The activation function at a node $j$ in the hidden layer takes the value:

<div>$$
\begin{align}
x_{j} &= \xi_{1} w_{1j} + \xi_{2} w_{2j} \\[0.5em]
&= \sum_{i \in I} \xi_{i} w_{i j}

\end{align}
$$</div>

where $\xi\_{i}$ is the value of the $i^{\text{th}}$ input node and $w\_{i j}$ is the weight of the connection between $i^{\text{th}}$ input node and the $j^{\text{th}}$ hidden node. **In short:** at each hidden layer node, multiply each input value by the connection received by that node and add them together. 

**Note:** the weights are initisliased when the network is setup. Sometimes they are all set to 1, or often they're set to some small random value.

We apply the activation function on $x\_{j}$ at the $j^{\text{th}}$ hidden node and get:

<div>$$
\begin{align}
\mathcal{O}_{j} &= \sigma(x_{j}) \\
&= \sigma(  \xi_{1} w_{1j} + \xi_{2} w_{2j})
\end{align}
$$</div>

$\mathcal{O}\_{j}$ is the output of the $j^{\text{th}}$ hidden node. This is calculated for each of the $j$ nodes in the hidden layer. The resulting outputs now become the input for the next layer in the network. In our case, this is the final output later. So for each of the $k$ nodes in $K$:

<div>$$
\begin{align}
\mathcal{O}_{k} &= \sigma(x_{k}) \\
&= \sigma \left( \sum_{j \in J}  \mathcal{O}_{j} w_{jk}  \right)
\end{align}
$$</div>

As we've reached the end of the network, this is also the end of the feed-foward pass. So how well did our network do at getting the correct result $\mathcal{O}\_{k}$? As this is the training phase of our network, the true results will be known an we cal calculate the error.

<h2 id="error">4. Error </h2>

[To contents][100]

We measure error at the end of each foward pass. This allows us to quantify how well our network has performed in getting the correct output. Let's define $t\_{k}$ as the expected or _target_ value of the $k^{\text{th}}$ node of the output layer $K$. Then the error $E$ on the entire output is:

<div id="eqerror">$$
\text{E} = \frac{1}{2} \sum_{k \in K} \left( \mathcal{O}_{k} - t_{k} \right)^{2}
$$</div>

Dont' be put off by the random 1/2 in front there, it's been manufactured that way to make the upcoming maths easier. The rest of this should be easy enough: get the residual (difference between the target and output values), square this to get rid of any negatives and sum this over all of the nodes in the output layer.

Good! Now how does this help us? Our aim here is to find a way to tune our network such that when we do a forward pass of the input data, the output is exactly what we know it should be. But we can't change the input data, so there are only two other things we can change:

1. the weights going into the activation function
2. the activation function itself

We will indeed consider the second case in another post, but the magic of NN is all about the _weights_. Getting each weight i.e. each connection between nodes, to be just the perfect value, is what back propagation is all about. The back propagation algorithm we will look at in the next section, but lets go ahead and set it up by considering the following: how much of this error $E$ has come from each of the weights in the network?

We're asking, what is the proportion of the error coming from each of the $W\_{jk}$ connections between the nodes in layer $J$ and the output layer $K$. Or in mathematical terms:

<div>$$
\frac{\partial{\text{E}}}{\partial{W_{jk}}} =  \frac{\partial{}}{\partial{W_{jk}}}  \frac{1}{2} \sum_{k \in K} \left( \mathcal{O}_{k} - t_{k} \right)^{2}
$$</div>

If you're not concerned with working out the derivative, skip this highlighted section.

<div class="highlight_section">

To tackle this we can use the following bits of knowledge: the derivative of the sum is equal to the sum of the derivatives i.e. we can move the derivative term inside of the summation:

<div>$$ \frac{\partial{\text{E}}}{\partial{W_{jk}}} =  \frac{1}{2} \sum_{k \in K} \frac{\partial{}}{\partial{W_{jk}}} \left( \mathcal{O}_{k} - t_{k} \right)^{2}$$</div>

* the weight $w\_{1k}$ does not affect connection $w\_{2k}$ therefore the change in $W_{jk}$ with respect to any node other than the current $k$ is zero. Thus the summation goes away:

<div>$$ \frac{\partial{\text{E}}}{\partial{W_{jk}}} =  \frac{1}{2} \frac{\partial{}}{\partial{W_{jk}}}  \left( \mathcal{O}_{k} - t_{k} \right)^{2}$$</div>

* apply the power rule knowing that $t_{k}$ is a constant:

<div>$$ 
\begin{align}
\frac{\partial{\text{E}}}{\partial{W_{jk}}} &=  \frac{1}{2} \times 2 \times \left( \mathcal{O}_{k} - t_{k} \right) \frac{\partial{}}{\partial{W_{jk}}}  \left( \mathcal{O}_{k}\right) \\
 &=  \left( \mathcal{O}_{k} - t_{k} \right) \frac{\partial{}}{\partial{W_{jk}}}  \left( \mathcal{O}_{k}\right)
\end{align}
$$</div>

* the leftover derivative is the chage in the output values with respect to the weights. Substituting $ \mathcal{O}\_{k} = \sigma(x_{k}) $ and the sigmoid derivative $\sigma^{\prime}( x ) = \sigma (x ) \left( 1 - \sigma ( x ) \right)$:

<div>$$ 
\frac{\partial{\text{E}}}{\partial{W_{jk}}} =  \left( \mathcal{O}_{k} - t_{k} \right) \sigma (x ) \left( 1 - \sigma ( x ) \right) \frac{\partial{}}{\partial{W_{jk}}}  \left( x_{k}\right)
$$</div>

* the final derivative, the input value $x\_{k}$ is just $\mathcal{O}\_{j} W\_{jk}$ i.e. output of the previous layer times the weight to this layer. So the change in  $\mathcal{O}\_{j} w\_{jk}$ with respect to $w\_{jk}$ just gives us the output value of the previous layer $ \mathcal{O}_{j} $ and so the full derivative becomes:

<div>$$ 
\begin{align}
\frac{\partial{\text{E}}}{\partial{W_{jk}}}  &=  \left( \mathcal{O}_{k} - t_{k} \right) \sigma (x ) \left( 1 - \sigma ( x ) \right) \frac{\partial{}}{\partial{W_{jk}}}  \left( \mathcal{O}_{j} W_{jk} \right) \\[0.5em]
&=\left( \mathcal{O}_{k} - t_{k} \right) \sigma (x )  \left( 1 - \mathcal{O}_{k}  \right) \mathcal{O}_{j} 
\end{align}
$$</div>

We can replace the sigmoid function with the output of the layer
</div>

The derivative of the error function with respect to the weights is then:

<div id="derror">$$ 
\frac{\partial{\text{E}}}{\partial{W_{jk}}}  =\left( \mathcal{O}_{k} - t_{k} \right) \mathcal{O}_{k}  \left( 1 - \mathcal{O}_{k}  \right) \mathcal{O}_{j}
$$</div>

We group the terms involving $k$ and define:

<div>$$
\delta_{k} = \mathcal{O}_{k}  \left( 1 - \mathcal{O}_{k}  \right)  \left( \mathcal{O}_{k} - t_{k} \right)
$$</div>

And therefore:

<div id="derrorjk">$$ 
\frac{\partial{\text{E}}}{\partial{W_{jk}}}  = \mathcal{O}_{j} \delta_{k} 
$$</div>

So we have an expression for the amount of error, called 'deta' ($\delta\_{k}$), on the weights from the nodes in $J$ to each node $k$ in $K$. But how does this help us to improve out network? We need to back propagate the error.

<h2 id="backPropagationGrads">5. Back Propagation - the gradients</h2>

[To contents][100]

Back propagation takes the error function we found in the previous section, uses it to calculate the error on the current layer and updates the weights to that layer by some amount.

So far we've only looked at the error on the output layer, what about the hidden layer? This also has an error, but the error here depends on the output layer's error too (because this is where the difference between the target $t\_{k}$ and output $\mathcal{O}\_{k}$ can be calculated). Lets have a look at the error on the weights of the hidden layer $W\_{ij}$:

<div>$$ \frac{\partial{\text{E}}}{\partial{W_{ij}}} =  \frac{\partial{}}{\partial{W_{ij}}}  \frac{1}{2} \sum_{k \in K} \left( \mathcal{O}_{k} - t_{k} \right)^{2}$$</div>

Now, unlike before, we cannot just drop the summation as the derivative is not directly acting on a subscript $k$ in the summation. We should be careful to note that the output from every node in $J$ is actually connected to each of the nodes in $K$ so the summation should stay. But we can still use the same tricks as before: lets use the power rule again and move the derivative inside (because the summation is finite):

<div>$$
\begin{align}
\frac{\partial{\text{E}}}{\partial{W_{ij}}} &=  \frac{1}{2} \times 2 \times  \frac{\partial{}}{\partial{W_{ij}}}   \sum_{k \in K} \left( \mathcal{O}_{k} - t_{k} \right)  \mathcal{O}_{k} \\
&= \sum_{k \in K} \left( \mathcal{O}_{k} - t_{k} \right) \frac{\partial{}}{\partial{W_{ij}}} \mathcal{O}_{k}
 \end{align}
 $$</div>
 
 Again, we substitute $\mathcal{O}\_{k} = \sigma( x\_{k})$ and its derivative and revert back to our output notation:
 
<div>$$
\begin{align}
\frac{\partial{\text{E}}}{\partial{W_{ij}}} &= \sum_{k \in K} \left( \mathcal{O}_{k} - t_{k} \right) \frac{\partial{}}{\partial{W_{ij}}} (\sigma(x_{k}) )\\
&= \sum_{k \in K} \left( \mathcal{O}_{k} - t_{k} \right) \sigma(x_{k}) \left( 1 - \sigma(x_{k}) \right) \frac{\partial{}}{\partial{W_{ij}}} (x_{k}) \\
&= \sum_{k \in K} \left( \mathcal{O}_{k} - t_{k} \right) \mathcal{O}_{k} \left( 1 - \mathcal{O}_{k} \right) \frac{\partial{}}{\partial{W_{ij}}} (x_{k})
 \end{align}
 $$</div>
 
 This still looks familar from the output layer derivative, but now we're struggling with the derivative of the input to $k$ i.e. $x_{k}$ with respect to the weights from $I$ to $J$. Let's use the chain rule to break apart this derivative in terms of the output from $J$:

<div> $$
\frac{\partial{ x_{k}}}{\partial{W_{ij}}} = \frac{\partial{ x_{k}}}{\partial{\mathcal{O}_{j}}}\frac{\partial{\mathcal{O}_{j}}}{\partial{W_{ij}}}
$$</div>

The change of the input to the $k^{\text{th}}$ node with respect to the output from the $j^{\text{th}}$ node is down to a product with the weights, therefore this derivative just becomes the weights $W\_{jk}$. The final derivative has nothing to do with the subscript $k$ anymore, so we're free to move this around - lets put it at the beginning:

<div>$$
\begin{align}
\frac{\partial{\text{E}}}{\partial{W_{ij}}} &= \frac{\partial{\mathcal{O}_{j}}}{\partial{W_{ij}}}  \sum_{k \in K} \left( \mathcal{O}_{k} - t_{k} \right) \mathcal{O}_{k} \left( 1 - \mathcal{O}_{k} \right) W_{jk}
 \end{align}
 $$</div>
 
Lets finish the derivatives, remembering that the output of the node $j$ is just $\mathcal{O}\_{j} = \sigma(x\_{j}) $ and we know the derivative of this function too:
 
<div>$$
\begin{align}
\frac{\partial{\text{E}}}{\partial{W_{ij}}} &= \frac{\partial{}}{\partial{W_{ij}}}\sigma(x_{j})  \sum_{k \in K} \left( \mathcal{O}_{k} - t_{k} \right) \mathcal{O}_{k} \left( 1 - \mathcal{O}_{k} \right) W_{jk} \\
&= \sigma(x_{j}) \left( 1 - \sigma(x_{j}) \right)  \frac{\partial{x_{j} }}{\partial{W_{ij}}} \sum_{k \in K} \left( \mathcal{O}_{k} - t_{k} \right) \mathcal{O}_{k} \left( 1 - \mathcal{O}_{k} \right) W_{jk} \\
&= \mathcal{O}_{j} \left( 1 - \mathcal{O}_{j} \right)  \frac{\partial{x_{j} }}{\partial{W_{ij}}} \sum_{k \in K} \left( \mathcal{O}_{k} - t_{k} \right) \mathcal{O}_{k} \left( 1 - \mathcal{O}_{k} \right) W_{jk}
 \end{align}
 $$</div>
 
 The final derivative is straightforward too, the derivative of the input to $j$ with repect to the weights is just the previous input, which in our case is $\mathcal{O}\_{i}$,
 
<div>$$
\begin{align}
\frac{\partial{\text{E}}}{\partial{W_{ij}}} &= \mathcal{O}_{j} \left( 1 - \mathcal{O}_{j} \right)  \mathcal{O}_{i} \sum_{k \in K} \left( \mathcal{O}_{k} - t_{k} \right) \mathcal{O}_{k} \left( 1 - \mathcal{O}_{k} \right) W_{jk}
 \end{align}
 $$</div>
 
 Almost there! Recall that we defined $\delta\_{k}$ earlier, lets sub that in:
 
<div>$$
\begin{align}
\frac{\partial{\text{E}}}{\partial{W_{ij}}} &= \mathcal{O}_{j} \left( 1 - \mathcal{O}_{j} \right)  \mathcal{O}_{i} \sum_{k \in K} \delta_{k} W_{jk}
 \end{align}
 $$</div>
 
 To clean this up, we now define the 'delta' for our hidden layer:
 
<div>$$
\delta_{j} = \mathcal{O}_{i} \left( 1 - \mathcal{O}_{j} \right)   \sum_{k \in K} \delta_{k} W_{jk}
$$</div>

Thus, the amount of error on each of the weights going into our hidden layer:

<div id="derrorij">$$ 
\frac{\partial{\text{E}}}{\partial{W_{ij}}}  = \mathcal{O}_{i} \delta_{j} 
$$</div>

**Note:** the reason for the name _back_ propagation is that we must calculate the errors at the far end of the network and work backwards to be able to calculate the weights at the front.

<h2 id="bias">6.  Bias </h2>

[To contents][100]

Lets remind ourselves what happens inside our hidden layer nodes:

<div class="container">
	<div class="figure_container">
		<img title="Simple NN"  class="figure_img" src="/img/simpleNN/nodeInsideNoBias.png">
		<div class="figure_caption">
			<font color="blue">Figure 3</font>: The insides of a hidden layer node, $j$.
		</div>
	</div>
</div>

1. Each feature $\xi\_{i}$ from the input layer $I$ is multiplied by some weight $w\_{ij}$
2. These are added together to get $x\_{i}$ the total, weighted input from the nodes in $I$
3. $x\_{i}$ is passed through the activation, or transfer, function $\sigma(x\_{i})$
4. This gives the output $\mathcal{O}\_{j}$ for each of the $j$ nodes in hidden layer $J$
5. $\mathcal{O}\_{j}$ from each of the $J$ nodes becomes $\xi\_{j}$ for the next layer

When we talk about the _bias_ term in NN, we are talking about an additional parameter that is inluded in the summation of step 2 above. The bias term is usually denoted with the symbol $\theta$ (theta). It's function is to act as a threshold for the activation (transfer) function. It is given the value of 1 and is not connected to anything else. As such, this means that any derivative of the node's output with respect to the bias term would just give a constant, 1. This allows us to just think of the bias term as an output from the node with the value of 1. This will be updated later during backpropagation to change the threshold at which the node fires.

Lets update the equation for $x\_{i}$ put it on the diagram:

<div>$$
\begin{align}
x_{i} &= \xi_{1j} w_{1j} + \xi_{2j} w_{2j} + \theta_{j} \\[0.5em]
\sigma( x_{i} ) &= \sigma \left( \sum_{i \in I} \left( \xi_{ij} w_{ij} \right) + \theta_{j} \right)
\end{align}
$$</div>

<div class="container">
	<div class="figure_container">
		<img title="Simple NN"  class="figure_img" src="/img/simpleNN/nodeInside.png">
		<div class="figure_caption">
			<font color="blue">Figure 4</font>: The insides of a hidden layer node, $j$ including the bias term.
		</div>
	</div>
</div>

<h2 id="backPropagationAlgorithm">7. Back Propagation - the algorithm</h2>

[To contents][100]

Now we have all of the pieces! We've got the initial outputs after our feed-forward, we have the equations for the delta terms (the amount by which the error is based on the different weights) and we know we need to update our bias term too. So what does it look like:

1. Input the data into the network and feed-forward
2. For each of the _output_ nodes calculate:

	<div>$$
	\delta_{k} = \mathcal{O}_{k}  \left( 1 - \mathcal{O}_{k}  \right)  \left( \mathcal{O}_{k} - t_{k} \right)
	$$</div>

3. For each of the _hidden layer_ nodes calculate:

	<div>$$
	\delta_{j} = \mathcal{O}_{i} \left( 1 - \mathcal{O}_{j} \right)   \sum_{k \in K} \delta_{k} W_{jk}
	$$</div>
	
4. Calculate the changes that need to be made to the weights and bias terms:

	<div>$$
	\begin{align}
	\Delta W &= -\eta \ \delta_{l} \ \mathcal{O}_{l-1} \\
	\Delta\theta &= -\eta \ \delta_{l}
	\end{align}
	$$</div>
	
5. Update the weights and biases across the network:

	<div>$$
	\begin{align}
	W + \Delta W &\rightarrow W \\
	\theta + \Delta\theta &\rightarrow \theta
	\end{align}
	$$</div>
	
Here, $\eta$ is just a small number that limit the size of the deltas that we compute: we don't want the network jumping around everywhere. The $l$ subscript denotes the deltas and output for that layer $l$. That is, we compute the delta for each of the nodes in a layer and vectorise them. Thus we can compute the element-wise product with the output values of the previous layer and get our update $\Delta W$ for the weights of the current later. Similarly with the bias term.

This algorithm is looped over and over until the error between the output and the target values is below some set threshold. Depending on the size of the network i.e. the number of layers and number of nodes per layer, it can take a long time to complete one 'epoch' or run through of this algorithm.
	
[100]:{{< relref "#toctop" >}}

_Some of the ideas notation in this tutorial comes from the good videos by [Ryan Harris](https://www.youtube.com/playlist?list=PL29C61214F2146796 " NN Videos")_
 