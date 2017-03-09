+++
date = "2017-03-08T10:43:07Z"
title = "Transfer Functions"
description = "An insight into various activation functions"
topics = ["tutorials"]
tags = ["neural network", "transfer", "activation", "gaussian", "sigmoid", "linear", "tanh"]
social=true
+++

As promised in the previous post, we'll take a look at some of the different activation functions that could be used in our nodes. Again **please** let me know if there's anything I've gotten totally wrong - I'm very much learning too.

<div id="toctop"></div>

1. [Linear Function][1]
2. [Sigmoid Function][2]
3. [Hyperbolic Tangent Function][3]
4. [Gaussian Function][4]
5. [Heaviside (step) Function][5]
6. [Ramp Function][6]
	1. [Rectified Linear Unit (ReLU)][7]

[1]:{{< relref "#linear" >}}
[2]:{{< relref "#sigmoid" >}}
[3]:{{< relref "#tanh" >}}
[4]:{{< relref "#gaussian" >}}
[5]:{{< relref "#step" >}}
[6]:{{< relref "#ramp" >}}
[7]:{{< relref "#relu" >}}

<h2 id="linear"> Linear (Identity) Function </h2>

[To contents][100]

### What does it look like?

<div  id="fig1" class="figure_container">
		<div class="figure_images">
		<img title="Simple NN" src="/img/transferFunctions/linear.png" width="90%"><img title="Simple NN" src="/img/transferFunctions/dlinear.png" width="90%">
		</div>
		<div class="figure_caption">
			<font color="blue">Figure 1</font>: The linear function (left) and its derivative (right)
		</div>
</div>


### Formulae

<div>$$
f \left( x_{i} \right) = x_{i}
$$</div>

### Python Code

```python
def linear(x, Derivative=False):
    if not Derivative:
        return x
    else:
        return 1.0
```

### Why is it used?

If there's a situation where we want a node to give its output without applying any thresholds, then the identity (or linear) function is the way to go.

Hopefully you can see why it is used in the final output layer nodes as we only want these nodes to do the $ \text{input} \times \text{weight}$ operations before giving us its answer without any further modifications.

<font color="blue">

**Note:** The linear function is not used in the hidden layers. We must use non-linear transfer functions in the hidden layer nodes or else the output will only ever end up being a linearly separable solution.

</font>

<br>

---

<h2 id="sigmoid"> The Sigmoid (or Fermi) Function </h2>

[To contents][100]

### What does it look like?


<div  id="fig2" class="figure_container">
		<div class="figure_images">
		<img title="Simple NN" src="/img/transferFunctions/sigmoid.png" width="90%"><img title="Simple NN" src="/img/transferFunctions/dsigmoid.png" width="90%">
		</div>
		<div class="figure_caption">
			<font color="blue">Figure 2</font>: The sigmoid function (left) and its derivative (right)
		</div>
</div>

### Formulae

<div >$$
f\left(x_{i} \right) = \frac{1}{1 + e^{  - x_{i}  }}, \ \
f^{\prime}\left( x_{i} \right) = \sigma(x_{i}) \left( 1 -  \sigma(x_{i}) \right)
$$</div>

### Python Code

```python
def sigmoid(x,Derivative=False):
    if not Derivative:
        return 1 / (1 + np.exp (-x))
    else:
        out = sigmoid(x)
        return out * (1 - out)
```



### Why is it used?

This function maps the input to a value between 0 and 1 (but not equal to 0 or 1). This means the output from the node will be a high signal (if the input is positive) or a low one (if the input is negative). This function is often chosen as it is one of the easiest to hard-code in terms of its derivative. The simplicity of its derivative allows us to efficiently perform back propagation without using any fancy packages or approximations. The fact that this function is smooth, continuous (differentiable), monotonic and bounded means that back propagation will work well.

The sigmoid's natural threshold is 0.5, meaning that any input that maps to a value above 0.5 will be considered high (or 1) in binary terms.


<br>

---

<h2 id="tanh"> Hyperbolic Tangent Function ( $\tanh(x)$ ) </h2>

[To contents][100]

### What does it look like?

<div  id="fig3" class="figure_container">
		<div class="figure_images">
		<img title="Simple NN" src="/img/transferFunctions/tanh.png" width="90%"><img title="Simple NN" src="/img/transferFunctions/dtanh.png" width="90%">
		</div>
		<div class="figure_caption">
			<font color="blue">Figure 3</font>: The hyperbolic tangent function (left) and its derivative (right)
		</div>
</div>

### Formulae

<div >$$
f\left(x_{i} \right) = \tanh\left(x_{i}\right),
f^{\prime}\left(x_{i} \right) = 1 - \tanh\left(x_{i}\right)^{2}
$$</div>

### Why is it used?

This is a very similar function to the previous sigmoid function and has much of the same properties: even its derivative is straight forward to compute. However, this function allows us to map the input to any value between -1 and 1 (but not inclusive of those). In effect, this allows us to apply a plenalty to the node (negative) rather than just have the node not fire at all. It also gives us a larger range of output to play with in the positive end of the scale meaning finer adjustments can be made.

This function has a natural threshold of 0, meaning that any input which maps to a value greater than 0 is considered high (or 1) in binary terms.

Again, the fact that this function is smooth, continuous (differentiable), monotonic and bounded means that back propagation will work well. The subsequent functions don't all have these properties which makes them more difficult to use in back propagation (though it is done).
<br>

---

## What's the difference between the sigmoid and hyperbolic tangent?

They both achieve a similar mapping, are both continuous, smooth, monotonic and differentiable, but give out different values. For a sigmoid function, a large negative input generates an almost zero output. This lack of output will affect all subsequent weights in the network which may not be desirable - effectively stopping the next nodes from learning. In contrast, the $\tanh$ function supplies -1 for negative values, maintaining the output of the node and allowing subsequent nodes to learn from it.

---

<h2 id="gaussian"> Gaussian Function </h2>

[To contents][100]

### What does it look like?

<div  id="fig4" class="figure_container">
		<div class="figure_images">
		<img title="Simple NN" src="/img/transferFunctions/gaussian.png" width="90%"><img title="Simple NN" src="/img/transferFunctions/dgaussian.png" width="90%">
		</div>
		<div class="figure_caption">
			<font color="blue">Figure 4</font>: The gaussian function (left) and its derivative (right)
		</div>
</div>

### Formulae

<div >$$
f\left( x_{i}\right ) = e^{ -x_{i}^{2}}, \ \
f^{\prime}\left( x_{i}\right ) = - 2x e^{ - x_{i}^{2}}
$$</div>

### Python Code

```python
def gaussian(x, Derivative=False):
    if not Derivative:
        return np.exp(-x**2)
    else:
        return -2 * x * np.exp(-x**2)
```

### Why is it used?

The gaussian function is an even function, thus is gives the same output for equally positive and negative values of input. It gives its maximal output when there is no input and has decreasing output with increasing distance from zero. We can perhaps imagine this function is used in a node where the input feature is less likely to contribute to the final result.

<br>

---

<h2 id="step"> Step (or Heaviside) Function </h2>

[To contents][100]

### What does it look like?

<div  id="fig5" class="figure_container">
		<div class="figure_images">
		<img title="Simple NN" src="/img/transferFunctions/step.png" width="90%">
		</div>
		<div class="figure_caption">
			<font color="blue">Figure 5</font>: The Heaviside function (left) and its derivative (right)
		</div>
</div>

### Formulae

<div>$$
    f(x)= 
\begin{cases}
\begin{align}
    0  \ &: \ x_{i} \leq T\\
    1 \ &: \ x_{i} > T\\
    \end{align}
\end{cases}
$$</div>

### Why is it used?

Some cases call for a function which applies a hard thresold: either the output is precisely a single value, or not. The other functions we've looked at have an intrinsic probablistic output to them i.e. a higher output in decimal format implying a greater probability of being 1 (or a high output). The step function does away with this opting for a definite high or low output depending on some threshold on the input $T$.

However, the step-function is discontinuous and therefore non-differentiable (its derivative is the Dirac-delta function). Therefore use of this function in practice is not done with back-propagation.

<br>

---

<h2 id="ramp"> Ramp Function </h2>

[To contents][100]

### What does it look like?

<div  id="fig6" class="figure_container">
		<div class="figure_images">
		<img title="Simple NN" src="/img/transferFunctions/ramp.png" width="90%"><img title="Simple NN" src="/img/transferFunctions/dramp.png" width="90%">
		</div>
		<div class="figure_caption">
			<font color="blue">Figure 6</font>: The ramp function (left) and its derivative (right) with $T1=-2$ and $T2=3$.
		</div>
</div>


### Formulae

<div>$$
    f(x)= 
\begin{cases}
\begin{align}
    0 \ &: \ x_{i} \leq T_{1}\\[0.5em]
    \frac{\left( x_{i} - T_{1} \right)}{\left( T_{2} - T_{1} \right)} \ &: \ T_{1} \leq x_{i} \leq T_{2}\\[0.5em]
    1 \ &: \ x_{i} > T_{2}\\
    \end{align}
\end{cases}
$$</div>

### Python Code

```python
def ramp(x, Derivative=False, T1=0, T2=np.max(x)):
    out = np.ones(x.shape)
    ids = ((x < T1) | (x > T2))
    if not Derivative:
        out = ((x - T1)/(T2-T1))
        out[(x < T1)] = 0
        out[(x > T2)] = 1
        return out
    else:
        out[ids]=0
        return out
```

### Why is it used?

The ramp function is a truncated version of the linear function. From its shape, the ramp function looks like a more definitive version of the sigmoid function in that its maps a range of inputs to outputs over the range (0 1) but this time with definitive cut off points $T1$ and $T2$. This gives the function the ability to fire the node very definitively above a threshold, but still have some uncertainty in the lower regions. It may not be common to see $T1$ in the negative region unless the ramp is equally distributed about $0$.

<h3 id="relu"> 6.1 Rectified Linear Unit (ReLU) </h3>

There is a popular, special case of the ramp function in use in the powerful _convolutional neural network_ (CNN) architecture called a _**Re**ctifying **L**inear **U**nit_ (ReLU). In a ReLU, $T1=0$ and $T2$ is the maximum of the input giving a linear function with no negative values as below:

<div  id="fig7" class="figure_container">
		<div class="figure_images">
		<img title="Simple NN" src="/img/transferFunctions/relu.png" width="90%"><img title="Simple NN" src="/img/transferFunctions/drelu.png" width="90%">
		</div>
		<div class="figure_caption">
			<font color="blue">Figure 7</font>: The Rectified Linear Unit (ReLU) (left) with its derivative (right).
		</div>
</div>

and in Python:

```python
def relu(x, Derivative=False):
    if not Derivative:
        return np.maximum(0,x)
    else:
        out = np.ones(x.shape)
        out[(x < 0)]=0
        return out
```

[100]:{{< relref "#toctop" >}}
