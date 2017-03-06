+++
title = "Simple Neural Network - Mathematics"
description = "Understanding the maths of Neural Networks"
topics = ["tutorials"]
tags = ["neural network","back propagation", "machine learning"]
draft = true
date = "2017-03-06T17:04:53Z"

+++

Tutorials on neural networks (NN) can be found all over the internet. Though many of them are the same, each is written (or recorded) slightly differently. This means that I always feel like I learn something new or get a better understanding of things with every tutorial I see. I'd like to make this tutorial as clear as I can, so sometimes the maths may be simplistic, but hopefully it'll give you a good unserstanding of what's going on. First though, lets take a look at what a NN looks like.

<h2 id="transferFunction"> Transfer Function </h2>


<div id="eqsigmoidFunction">$$ \sigma ( x ) = \frac{1}{1 + e^{-x}}  $$</div>

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


<div id="eqdsigmoid">$$ \sigma^{\prime}( x ) = \sigma (x ) \left( 1 - \sigma ( x ) \right) $$</div>

<h2 id="error"> Error </h2>

<div id="eqerror">$$ \text{E} = \frac{1}{2} \sum_{k \in K} \left( \mathcal{O}_{k} - t_{k} \right)^{2} $$</div>

<div>$$ \frac{\partial{\text{E}}}{\partial{W_{jk}}} =  \frac{\partial{}}{\partial{W_{jk}}}  \frac{1}{2} \sum_{k \in K} \left( \mathcal{O}_{k} - t_{k} \right)^{2}$$</div>


<h2id="backPropagation"></h2>
<div>$$ $$</div>

<div>$$ $$</div>

<div>$$ $$</div>

<div>$$ $$</div>
