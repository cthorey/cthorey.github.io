---
layout: post
title: Thoughts on backpropagation
date: 2016-01-28T20:35:04+01:00
---

This  past week,  I  have been  working on  the  assignments from  the
Stanford                            CS                           class
[CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/). In
particular,  I spent  a few  hours  deriving a  correct expression  to
backpropagate          the           batchnorm          regularization
([Assigment 2 - Batch Normalization](http://cs231n.github.io/assignments2016/assignment2/))
. While this post is mainly for me not to forget about what insights I
have gained  in solving this  problem, I hope  it could be  usefull to
others that are struggling with back propagation.

Batch normalization  is a recent  idea introduced  by [1] to  ease the
training of large networks. The idea behind it is that neural networks
tends  to   learn  better  when   their  input  features   consist  of
uncorrelated features with zero mean  and unit variance. As each layer
with a  neural network see  the activations  of the previous  layer as
input, the same idea could be apply to each layer. Batch normalization
does  exactly that  by normalizing  the activations  over the  current
batch in each hidden layer, generally right before the non-linearity.

To be more  specific, for a given input batch  **x** of size **(N,D)**
going through  a hidden layer  of size H,  some weights **w**  of size
**(D,H)** and a bias **b** of size **(H,)**, the structure looks like

1. Affine  transformation $$h  =  WX+b$$  where **h**  contains  the
   results of the linear transformation (size **(N,H)**).
2. Batch  normalization transform  $$y = \gamma  \hat{h}+\beta$$ where
   $\gamma$ and $\beta$ are learnable parameters, **y** contains the
   zero mean  and unit  variance version of  **h** (size  **(N,H)**) and
   where $$ \hat{h}= (h-\mu)(\sigma^2+\epsilon)^{-1/2}$$ is the heart of
   batch normalization.
3. Non-linearity activation, say ReLu for our example $$a = ReLu(y)$$
   which now see  a zero mean and unit variance  input and where **a**
   contains the activations of size **(N,H)**.

Implementing the forward pass of the batch norm transformation is straightforward

~~~ python
# Forward pass 
mu = 1/N*np.sum(h,axis =0) # Size (H,) 
sigma2 = 1/N*np.sum((h-mu)**2,axis=0)# Size (H,) 
hath = (h-mu)*(sigma2+epsilon)**(-1./2.)
y = gamma*hath+beta 
~~~


The tricky
part comes with  the backward pass. As the  assignment proposes, there
are two strategies to implement it.

1. Write  out a  computation graph composed  of simple  operations and
   backprop through all intermediate values
2.  Work out the derivatives on paper. 

The 2nd step made me realise I did not fully understood backprogation before this assignment. Backpropation, an abbreviation for "backward propagation of errors", calculates the gradient of a loss function **L** with respect to all the parameters of the network. In our case, we need to calculate the gradient with respect to $\gamma$, $\beta$ and the input $h$.

Mathematically, this reads
$$\frac{dL}{d\gamma}, \frac{dL}{d\beta},\frac{dL}{dh}$$ where each gradient with respect to a quantity contains a vector of size equal to the quantity itself. For instance, the gradient with respect to the input $h$ literally reads

$$
\begin{equation}
\frac{dL}{dh} =
\begin{pmatrix}
   \frac{dL}{dh_{11}} & .. & \frac{dL}{dh_{1H}} \\
   .. & \frac{dL}{dh_{il}} & .. \\
   \frac{dL}{dh_{N1}} & ... & \frac{dL}{dh_{NH}}
\end{pmatrix}.
\end{equation}
$$

To derive a close form expression for those expressions, we first have to recall that the main idea behind backpropagation is chain rule. Indeed, thanks to the backward pass into ReLu, we already know 

$$
\begin{equation}
\frac{dL}{dy} =
\begin{pmatrix}
   \frac{dL}{dy_{11}} & ... & \frac{dL}{dy_{1H}} \\
   ... & \frac{dL}{dy_{kl}} & ... \\
   \frac{dL}{dy_{N1}} & ... & \frac{dL}{dy_{NH}} \\
\end{pmatrix}.
\end{equation}
$$

where 
$$ y_{kl} = \gamma_l \hat{h}_{kl}+\beta_l$$
Applying chain rule to calculate the gradient with respect to $\gamma$ thus gives

$$
\begin{eqnarray}
\frac{dL}{d\gamma_{j}} &=& \sum_{k,l}\frac{dL}{dy_{kl}}\frac{dy_{kl}}{d\gamma_{j}}\\
&=& \sum_{k,l}\frac{dL}{dy_{kl}}\hat{h}_{kl}\delta_{j,l}\\
&=& \sum_{k}\frac{dL}{dy_{kj}}\hat{h}_{kj}
\end{eqnarray}
$$

which translates in one line of python to

~~~
dgamma = np.sum(dy*hath,axis=0)
~~~
{: .python}


For me, the aha-moment came when I realised that I missed the sum over all the output terms when chaining by $y$. In particular desperately trying

~~~
dgamma = dy*hath
~~~
{: .python}

definitely not worked ! For the gradient with respect to $\beta$, easy

$$
\begin{eqnarray}
\frac{dL}{d\beta_{j}} &=& \sum_{k,l}\frac{dL}{dy_{kl}}\frac{dy_{kl}}{d\beta_{j}}\\
&=& \sum_{k,l}\frac{dL}{dy_{kl}}\delta_{j,l}\\
&=& \sum_{k}\frac{dL}{dy_{kj}}
\end{eqnarray}
$$
