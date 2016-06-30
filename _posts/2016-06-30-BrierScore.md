---
layout: post
title: Implementation of the grad and hess for brier score?
date: 2016-06-30T20:35:04+01:00
published: true
---

# Loss function for the Brier score

The loss function looks like

$$\
\begin{equation}
\mathcal{L} = \sum_i \sum_j \left(\sigma(\hat{y}_{ij})-y_{ij}\right)^2w_{ij}
\end{equation}
$$

with

$$ \sigma_{ij} = \frac{e^{\hat{y}_{ij}}}{\sum_{n} e^{\hat{y}_{in}}}$$

where $\hat{y_{ij}}$  is the  target variable  and $y_{ij}$  the given
proba for example $i$ class $j$

# Gradient

## Analytics

Let's compute this thing.

First let's look at $\sigma$.

$$
\begin{eqnarray}
\frac{\partial      \sigma(\hat{y}_{ij})}{\partial      \hat{y}_{kl}}&=&
\delta_{ik}\delta_{jl}\sigma(\hat{y}_{ij})-\delta_{ik}\sigma(\hat{y}_{ij})\sigma(\hat{y}_{il})\\
&=&\delta_{ik}\sigma(\hat{y}_{ij})\left(\delta_{jl}-\sigma(\hat{y}_{il})\right)
\end{eqnarray}
$$

Ok. What about the whole thing ?

$$
\begin{eqnarray}
\frac{\partial  \mathcal{L}}{\partial   \hat{y}_{kl}}&=&\sum_i  \sum_j
2\left(\delta_{ik}\sigma(\hat{y}_{ij})\left(\delta_{jl}-\sigma(\hat{y}_{il})\right)\right)
\left(\sigma(\hat{y}_{ij})-y_{ij}\right)w_{ij}\\
&=&\sum_i  \sum_j2\delta_{ik}\delta_{jl}\sigma(\hat{y}_{ij})\left(\sigma(\hat{y}_{ij})-y_{ij}\right)w_{ij}-\sum_i  \sum_j
2\delta_{ik}\sigma(\hat{y}_{ij})\sigma(\hat{y}_{il})
\left(\sigma(\hat{y}_{ij})-y_{ij}\right)w_{ij}\\
&=&2\sigma(\hat{y}_{kl})\left(\sigma(\hat{y}_{kl})-y_{kl}\right)w_{kl}- \sum_j
2\sigma(\hat{y}_{kj})\sigma(\hat{y}_{kl})
\left(\sigma(\hat{y}_{kj})-y_{kj}\right)w_{kj}\\
&=&2\sigma(\hat{y}_{kl})\left(\left(\sigma(\hat{y}_{kl})-y_{kl}\right)w_{kl}
- \sum_j \sigma(\hat{y}_{kj})\left(\sigma(\hat{y}_{kj})-y_{kj}\right)w_{kj}\right)
\end{eqnarray}
$$

## Python implementation
{% highlight python %}
def grads(preds,dX):
    weights = dX.weigths
    labels = dX.labels
    preds = np.asarray(softmax(preds))
    tmp = np.sum(preds*(preds-labels)*weights,axis=1)[:,np.newaxis]
    grads = 2*preds*((preds-labels)*weights-tmp)
    return grads
{% endhighlight %}

# Hessian

## Analytics

Let's compute this thing.

First let's look at $\sigma$.

$$
\begin{eqnarray}
\frac{\partial  \sigma(\hat{y}_{kl})}{\partial   \hat{y}_{kl}}&=&\sigma(\hat{y}_{kl})\left(1-\sigma(\hat{y}_{kl})\right)
\end{eqnarray}
$$

and

$$
\begin{eqnarray}
\frac{\partial  \sigma(\hat{y}_{kj})}{\partial   \hat{y}_{kl}}&=&\sigma(\hat{y}_{kj})\left(\delta_{jl}-\sigma(\hat{y}_{kl})\right)
\end{eqnarray}
$$

$$
\begin{eqnarray}
\frac{\partial^2  \mathcal{L}}{\partial   \hat{y}_{kl}^2}&=&2\sigma(\hat{y}_{kl})\left(1-\sigma(\hat{y}_{kl})\right)\left(\left(\sigma(\hat{y}_{kl})-y_{kl}\right)w_{kl}
-                                                               \sum_j
  \sigma(\hat{y}_{kj})\left(\sigma(\hat{y}_{kj})-y_{kj}\right)w_{kj}\right)\\
  &+&     2\sigma(\hat{y}_{kl})\frac{\partial  }{\partial
  \hat{y}_{kl}}\left(\left(\sigma(\hat{y}_{kl})-y_{kl}\right)w_{kl}\right)\\
    &-&     2\sigma(\hat{y}_{kl})\frac{\partial}{\partial
  \hat{y}_{kl}}\left(\sum_j
  \sigma(\hat{y}_{kj})\left(\sigma(\hat{y}_{kj})-y_{kj}\right)w_{kj}\right)\\
  &=&2\sigma(\hat{y}_{kl})\left(1-\sigma(\hat{y}_{kl})\right)\left(\left(\sigma(\hat{y}_{kl})-y_{kl}\right)w_{kl}
-                                                               \sum_j
  \sigma(\hat{y}_{kj})\left(\sigma(\hat{y}_{kj})-y_{kj}\right)w_{kj}\right)\\
  &+&     2\sigma(\hat{y}_{kl})w_{kl}\sigma(\hat{y}_{kl})\left(1-\sigma(\hat{y}_{kl})\right)\\
    &-&     2\sigma(\hat{y}_{kl})\frac{\partial }{\partial
  \hat{y}_{kl}}\left(\sum_j
  \sigma(\hat{y}_{kj})\left(\sigma(\hat{y}_{kj})-y_{kj}\right)w_{kj}\right)\\
\end{eqnarray}
$$


$$
\begin{eqnarray}
\frac{\partial }{\partial
  \hat{y}_{kl}}\left(\sum_j
  \sigma(\hat{y}_{kj})\left(\sigma(\hat{y}_{kj})-y_{kj}\right)w_{kj}\right)&=&\sum_j
  \sigma(\hat{y}_{kj})\left(\delta_{jl}-\sigma(\hat{y}_{kl})\right)\left(\sigma(\hat{y}_{kj})-y_{kj}\right)w_{kj}\\
  &+&\sum_j\sigma(\hat{y}_{kj})\sigma(\hat{y}_{kj})\left(\delta_{jl}-\sigma(\hat{y}_{kl})\right)w_{kj}\\
  &=&\sum_j
  \sigma(\hat{y}_{kj})\delta_{jl}\left(\sigma(\hat{y}_{kj})-y_{kj}\right)w_{kj}-\sum_j
  \sigma(\hat{y}_{kj})\sigma(\hat{y}_{kl})\left(\sigma(\hat{y}_{kj})-y_{kj}\right)w_{kj}\\\\
  &+&\sum_j\sigma(\hat{y}_{kj})\sigma(\hat{y}_{kj})\delta_{jl}w_{kj}
  -\sum_j\sigma(\hat{y}_{kj})\sigma(\hat{y}_{kj})\sigma(\hat{y}_{kl})w_{kj}\\
    &=&
  \sigma(\hat{y}_{kl})\left(\sigma(\hat{y}_{kl})-y_{kl}\right)w_{kl}-\sigma(\hat{y}_{kl})\sum_j
  \sigma(\hat{y}_{kj})\left(\sigma(\hat{y}_{kj})-y_{kj}\right)w_{kj}\\\\
  &+&\sigma(\hat{y}_{kl})\sigma(\hat{y}_{kl})w_{kl}
  -\sigma(\hat{y}_{kl})\sum_j\sigma(\hat{y}_{kj})\sigma(\hat{y}_{kj})w_{kj}\\
      &=&
  \sigma(\hat{y}_{kl})\left(w_{kl}\left(2\sigma(\hat{y}_{kl})-y_{kl}\right)
  -\sum_j w_{kj} \sigma(\hat{y}_{kl})\left(2\sigma(\hat{y}_{kj})-y_{kj}\right) \right)
\end{eqnarray}
$$

Then

$$
\begin{eqnarray}
\frac{\partial^2  \mathcal{L}}{\partial   \hat{y}_{kl}^2}&=&2\sigma(\hat{y}_{kl})\left(1-\sigma(\hat{y}_{kl})\right)\left(\left(\sigma(\hat{y}_{kl})-y_{kl}\right)w_{kl}
-                                                               \sum_j
  \sigma(\hat{y}_{kj})\left(\sigma(\hat{y}_{kj})-y_{kj}\right)w_{kj}\right)\\
  &+&     2\sigma(\hat{y}_{kl})w_{kl}\sigma(\hat{y}_{kl})\left(1-\sigma(\hat{y}_{kl})\right)\\
    &-&      \sigma(\hat{y}_{kl})\left(w_{kl}\left(2\sigma(\hat{y}_{kl})-y_{kl}\right)
  -\sum_j w_{kj} \sigma(\hat{y}_{kl})\left(2\sigma(\hat{y}_{kj})-y_{kj}\right) \right)
\end{eqnarray}
$$
