---
layout: post
title: Backpropagation through a convolutional layer
date: 2016-02-02T09:05:36+01:00
published: false
---

In the last  post, I look into the derivation  of the gradient flowing
through a batch normalization. 

## Convolutional layer

We are going to follow the notation of the assignment 

{% highlight python %}
- N, C, H, W = x.shape
- F, C, HH, WW = w.shape
- N, F, Hh, Hw = dout.shape
- S = conv_param['stride']
{% endhighlight %}

In addition, a bunch of indices will be used in the derivation. Unless
differently specifies, I'll use the following

- $n$ for the different images
- $f$ for the different filter
- $c$ for the different channel
- $f,k$ for the spatial ouput


### Forward pass

For  instance, image  an input  $X$ of  size $H=W=5$  convolve with  a
filter  of  size  $HH=WW=3$  with   stride  $S=2$  with  zero  padding
($P=0$). The output $y$ should then be of size $$Hh=Hw=2$$ as

$$ Hh = 1 + (H + 2P - HH) / S$$

In that case, the input $x$ reads

$$
\begin{equation}
x =
\begin{pmatrix}
    x_{00} & x_{01} & x_{02}& x_{03}& x_{04}\\
    x_{10} & x_{11} & x_{12}& x_{13}& x_{14}\\
    x_{20} & x_{21} & x_{22}& x_{23}& x_{24}\\
    x_{30} & x_{31} & x_{32}& x_{33}& x_{34}\\
    x_{40} & x_{41} & x_{42}& x_{43}& x_{44}
\end{pmatrix}.
\end{equation}
$$

,the filter $w$ reads

$$
\begin{equation}
x =
\begin{pmatrix}
    w_{00} & w_{01} & w_{02} \\
    w_{10} & w_{11} & w_{12} \\
    w_{20} & w_{21} & w_{22}
\end{pmatrix}.
\end{equation}
$$

and the bias is just $b$ as we consider only one filter.


First,  let's  look  specically  at  $y_{11}$.  During  the  convolution
process, this term  resumes to the multiplication  element-wise of the
lower right part of $x$ witht the filter. Mathematically, it reads

$$
\begin{eqnarray}
y_{11} &=& x_{22}w_{00}+x_{23}w_{01}+...+x_{43}w_{21}+w_{44}w_{22} \\
       &=& \sum_{p=2}^{4}\sum_{q=2}^{4} x_{p,q} w_{p-p_0,q-q_0}+b
\end{eqnarray}
$$

where the beginning  of the sum is  given by the index  of the element
times the stride $S$, and the size of  the sum is given by the size of
the filter.

Therefore, generalizing from this example, we see that, given a stride
$S$, the output reads

$$
y_{kl}=  \sum_{p=p_0}^{p=p_0+\Delta   p}\sum_{q=q_0}^{q=q_0+\Delta  q}
x^{pad}_{p,q} w_{p-p_0,q-q_0}+b
$$

where

$$
\begin{eqnarray}
p_0 &=& Sk\\
q_0 &=& Sl\\
\Delta p&=& HH\\
\Delta q &=& WW
\end{eqnarray}
$$

and $x^{pad}$  is the input padded  with the adequate number  of zeros
(given by  $P$).  With  $N$ images,  $F$ filter  and $C$  channel, the
example  above  easily  generalized  and  the  forward  pass  for  the
convolutional layer reads

$$
\begin{equation}
y_{n,f,k,l} = \sum_c\sum_{p=p_0}^{p=p_0+\Delta p}\sum_{q=q_0}^{q=q_0+\Delta q} x^{pad}_{n,c,p,q}w_{f,c,p-p_0,q-q_0}+b_f
\end{equation}
$$

In python, it looks like

{% highlight python %}
    out = np.zeros((N, F, Hh, Hw))
    for n in range(N):# First, iterate over all the images
        for f in range(F):# Second, iterate over all the kernels
            # Then we have to coupute for each img the activation
            # resulting from the convolution
            for k in range(Hh):
                for l in range(Hw):
                    p_0 = S * k
                    p_1 = S * k + HH
                    q_0 = S * l
                    q_1 = S * l + WW
                    out[n, f, k, l] = np.sum(
                        x_pad[n, :, p_0:p_1, q_0:q_1] * w[f, :]) + b[f]
{% endhighlight %}

## Backward pass

During the backward pass, we have to compute $$\frac{d\mathcal{L}}{dw},
\frac{d\mathcal{L}}{dx},\frac{d\mathcal{L}}{db}$$    where    each
gradient with respect to a quantity contains a vector of size equal to
the  quantity  itself  and  where  we  know  from  the  previous  pass
$$\frac{d\mathcal{L}}{dy}$$                (see               previous
[post](http://cthorey.github.io/backpropagation/) for more details).

### Gradient with respect to the weights  $$\frac{d\mathcal{L}}{dw}$$
















