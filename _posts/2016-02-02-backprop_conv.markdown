---
layout: post
title: Backpropagation through a convolutional layer
date: 2016-02-02T09:05:36+01:00
published: true
---

This post is a follow-up on  the second assignment proposed as part of
the                 Stanford                  CS                 class
[CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/).

The  last  part  of  this  assignement  is  the  implementation  of  a
convolutional neural  network.  Among  other things, this  implies the
computation of  the gradient from the  loss back to the  input through
convolutional layers.

Instead of writing everything on  papers and, undoubtedly, lost myself
in the indices, I decided to write the derivation in this post. 

## Convolutional layer

I am going to follow the notation of the assignment 

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

## Forward pass

The first step  is the implementation of the forward  pass. Instead of
jumping into it, let's first look at  an example; an input $X$ of size
$H=W=5$ convolve  with a  filter of size  $HH=WW=3$ with  stride $S=2$
with  zero padding  ($P=0$). 

Graphically, $x$ looks like

$$
\begin{equation}
x =
\begin{pmatrix}
    x_{00} & x_{01} & x_{02}& x_{03}& x_{04}\\
    x_{10} & x_{11} & x_{12}& x_{13}& x_{14}\\
    x_{20} & x_{21} & x_{22}& x_{23}& x_{24}\\
    x_{30} & x_{31} & x_{32}& x_{33}& x_{34}\\
    x_{40} & x_{41} & x_{42}& x_{43}& x_{44}
\end{pmatrix},
\end{equation}
$$

the filter $w$

$$
\begin{equation}
x =
\begin{pmatrix}
    w_{00} & w_{01} & w_{02} \\
    w_{10} & w_{11} & w_{12} \\
    w_{20} & w_{21} & w_{22}
\end{pmatrix},
\end{equation}
$$

and the bias  is just $b$ as  we consider only one  filter. The output
$y$ is a vector of size $$Hh=Hw=2$$ as $Hh$ is given by

$$ Hh = 1 + (H + 2P - HH) / S.$$

First,  let's look  specically  at $y_{11}$.   During the  convolution
process, this term  resumes to the multiplication  element-wise of the
lower right part of $x$ with the filter. Mathematically, it reads

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
(given by  $P$).  With  $N$ images,  $F$ filters  and $C$  channels, the
example  above  easily  generalized  and  the  forward  pass  for  the
convolutional layer reads

$$
\begin{eqnarray}
y_{n,f,k,l}            &=&            \sum_c\sum_{p=p_0}^{p=p_0+\Delta
p}\sum_{q=q_0}^{q=q_0+\Delta q} x^{pad}_{n,c,p,q} w_{f,c,p-p_0,q-q_0}+b_f\\
&=&       \sum_c\sum_{p=0}^{\Delta       p}\sum_{q=0}^{\Delta       q}
x^{pad}_{n,c,p+p_0,q+q_0}w_{f,c,p,q}+b_f\\
&=& \sum_c\sum_{p=0}^{HH}\sum_{q=0}^{WW} x^{pad}_{n,c,p+kS,q+lS}w_{f,c,p,q}+b_f
\end{eqnarray}
$$

which  nicely translates  mathematically the  fact that  to obtain  the
output  at a  given position  $(n,f,k,l)$, select  the subpart  of the
image of  size $$(HH,WW)$$, multiply it  by the filter and  sum all the
resulting terms, i.e. the convolution!

In python, it looks like

{% highlight python %}
out = np.zeros((N, F, Hh, Hw))
for n in range(N):  # First, iterate over all the images
    for f in range(F):  # Second, iterate over all the kernels
        for k in range(Hh):
            for l in range(Hw):
                out[n, f, k, l] = np.sum(x_pad[n, :, k * S:k * S + HH, l * S:l * S + WW] * w[f, :]) + b[f]
{% endhighlight %}

## Backward pass

During the backward pass, we have to compute $$\frac{d\mathcal{L}}{dw},
\frac{d\mathcal{L}}{dx},\frac{d\mathcal{L}}{db}$$    where    each
gradient with respect to a quantity contains a vector of size equal to
the  quantity  itself  and  where  we  know  from  the  previous  pass
$$\frac{d\mathcal{L}}{dy}$$                (see               previous
[post](http://cthorey.github.io/backpropagation/) for more details).

### Gradient with respect to the weights  $$\frac{d\mathcal{L}}{dw}$$

The gradient of the loss with respect to the weights has the same size
as  the  weights  themselves   ($F$,$C$,$HH$,$WW$).  Chaining  by  the
gradient of the loss with respect to the outputs $y$, it reads

$$
\begin{eqnarray}
\frac{d\mathcal{L}}{dw_{f',c',i,j}} = \sum_{n,f,k,l}\frac{d\mathcal{L}}{dy_{n,f,k,l}}\frac{dy_{n,f,k,l}}{dw_{f',c',i,j}}.
\end{eqnarray}
$$

The expression  of $y_{n,f,k,l}$ is  derived above and  therefore, its
derivative with respect to the weight $w$ reads

$$
\begin{eqnarray}
\frac{dy_{n,f,k,l}}{dw_{f',c',i,j}} &=&           \frac{d}{dw_{f',c',i,j}}\left(\sum_c\sum_{p=0}^{p=\Delta
p}\sum_{q=0}^{q=\Delta                                              q}
x^{pad}_{n,c,p+p_0,q+q_0}w_{f,c,p,q}\right)\\
&=&      \sum_c\sum_{p=0}^{p=\Delta     p}\sum_{q=0}^{q=\Delta      q}
\delta_{c,c'}\delta_{f,f'}\delta_{p,i}\delta_{q,j}x^{pad}_{n,c,p+p_0,q+q_0}\\
&=& \delta_{f,f'}x^{pad}_{n,c',i+Sk,j+Sl}
\end{eqnarray}
$$

Injecting  this expression  back into  the gradient  of the  loss with
respect to the weights, we then have

$$
\begin{eqnarray}
\frac{d\mathcal{L}}{dw_{f',c',i,j}}                                  &=&
\sum_{n,f,k,l}\frac{d\mathcal{L}}{dy_{n,f,k,l}}\delta_{f,f'}x^{pad}_{n,c',i+Sk,j+Sl}\\
&=& \sum_{n,k,l}\frac{d\mathcal{L}}{dy_{n,f',k,l}}x^{pad}_{n,c',i+Sk,j+Sl}
\end{eqnarray}
$$

which in python translates

{% highlight python %}
dw = np.zeros((F, C, HH, WW))
for fprime in range(F):
    for cprime in range(C):
        for i in range(HH):
            for j in range(WW):
                sub_xpad = x_pad[:, cprime, i:i + Hh * S:S, j:j + Hw * S:S]
                dw[fprime, cprime, i, j] = np.sum(dout[:, fprime, :, :] * sub_xpad)
{% endhighlight %}

### Gradient with respect to the bias $$\frac{d\mathcal{L}}{db}$$

The gradient of the loss with respect to the bias is of size ($F$).  Chaining by the
gradient of the loss with respect to the outputs $y$ and simplifying, it reads

$$
\begin{eqnarray}
\frac{d\mathcal{L}}{db_{f'}}                                       &=&
\sum_{n,f,k,l}\frac{d\mathcal{L}}{dy_{n,f,k,l}}\frac{dy_{n,f,k,l}}{db_{f'}}\\
&=& \sum_{n,f,k,l}\frac{d\mathcal{L}}{dy_{n,f,k,l}}\delta_{f,f'}\\
&=& \sum_{n,k,l}\frac{d\mathcal{L}}{dy_{n,f',k,l}}
\end{eqnarray}
$$

which, in python, translates

{% highlight python %}
db = np.zeros((F))
for fprime in range(F):
    db[fprime] = np.sum(dout[:, fprime, :, :])
{% endhighlight %}




### Gradient with respect to the input $$\frac{d\mathcal{L}}{dx}$$

As above, we first  chain by the gradient of the  loss with respect to
the output $y$, which gives

$$
\begin{eqnarray}
\frac{d\mathcal{L}}{dx_{n',c',i,j}}                                  &=&
\sum_{n,f,k,l}\frac{d\mathcal{L}}{dy_{n,f,k,l}}\frac{dy_{n,f,k,l}}{dx_{n',c',i,j}}
\end{eqnarray}
$$

The second term reads

$$
\begin{eqnarray}
\frac{dy_{n,f,k,l}}{dx_{n',c',i,j}} &=& \frac{d}{dx_{n',c',i,j}}\left(\sum_c\sum_{p=0}^{HH}\sum_{q=0}^{WW} x^{pad}_{n,c,p+kS,q+lS}w_{f,c,p,q}+b_f\right)
\end{eqnarray}
$$

where we  have to handle  carefully the fact  that $y$ depends  on the
padded version of $x^{pad}$ and not $x$ itself. In other words, we better first
chain with the  gradient of $y$ with respect to  the padded version of
$x^{pad}$ to get

$$
\begin{eqnarray}
\frac{dy_{n,f,k,l}}{dx_{n',c',i,j}}                                &=&
\sum_{n",c",i",j"}\frac{dy_{n,f,k,l}}{dx^{pad}_{n",c",i",j"}}\frac{dx^{pad}_{n",c",i",j"}}{dx_{n',c',i,j}}.
\end{eqnarray}
$$

Let's first  look at the second  term, the gradient of  $x^{pad}$ with
respect to $x$.  There is a simple relationship between  the two which
reads

$$
\begin{equation}
x^{pad}_{n",x",i",j"} = x_{n",x",i"-P,j"-P} \mathbb{1}(P \le i" \le H-P)\mathbb{1}(P \le j" \le W-P)
\end{equation}
$$

then, 

$$
\begin{equation}
\frac{dx^{pad}_{n",c",i",j"}}{dx_{n',c',i,j}} = \delta_{n",n'}\delta_{c",c'}\delta_{i,i"-P}\delta_{j,j"-P}
\end{equation}
$$

For the first term,

$$
\begin{eqnarray}
\frac{dy_{n,f,k,l}}{dx^{pad}_{n",c",i",j"}}                        &=&
\frac{d}{dx^{pad}_{n",c",i",j"}}\left(\sum_c \sum_{p=0}^{HH}\sum_{q=0}^{WW} x^{pad}_{n,c,p+kS,q+lS}w_{f,c,p,q}+b_f\right)\\
&=&  \sum_c \sum_{p=0}^{HH}\sum_{q=0}^{WW}
\delta_{n,n"}\delta_{c,c"}\delta_{p+kS,i"}\delta_{q+lS,j"}w_{f,c,p,q}\\
&=&  \sum_{p=0}^{HH}\sum_{q=0}^{WW}
\delta_{n,n"}\delta_{p+kS,i"}\delta_{q+lS,j"}w_{f,c",p,q}\\
\end{eqnarray}
$$

Then, putting both terms together, we find that

$$
\begin{eqnarray}
\frac{dy_{n,f,k,l}}{dx_{n',c',i,j}}                                &=&
\sum_{n",c",i",j"}\frac{dy_{n,f,k,l}}{dx^{pad}_{n",c",i",j"}}\frac{dx^{pad}_{n",c",i",j"}}{dx_{n',c',i,j}}\\
&=&
\delta_{n,n'}\sum_{i",j"}  \sum_{p=0}^{HH}\sum_{q=0}^{WW}
\delta_{p+kS,i"}\delta_{i,i"-P}\delta_{j,j"-P}
\delta_{q+lS,j"}w_{f,c',p,q}\\
&=&
\delta_{n,n'} \sum_{p=0}^{HH}\sum_{q=0}^{WW}
\delta_{p+kS,i+P}
\delta_{q+lS,j+P}w_{f,c',p,q}
\end{eqnarray}
$$

Hence, the gradient of the loss with respect to the inputs finnally reads

$$
\begin{eqnarray}
\frac{d\mathcal{L}}{dx_{n',c',i,j}}                                  &=&
\sum_{n,f,k,l}\frac{d\mathcal{L}}{dy_{n,f,k,l}}\frac{dy_{n,f,k,l}}{dx_{n',c',i,j}}\\
&=&\sum_{f,k,l} \sum_{p=0}^{HH}\sum_{q=0}^{WW} \frac{d\mathcal{L}}{dy_{n',f,k,l}}
\delta_{p+kS,i+P} \delta_{q+lS,j+P}w_{f,c',p,q} \\
\end{eqnarray}
$$

which  in  python can  be  written  with  a  nice intrication  of  $9$
beautifull loops ;)

{% highlight python %}
# For dx : Size (N,C,H,W)
dx = np.zeros((N, C, H, W))
for nprime in range(N):
    for cprime in range(C):
        for i in range(H):
            for j in range(W):
                for f in range(F):
                    for k in range(Hh):
                        for l in range(Hw):
                            for p in range(HH):
                                for q in range(WW):
                                    if (p + k * S == i + P) & (q + S * l == j + P):
                                        dx[nprime, cprime, i, j] += dout[nprime,
                                                                             f, k, l] *
                                                                             w[f, cprime, p, q]

{% endhighlight %}

I am sure  this $9$ loops are  not strickly necessary, but,  we are ask
not to care to much about efficiency, so, why bother ... :)


