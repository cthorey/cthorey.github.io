---
layout: post
title: Thoughts on backpropagation
date: 2016-01-28T20:35:04+01:00
---

This past week, I have been working on the assignments from the Stanford  CS  class [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/). In particular, I spent a few hours deriving a correct expression to backpropagate the batchnorm regularization ([Assigment 2 - Batch Normalization](http://cs231n.github.io/assignments2016/assignment2/)) . While this post is mainly for me not to forget about what insights I have gained in solving this problem, I hope it could be usefull to others that are struggling with back propagation.

Batch normalization is a recent idea introduced by [1] to ease the training of large networks. The idea behind it is that neural networks tends to learn better when their input features consist of uncorrelated features with zero mean and unit variance. As each layer with a neural network see the activations of the previous layer as input, the same idea could be apply to each layer. Batch normalization does exactly that by normalizing the activations over the current batch in each hidden layer, generally right before the non-linearity. 

To be more specific, given an input batch **x** of size **(N,D)** going through a hidden layer of size H, some weights **w** of size **(D,H)** and a bias **b** of size **(H,)**, the regular structure of each layer looks like:

1. Affine transformation: $$h = WX+b$$ where **h** contains the results of the linear transformation (size **(N,H)**).
2. Non-linearity, say ReLu for our example: $$a = ReLu(h)$$ where **a** contains the activations of size **(N,H)**.

In contrast, with batch normalization, each layer which contain a non-linearity looks like:

1. Affine transformation: $$h = WX+b$$ where **h** contains the results of the linear transformation (size **(N,H)**).
2. Batch normalization transform
 $$y = \hat{h}= (h-\mu)(\sigma^2+\epsilon)^{-1/2}$$ where **y** contains the  zero mean and unit variance version of **h** (size **(N,H)**).
4. Non-linearity activation, say ReLu for our example: $$a = ReLu(h)$$ where **a** contains the activations of size **(N,H)**.

Normally, the normalization is followed by potential rescaling and translation $$y= \gamma \hat{h} +\beta$$ where $\gamma$ and $\beta$ can be learn by the network. The network can then learn to undo the **hat** transformation if it wants to by learning
$$ \gamma  = (\sigma^2+\epsilon)^{-1/2}; \beta = \mu$$ for instance. However, we are going to focus on the **hat** transformation in the following.

The **hat** transformation is at the heart of batch normalization and



