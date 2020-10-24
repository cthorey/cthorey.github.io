---
layout: page
title: geocolab
description: Naviguate the program of the AGU fall meeting. 
img: 
importance: 3
---

American Geophysical  Union (AGU)  meeting is a  geoscience conference
held  every year  in December  in  San Francisco.  With nearly  24,000
attendees, AGU  Fall Meeting  is the largest  Earth and  Space science
meeting in  the world.  As such,  it represents  an appealing  tool to
explore the critical trends in the geoscience academic world.

I went to AGU every year since the beginning of my PhD. As a young PhD
student, I used  to find difficult to explore  which presentations and
which    subjects     to    attend    given    the     thousands    of
contributions. Therefore, I decided to  develop a tool which would aim
both at facilitating the navigation within the conference program, and
favoring collaboration between researchers based on the content of the
conference itself. Especially for newcomers  in the field, it could be
very  nice if  such tool  existed, simply  to identify  who and  where
people are working on a specific thematic.

You  can take  a look  at the  current version  of the  recommendation
system through binder 

[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org/repo/cthorey/geocolab)

The model resumes to a tf-idf normalized Bag of Word model of the
abstract corpus whose dimension has  been further reduced to 500 using
Latent Semantic Analysis (LSA). The results are very promising! 

Detailed       of       the       implementation       are       given
[here](http://cthorey.github.io/AGU_Part3/).
