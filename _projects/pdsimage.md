---
layout: page
title: pdsimage
description: Plotting image from the surface of the Moon.
img: /assets/img/Screenshot from 2020-10-24 17-23-41.png
importance: 1
---

The Lunar Reconnaissance Orbiter ([LRO](https://github.com/cthorey/pdsimage)) mission has gathered some fantastic data of the lunar surface and, thanks to NASA, all are available for free on the internet.

However, using them can be criptic without a bit of experience. [pdsimage](https://pdsimage.readthedocs.io/en/master/index.html) aims to facilitate the extraction of knowledge from two set of data gathered by two experiments on board of LRO:

- The Lunar Orbiter Laser Altimer ([LOLA](http://lunar.gsfc.nasa.gov/lola/)) experiment which collected high resolution topographic data of the lunar surface 
- The Lunar Reconnaissance Orbiter Camera ([LROC](http://lroc.sese.asu.edu/about)) experiment which collected images of the lunar surface. In particular, we focus on Wide Angle Images taken from the Wide Angle Camera ([WAC](http://lroc.sese.asu.edu/images))

Unlike previous python PDS library, this module allows to easily extract data for arbitrarily sized region at the lunar surface. It is particularly suited for studying geological unit at the surface
of the Moon. It also includes some extension to ease the study of lunar impact craters and low-slope domes.

[FULL DOCUMENTATION](http://pdsimage.readthedocs.org/)
