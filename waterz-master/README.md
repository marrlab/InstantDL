# waterz

Pronounced *water-zed*. A simple watershed and region agglomeration library for
affinity graphs.

Based on the watershed implementation of [Aleksandar Zlateski](https://bitbucket.org/poozh/watershed) and [Chandan Singh](https://github.com/TuragaLab/zwatershed).

# Installation

`python setup.py install`

Requires `numpy` and `cython`.

# Usage

```
import waterz
import numpy as np

# affinities is a [3,depth,height,width] numpy array of float32
affinities = ...

thresholds = [0, 100, 200]

segmentations = waterz.agglomerate(affinities, thresholds)
```
