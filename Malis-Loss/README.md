# Malis Loss

## What is this?
This repository implements MALIS loss for 2D and 3D data in Tensorflow/Keras and Pytorch:

SC Turaga, KL Briggman, M Helmstaedter, W Denk, HS Seung (2009). *Maximin learning of image segmentation*. _Advances in Neural Information Processing Systems (NIPS) 2009_.

http://papers.nips.cc/paper/3887-maximin-affinity-learning-of-image-segmentation

## Performance of malis loss:
![image](https://github.com/HelmholtzAI-Consultants-Munich/Malis-Loss/blob/master/README_files/result.png)

This figure compares the performance of malis loss and cross entropy loss for segmenting mitochondrion. The two networks (UNet, please check /example/keras_example.py) were trained totally same except for the loss. The evaluation criteria (values shown on the figure) for segmentation is based on counting the number of correctly segmented mitochondria (Dice coefficient > 70 %), divided by the average of total number of ground-truth mitochondria and that of automatic segmented mitochondria (similar to the definition of the Dice coefficient, but number-based rather than pixel-based). The final average scores for these three cases are summarized in this table:


cases| cross entropy loss  | malis loss |
-------| ------------- | ------------- |
Image 1 | 0.57  | 0.71  |
Image 2 | 0.68  | 0.80  |
Image 3 | 0.39  | 0.77  |

It can be shown that for this criteria, malis loss can achieved better performance than cross entropy loss. Malis loss performs better in separating nearby mitochondria and generating more clear boundaries, especially when the input data has many adjacent mitochondria (eg. Image 3).


## Installation:
```
./make.sh            (Building c++ extension only: run inside directory)
pip install .        (Installation as python package: run inside directory)
```


### Installation example in anaconda:
```
conda create -n malis python=3.7
conda activate malis
conda install cython numpy gxx_linux-64
conda install -c anaconda boost       or conda install -c conda-forge boost
./make.sh
pip install .
```

## Example Usage:
See keras_example and pytorch_example in example folder for further detailed training(with UNet) examples.
Please install pytorch or tensorflow/Keras according to your needs, eg.:
```
conda install -c pytorch pytorch                or
conda install -c anaconda tensorflow-gpu,keras
```
**To use malis loss, the groundtruth should be processed so that every connected region in it has a different instance number.**
One easy way to achieve it is using skimage:
```
from skimage.measure import label
gt = label(gt)
```
### Using Keras/Tensorflow (channel last):

#### 2D usage
```
import malis as m
from malis.malis_keras import malis_loss2d

model = ... (set the channel of output layer as 2)
model.compile(optimizer, loss = malis_loss2d)
```

#### 3D usage (please use batch size as 1)
```
import malis as m
from malis.malis_keras import malis_loss3d

model = ... (set the channel of output layer as 3)
model.compile(optimizer, loss = malis_loss3d)
```

### Using Pytorch: 
#### 2D usage
```
import malis
from malis.malis_torch import malis_loss2d
    
loss = malis_loss2d(seg_gt, pred_aff)
```
#### 3D usage (please use batch size as 1)
```
import malis
from malis.malis_torch import malis_loss3d
    
loss = malis_loss3d(seg_gt, pred_aff)
```
### Postprocessing:
The output of network should be affinity graphs, to obtain final segmentation graphs, threshold should be manully selected and than apply affgraph_to_seg functions. An example is like below:
```
import malis as m
import numpy as np

aff = .... # predicted affinity graph from trained model
aff = np.where(aff<threshold,0,1)
nhood = m.mknhood3d(1)[:-1]  # or malis.mknhood3d(1) for 3d data prediction
seg = m.affgraph_to_seg(aff,nhood)
```
More detailed example including automaticaly selecting threshold could be found in /postprocessing/postprocessing.ipynb.
### Useful Functions of malis loss in python:
```
import malis as m
nhood = m.mknhood3d(): Makes neighbourhood structures
aff = m.seg_to_affgraph(seg_gt,nhood): Construct an affinity graph from a segmentation
seg = m.affgraph_to_seg(affinity,nhood)[0]: Obtain a segementation graph from an affinity graph
```

