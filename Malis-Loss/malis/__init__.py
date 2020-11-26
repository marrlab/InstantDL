import numpy as np
from .wrappers import malis_weights, mknhood3d, seg_to_affgraph,affgraph_to_seg


## using keras: from malis.malis_keras import malis_loss:   loss = malis_loss(seg_gt,aff_pred)
## using pytorch: from malis.malis_torch import malis_loss: loss = malis_loss(aff_pred,seg_gt)