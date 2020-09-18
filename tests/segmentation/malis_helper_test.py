import numpy as np
from instantdl.utils import *

def test_object_segmentation():
    seg = np.ones((128,128,1))
    seg[50:60,50:60,0] = 0
    gt = np.ones((128,128))
    
    score = object_segmentation(seg,gt,2)
    assert score == 1.0
    
    
def test_find_threshold():
    seg = [np.ones((1,128,128,1))]
    seg[0][0,50:60,50:60,0] = 0
    gt = [np.ones((128,128))]
        
    opt_threshold, result, output_seg = find_threshold(seg,gt,2)
    assert opt_threshold == 0.1
    assert result == 1.0
    
    
  