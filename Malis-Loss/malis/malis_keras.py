import keras.backend as K
import tensorflow as tf
import numpy as np
from .wrappers import malis_weights,mknhood3d,seg_to_affgraph

def pairs_to_loss_keras(pos_pairs, neg_pairs, pred, margin=0.3, pos_loss_weight=0.5):
    """
    Computes MALIS loss weights from given positive and negtive weights.
    
    Roughly speaking the malis weights quantify the impact of an edge in
    the predicted affinity graph on the resulting segmentation.
    
    Input:
        pos_pairs: (batch_size, H, W, C)
           Contains the positive pairs 
        neg_pairs: (batch_size, H, W, C)
           Contains the negative pairs 
        pred:  (batch_size, H, W, C)
            affinity predictions from network
    Returns:
        malis_loss: scale
            final malis loss
        
    """
    pos_t = tf.cast(pos_pairs,dtype=tf.float32)
    pos_t = tf.math.divide_no_nan(pos_t,tf.reduce_sum(pos_t))

    neg_t = tf.cast(neg_pairs,dtype=tf.float32)
    neg_t = tf.math.divide_no_nan(neg_t,tf.reduce_sum(neg_t))
    
    neg_loss_weight = 1 - pos_loss_weight
    zeros_helpervar = tf.zeros(shape=tf.shape(pred))

    pos_loss = tf.where(1 - pred - margin > 0,
                        (1 - pred - margin)**2,
                        zeros_helpervar)
    pos_loss = pos_loss * pos_t
    pos_loss = tf.reduce_sum(pos_loss) * pos_loss_weight

    neg_loss = tf.where(pred - margin > 0,
                        (pred - margin)**2,
                        zeros_helpervar)
    neg_loss = neg_loss * neg_t
    neg_loss = tf.reduce_sum(neg_loss) * neg_loss_weight
    malis_loss = (pos_loss + neg_loss) * 2  # because of the pos_loss_weight and neg_loss_weight

    return malis_loss


def malis_loss2d(y_true,y_pred): 
    '''
    Computes 2d MALIS loss given predicted affinity graphs and segmentation groundtruth
    
    Roughly speaking malis weights (pos_pairs and neg_pairs) quantify the 
    impact of an edge in the predicted affinity graph on the resulting segmentation.
    
    Input:
       y_true: Tensor (batch_size, H, W, C = 1)
          segmentation groundtruth
       y_pred: Tensor (batch_size, H, W, C = 3)
           affinity predictions from network
    Returns:
       loss: Tensor(scale)
              malis loss 
      
    Outline:
    - Computes for all pixel-pairs the MaxiMin-Affinity
    - Separately for pixel-pairs that should/should not be connected
    - Every time an affinity prediction is a MaxiMin-Affinity its weight is
      incremented by one in the output matrix (in different slices depending
      on whether that that pair should/should not be connected)
    '''
    
    #########  make sure seg_true and y_pred has the correct shape -> (H,W,C'), (2,H,W,batch) for each     
    x = K.int_shape(y_pred)[1]  # H
    y = K.int_shape(y_pred)[2]  # W

    seg_true = K.reshape(y_true,(x,y,-1))              # (H,W,D)
    y_pred = K.permute_dimensions(y_pred,(3,1,2,0))   # (C=3,H,W,D)

    #########
    nhood = mknhood3d(1)[:-1]
    nhood = tf.cast(nhood,tf.int32)
    
    gtaff = tf.numpy_function(func = seg_to_affgraph,inp=[seg_true, nhood],
                                             Tout=tf.int16) # get groundtruth affinity
    
    weights_pos,weights_neg = tf.py_function(malis_weights,
                         [y_pred, gtaff, seg_true, nhood],
                         [tf.int32,tf.int32])

    loss = pairs_to_loss_keras(weights_pos, weights_neg, y_pred)
    
    return loss


def malis_loss3d(y_true,y_pred): 
    '''
    Computes 3d MALIS loss given predicted affinity graphs and segmentation groundtruth
    
    Roughly speaking malis weights (pos_pairs and neg_pairs) quantify the 
    impact of an edge in the predicted affinity graph on the resulting segmentation.
    
    Input:
       y_true: Tensor (batch_size=1, H, W, D, C=1)
          segmentation groundtruth
       y_pred: Tensor (batch_size=1, H, W, D, C=4)
           affinity predictions from network
    Returns:
       loss: Tensor(scale)
              malis loss 
    
    Outline:
    - Computes for all pixel-pairs the MaxiMin-Affinity
    - Separately for pixel-pairs that should/should not be connected
    - Every time an affinity prediction is a MaxiMin-Affinity its weight is
      incremented by one in the output matrix (in different slices depending
      on whether that that pair should/should not be connected)
    '''
    
    ######### make sure seg_true and y_pred has the correct shape -> (H,W,D),(3,H,W,D) for each   
    x = K.int_shape(y_pred)[1]  # H
    y = K.int_shape(y_pred)[2]  # W
    z = K.int_shape(y_pred)[3]  # D

    seg_true = K.reshape(y_true,(x,y,z))              # (H,W,D)
    y_pred = K.permute_dimensions(y_pred[0],(3,0,1,2))   # (C=3,H,W,D)

    #########
    nhood = mknhood3d(1)
    nhood = tf.cast(nhood,tf.int32)
    gtaff = tf.numpy_function(func = seg_to_affgraph,inp=[seg_true, nhood],
                                             Tout=tf.int16) # get positive and negtive malis weights

    weights_pos,weights_neg = tf.py_function(malis_weights,
                         [y_pred, gtaff, seg_true, nhood],
                         [tf.int32,tf.int32])

    loss = pairs_to_loss_keras(weights_pos, weights_neg, y_pred)

    return loss
