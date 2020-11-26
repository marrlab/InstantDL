import torch
from torch.autograd import Function
import numpy as np
from .wrappers import malis_weights,mknhood3d,seg_to_affgraph

class torchloss(Function):
    '''
    Wrap a numpy function (get_pairs) into pytorch graph 
    
    Input:
       y_true: Tensor 
          segmentation groundtruth
       y_pred: Tensor
           affinity predictions from network
       nhood: numpy array
           neighborhood structure
           
    Output: 
       weights_pos: Tensor (same shape as y_pred)
               positive weights for malis loss, matching pairs
       weights_neg: Tensor (same shape as y_pred)
               negative weights for malis loss, nonmatching pairs
               
    Usage: 
       torchloss.apply(aff_pred, seg_gt,nhood)
    '''
    @staticmethod
    def forward(ctx, aff_pred, seg_gt,nhood=None):
        aff_pred = aff_pred.detach().numpy() # detach so we can cast to NumPy
        seg_gt = seg_gt.detach().numpy()       
            
        gtaff = seg_to_affgraph(seg_gt, nhood) # get groundtruth affinity
        weights_pos,weights_neg = malis_weights(aff_pred, gtaff, seg_gt, nhood) 
        weights_pos = weights_pos.astype(np.float32)
        weights_neg = weights_neg.astype(np.float32)

        return torch.FloatTensor(weights_pos),torch.FloatTensor(weights_neg)

    @staticmethod
    def backward(ctx, grad_output1,grad_output2):
        #The backward pass computes the gradient wrt the input and the gradient wrt the filter.
        return None,None,None

def pairs_to_loss_torch(pos_pairs, neg_pairs, output, margin=0.3, pos_loss_weight=0.5):
    '''
    Computes MALIS loss weights from given positive and negtive weights.
    
    Roughly speaking the malis weights quantify the impact of an edge in
    the predicted affinity graph on the resulting segmentation.
    
    Inputs:
        pos_pairs: (batch_size, H, W, C)
           Contains the positive pairs 
        neg_pairs: (batch_size, H, W, C)
           Contains the negative pairs 
        pred:  (batch_size, H, W, C)
            affinity predictions from network
    
    Returns:
        malis_loss: scale
            final malis loss
    '''
    neg_loss_weight = 1 - pos_loss_weight
    zeros_helpervar = torch.zeros(output.size())

    pos_loss = torch.where(1 - output - margin > 0,
                        (1 - output - margin)**2,
                        zeros_helpervar)

    pos_loss = pos_loss * pos_pairs
    pos_loss = torch.sum(pos_loss) * pos_loss_weight

    neg_loss = torch.where(output - margin > 0,
                        (output - margin)**2,
                        zeros_helpervar)
    neg_loss = neg_loss * neg_pairs
    neg_loss = torch.sum(neg_loss) * neg_loss_weight
    loss = (pos_loss + neg_loss) * 2 
    
    return loss

def malis_loss2d(seg_gt,output): 
    '''
    Computes 2d MALIS loss given predicted affinity graphs and segmentation groundtruth
    
    Roughly speaking malis weights (pos_pairs and neg_pairs) quantify the 
    impact of an edge in the predicted affinity graph on the resulting segmentation.
    
    Input:
       output: Tensor(batch size, channel=2, H, W)
              predicted affinity graphs from network
       seg_gt: Tensor(batch size, channel=1, H, W)
               segmentation groundtruth     
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
    
    ######### make sure seg_gt and output has the correct shape -> (H,W,C'), (2,H,W,batch) for each 
    x,y = seg_gt.shape[2],seg_gt.shape[3]
    output = output.permute(1,2,3,0)           # (2,H,W,batch_size)
    seg_gt = seg_gt.reshape(x,y,-1)            # (H,W,C'=C*batch_size)
    #########
    
    nhood = mknhood3d(1)[:-1]  # define neighborhood structure, check mknhood3d in pairs_cython.pyx for further information
                               # calculating connectivity among x and y axis here
    pos_pairs,neg_pairs = torchloss.apply(output, seg_gt, nhood) # get positive and negtive malis weights
    loss = pairs_to_loss_torch(pos_pairs, neg_pairs, output) # computes final malis loss
    
    return loss

def malis_loss3d(seg_gt,output): 
    '''
    Computes 3d MALIS loss given predicted affinity graphs and segmentation groundtruth
    
    Roughly speaking malis weights (pos_pairs and neg_pairs) quantify the 
    impact of an edge in the predicted affinity graph on the resulting segmentation.
    
    Input:
       seg_gt: Tensor (batch size=1, channel=1, H, W, D)
          segmentation groundtruth
       output: Tensor (batch size=1, channel=3, H, W, D)
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
    ######### make sure seg_gt and output has the correct shape -> (H,W,D),(3,H,W,D) for each 
    x,y,z = seg_gt.shape[2],seg_gt.shape[3],seg_gt.shape[4]
    output = output.reshape(-1,x,y,z)         # (3,H,W,D)
    seg_gt = seg_gt.reshape(x,y,z)            # (H,W,D)
    #########
    
    nhood = mknhood3d(1) # define neighborhood structure, check mknhood3d in pairs_cython.pyx for further information
                         # calculating connectivity among x, y and z axis here
    pos_pairs,neg_pairs = torchloss.apply(output, seg_gt, nhood) # get positive and negtive malis weights
    loss = pairs_to_loss_torch(pos_pairs, neg_pairs, output) # computes final malis loss
    
    return loss