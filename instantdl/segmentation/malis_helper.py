def object_segmentation(seg, gt, dimension):
    """     
    This evaluation criteria is based on counting the number of correctly 
    segmented objects (Dice coefficient > 70 %), divided by the average of 
    total number of ground-truth objects.

    Input:
        seg: list
           List of segmentation images (batch,H,W,C)
        gt: list
           List of groundtruths (H,W)
        dimension: number
           2 -> 2D data
           3 -> 3D data
    Returns:
        score: scale
           The score of this criteria
    """
    from skimage.measure import label
    from collections import Counter
    from skimage import morphology
    import numpy as np
        
    seg = label(morphology.remove_small_objects(label(seg), 64))    
    if dimension == 2:
        groundtruth = label(morphology.remove_small_objects(label(np.expand_dims(gt,axis=-1)), 64))
    if dimension == 3:
        groundtruth = label(morphology.remove_small_objects(label(gt), 64))
    dice = []
    label_list = []

    for value in range(1,np.max(groundtruth)+1):

        tmpgt = np.copy(groundtruth)
        tmpgt = np.where(tmpgt == value,1,0)

        tmpseg = np.copy(seg)
        seg_label = Counter(tmpseg[groundtruth == value]).most_common(1)[0][0] # find the most common class in the crossponding areas
        if seg_label in label_list:   # if several objects are connected in the segmentation images, they are considered as wrongly segmented objects
            diceseg = 0
            index = label_list.index(seg_label)
            dice[index] = 0
        else:
            tmpseg = np.where(tmpseg == seg_label,1,0)
            diceseg = np.sum(tmpseg[tmpgt == 1])*2/float(np.sum(tmpseg) + np.sum(tmpgt))
        label_list.append(seg_label)
        dice.append(diceseg)
    
    score = sum(1 for i in dice if i > 0.7)/len(dice) # counting dice > 70%
    return score

def find_threshold(pred_aff, gt, dimension):
    """
    This function aims to automatically find an optimal threshold, 
    converting the predicted affinity graphs to the segmentation images, 
    to achive the highest score of the given evaluation indices. 
     
    The evaluation criteria used here is object segmentation.

    Input:
        pred_aff: list
           List of predicted affinity graphs (batch,H,W,C)
        gt: list
           List of crossponding groundtruths (H,W)
        dimension: number
           2 -> 2D data
           3 -> 3D data
    Returns:
        opt_threshold: scale
           The optimal threshold
        result: scale
           The highest score with the given evaluation criteria
        output_seg: list
           List of segmentation images with the optimal threshold
    """
    #from scipy.ndimage.morphology import binary_fill_holes
    import numpy as np
    import malis as m

    result = 0
    opt_threshold = 0
    output_seg = 0.6

    for threshold in np.arange(0.1, 1.0, 0.01):
        segmalis_dice = []
        final_seg = []
        
        if dimension == 2:
            for patch in range(len(pred_aff)):
                nhood = m.mknhood3d(1)[:-1]
                aff = np.transpose(np.expand_dims(np.where(pred_aff[patch][0]<threshold,0,1),axis=0),(3,1,2,0))
                seg = m.affgraph_to_seg(aff.astype(np.int32),nhood)[0]     # obtain segmentation from predicted affinity graphs
                seg = np.where(seg==0,0,1)
                #seg = binary_fill_holes(seg).astype(int)  # fill small holes in the isntances
                final_seg.append(seg)
                segmalis_dice.append(object_segmentation(seg, gt[patch], dimension))  # could be changed to other indices
                
        if dimension == 3:
            for patch in range(len(pred_aff)):
                nhood = m.mknhood3d(1)
                aff = np.transpose(np.where(pred_aff[patch][0]<threshold,0,1),(3,0,1,2))  #####
                seg = m.affgraph_to_seg(aff.astype(np.int32),nhood)[0]     # obtain segmentation from predicted affinity graphs
                seg = np.where(seg==0,0,1)
                #seg = binary_fill_holes(seg).astype(int)  # fill small holes in the isntances
                final_seg.append(seg)
                segmalis_dice.append(object_segmentation(seg, gt[patch], dimension))  # could be changed to other indices
                
        if sum(segmalis_dice)/len(segmalis_dice) > result:
            result = sum(segmalis_dice)/len(segmalis_dice)
            opt_threshold = threshold
            output_seg = final_seg
                
    return opt_threshold, result, output_seg