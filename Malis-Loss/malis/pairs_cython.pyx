import scipy.sparse
import sys
import os
import numpy as np
cimport numpy as np

cdef extern from "malis_cpp.h":
    void connected_components_cpp(const int n_vert,
                                  const int n_edge,
                                  const int* node1,
                                  const int* node2,
                                  const float* edge_weight,
                                  const int size_thresh,
                                  int* seg);
    
    void malis_loss_weights_cpp(const int n_vert,
                                const int* seg,
                                const int n_edge,
                                const int* node1,
                                const int* node2,
                                const float* edge_weight,
                                const int pos,
                                int* counts);
    
    void marker_watershed_cpp(const int n_vert,
                              const int* marker,
                              const int n_edge,
                              const int* node1,
                              const int* node2,
                              const float* edge_weight,
                              const int size_thresh,
                              int* seg);



def malis_loss_weights(np.ndarray[int,  ndim=1] seg_true,
                       np.ndarray[int,  ndim=1] node1,
                       np.ndarray[int,  ndim=1] node2,
                       np.ndarray[float,ndim=1] edge_weight,
                       np.int pos):
                    
    cdef int n_vert = seg_true.shape[0]
    cdef int n_edge = node1.shape[0]
    seg_true = np.ascontiguousarray(seg_true)
    node1 = np.ascontiguousarray(node1)
    node2 = np.ascontiguousarray(node2)
    edge_weight = np.ascontiguousarray(edge_weight)
    cdef np.ndarray[int,ndim=1] counts = np.zeros(edge_weight.shape[0],dtype=np.int32)
    
    malis_loss_weights_cpp(n_vert,
                           &seg_true[0],
                           n_edge,
                           &node1[0],
                           &node2[0],
                           &edge_weight[0],
                           pos,
                           &counts[0])
                   
    return counts



def connected_components(np.int n_vert,
                         np.ndarray[int,ndim=1] node1,
                         np.ndarray[int,ndim=1] node2,
                         np.ndarray[float,ndim=1] edge_weight,
                         int size_thresh=1):
                             
    cdef int n_edge = node1.shape[0]
    node1 = np.ascontiguousarray(node1)
    node2 = np.ascontiguousarray(node2)
    edge_weight = np.ascontiguousarray(edge_weight)
    cdef np.ndarray[int,ndim=1] seg = np.zeros(n_vert,dtype=np.int32)
    
    connected_components_cpp(n_vert,
                             n_edge,
                             &node1[0],
                             &node2[0],
                             &edge_weight[0],
                             size_thresh,
                             &seg[0])
                             
    unique, new_seg, seg_sizes = np.unique(seg, return_inverse=True, return_counts=True)
    if 0 not in unique: # if no BG present, new_seg will still have 0 component
        new_seg[new_seg==0] = unique[-1]+1 # we need to rename that one!
        
    new_seg = new_seg.astype(np.int32)    
    return new_seg, seg_sizes



def marker_watershed(np.ndarray[int,  ndim=1] marker,
                     np.ndarray[int,  ndim=1] node1,
                     np.ndarray[int,  ndim=1] node2,
                     np.ndarray[float,ndim=1] edge_weight,
                     int size_thresh=1):
                         
    cdef int n_vert = marker.shape[0]
    cdef int n_edge = node1.shape[0]
    marker = np.ascontiguousarray(marker)
    node1 = np.ascontiguousarray(node1)
    node2 = np.ascontiguousarray(node2)
    edge_weight = np.ascontiguousarray(edge_weight)
    cdef np.ndarray[int,ndim=1] seg = np.zeros(n_vert,dtype=np.int32)
    
    marker_watershed_cpp(n_vert,
                         &marker[0],
                         n_edge,
                         &node1[0],
                         &node2[0],
                         &edge_weight[0],
                         size_thresh,
                         &seg[0])
                         
    unique, new_seg, seg_sizes = np.unique(seg, return_inverse=True, return_counts=True)
    new_seg = new_seg.astype(np.int32)
    return new_seg, seg_sizes


