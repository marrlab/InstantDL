from libc.stdint cimport uint64_t
import numpy as np
cimport numpy as np

def evaluate(
        np.ndarray[uint64_t, ndim=3] segmentation,
        np.ndarray[uint64_t, ndim=3] gt):

    for d in range(3):
        assert segmentation.shape[d] == gt.shape[d], (
            "Shapes in dim %d do not match"%d)
    shape = segmentation.shape

    # the C++ part assumes contiguous memory, make sure we have it (and do 
    # nothing, if we do)
    if not segmentation.flags['C_CONTIGUOUS']:
        print("Creating memory-contiguous segmentation arrray (avoid this by passing C_CONTIGUOUS arrays)")
        segmentation = np.ascontiguousarray(segmentation)
    if gt is not None and not gt.flags['C_CONTIGUOUS']:
        print("Creating memory-contiguous ground-truth arrray (avoid this by passing C_CONTIGUOUS arrays)")
        gt = np.ascontiguousarray(gt)

    cdef uint64_t* segmentation_data
    cdef uint64_t* gt_data

    segmentation_data = &segmentation[0, 0, 0]
    gt_data = &gt[0, 0, 0]

    return compare_arrays(
        shape[0], shape[1], shape[2],
        gt_data,
        segmentation_data)

cdef extern from "frontend_evaluate.h":

    struct Metrics:
        double voi_split
        double voi_merge
        double rand_split
        double rand_merge

    Metrics compare_arrays(
            size_t          width,
            size_t          height,
            size_t          depth,
            const uint64_t* gt_data,
            const uint64_t* segmentation_data);
