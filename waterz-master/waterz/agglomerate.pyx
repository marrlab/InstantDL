from libcpp.vector cimport vector
from libc.stdint cimport uint64_t, uint32_t
from libcpp cimport bool
import numpy as np
cimport numpy as np

def agglomerate(
        affs,
        thresholds,
        gt=None,
        fragments=None,
        aff_threshold_low=0.0001,
        aff_threshold_high=0.9999,
        return_merge_history=False,
        return_region_graph=False):

    # the C++ part assumes contiguous memory, make sure we have it (and do 
    # nothing, if we do)
    if not affs.flags['C_CONTIGUOUS']:
        print("Creating memory-contiguous affinity arrray (avoid this by passing C_CONTIGUOUS arrays)")
        affs = np.ascontiguousarray(affs)
    if gt is not None and not gt.flags['C_CONTIGUOUS']:
        print("Creating memory-contiguous ground-truth arrray (avoid this by passing C_CONTIGUOUS arrays)")
        gt = np.ascontiguousarray(gt)
    if fragments is not None and not fragments.flags['C_CONTIGUOUS']:
        print("Creating memory-contiguous fragments arrray (avoid this by passing C_CONTIGUOUS arrays)")
        fragments = np.ascontiguousarray(fragments)

    print("Preparing segmentation volume...")

    if fragments is None:
        volume_shape = (affs.shape[1], affs.shape[2], affs.shape[3])
        segmentation = np.zeros(volume_shape, dtype=np.uint64)
        find_fragments = True
    else:
        segmentation = fragments
        find_fragments = False

    cdef WaterzState state = __initialize(affs, segmentation, gt, aff_threshold_low, aff_threshold_high, find_fragments)

    thresholds.sort()
    for threshold in thresholds:

        merge_history = mergeUntil(state, threshold)

        result = (segmentation,)

        if gt is not None:

            stats = {}
            stats['V_Rand_split'] = state.metrics.rand_split
            stats['V_Rand_merge'] = state.metrics.rand_merge
            stats['V_Info_split'] = state.metrics.voi_split
            stats['V_Info_merge'] = state.metrics.voi_merge

            result += (stats,)

        if return_merge_history:

            result += (merge_history,)

        if return_region_graph:

            result += (getRegionGraph(state),)

        if len(result) == 1:
            yield result[0]
        else:
            yield result

    free(state)

def __initialize(
        np.ndarray[np.float32_t, ndim=4] affs,
        np.ndarray[uint64_t, ndim=3]     segmentation,
        np.ndarray[uint32_t, ndim=3]     gt = None,
        aff_threshold_low  = 0.0001,
        aff_threshold_high = 0.9999,
        find_fragments = True):

    cdef float*    aff_data
    cdef uint64_t* segmentation_data
    cdef uint32_t* gt_data = NULL

    aff_data = &affs[0,0,0,0]
    segmentation_data = &segmentation[0,0,0]
    if gt is not None:
        gt_data = &gt[0,0,0]

    return initialize(
        affs.shape[1], affs.shape[2], affs.shape[3],
        aff_data,
        segmentation_data,
        gt_data,
        aff_threshold_low,
        aff_threshold_high,
        find_fragments)

cdef extern from "frontend_agglomerate.h":

    struct Metrics:
        double voi_split
        double voi_merge
        double rand_split
        double rand_merge

    struct Merge:
        uint64_t a
        uint64_t b
        uint64_t c
        double score

    struct ScoredEdge:
        uint64_t u
        uint64_t v
        double score

    struct WaterzState:
        int     context
        Metrics metrics

    WaterzState initialize(
            size_t          width,
            size_t          height,
            size_t          depth,
            const float*    affinity_data,
            uint64_t*       segmentation_data,
            const uint32_t* groundtruth_data,
            float           affThresholdLow,
            float           affThresholdHigh,
            bool            findFragments);

    vector[Merge] mergeUntil(
            WaterzState& state,
            float        threshold)

    vector[ScoredEdge] getRegionGraph(WaterzState& state)

    void free(WaterzState& state)
