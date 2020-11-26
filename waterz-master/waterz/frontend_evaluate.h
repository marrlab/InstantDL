#ifndef C_EVALUATE_H
#define C_EVALUATE_H

#include "backend/types.hpp"

typedef uint64_t SegID;

struct Metrics {

	double voi_split;
	double voi_merge;
	double rand_split;
	double rand_merge;
};

Metrics
compare_arrays(
		std::size_t  width,
		std::size_t  height,
		std::size_t  depth,
		const SegID* gt_data,
		const SegID* segmentation_data);

#endif

