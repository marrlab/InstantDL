#include "frontend_evaluate.h"
#include "evaluate.hpp"

Metrics
compare_arrays(
		std::size_t  width,
		std::size_t  height,
		std::size_t  depth,
		const SegID* gt_data,
		const SegID* segmentation_data) {

	// wrap gt array (no copy)
	volume_const_ref<SegID> gt(
			gt_data,
			boost::extents[width][height][depth]
	);

	// wrap segmentation array (no copy)
	volume_const_ref<SegID> segmentation(
			segmentation_data,
			boost::extents[width][height][depth]
	);

	auto m = compare_volumes(gt, segmentation);

	Metrics metrics;
	metrics.rand_split = std::get<0>(m);
	metrics.rand_merge = std::get<1>(m);
	metrics.voi_split  = std::get<2>(m);
	metrics.voi_merge  = std::get<3>(m);

	return metrics;
}
