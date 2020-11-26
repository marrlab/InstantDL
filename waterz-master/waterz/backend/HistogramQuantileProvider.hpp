#ifndef WATERZ_HISTOGRAM_QUANTILE_PROVIDER_H__
#define WATERZ_HISTOGRAM_QUANTILE_PROVIDER_H__

#include "StatisticsProvider.hpp"
#include "Histogram.hpp"
#include "discretize.hpp"

/**
 * A quantile provider using histograms to find an approximate quantile. This 
 * assumes that all values are in the range [0,1].
 */
template <typename RegionGraphType, int Q, typename Precision, int Bins = 256, bool InitWithMax = true>
class HistogramQuantileProvider : public StatisticsProvider {

public:

	typedef Precision ValueType;
	typedef typename RegionGraphType::EdgeIdType EdgeIdType;

	HistogramQuantileProvider(RegionGraphType& regionGraph) :
		_histograms(regionGraph) {}

	inline void addAffinity(EdgeIdType e, ValueType affinity) {

		int bin = discretize<int>(affinity, Bins);

		if (InitWithMax && _histograms[e].lowestBin() != Bins) {

			if (bin > _histograms[e].lowestBin())
				_histograms[e].clear();
			else
				return;
		}

		_histograms[e].inc(bin);
	}

	inline bool notifyEdgeMerge(EdgeIdType from, EdgeIdType to) {

		_histograms[to] += _histograms[from];
		_histograms[from].clear();

		return true;
	}

	inline ValueType operator[](EdgeIdType e) const {

		// pivot element, 1-based index
		int pivot = Q*_histograms[e].sum()/100 + 1;

		int sum = 0;
		int bin = 0;
		for (bin = 0; bin < Bins; bin++) {

			sum += _histograms[e][bin];

			if (sum >= pivot)
				break;
		}

		return undiscretize<Precision>(bin, Bins);
	}

private:

	typename RegionGraphType::template EdgeMap<Histogram<Bins>> _histograms;
};

#endif // WATERZ_HISTOGRAM_QUANTILE_PROVIDER_H__

