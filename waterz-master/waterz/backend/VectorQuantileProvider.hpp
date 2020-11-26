#ifndef WATERZ_VECTOR_QUANTILE_PROVIDER_H__
#define WATERZ_VECTOR_QUANTILE_PROVIDER_H__

#include <vector>
#include <algorithm>
#include "StatisticsProvider.hpp"

/**
 * A quantile provider using std::vector and std::nth_element to find the exact 
 * quantile.
 */
template <typename RegionGraphType, int Q, typename Precision, bool InitWithMax = true>
class VectorQuantileProvider : public StatisticsProvider {

public:

	typedef Precision ValueType;
	typedef typename RegionGraphType::EdgeIdType EdgeIdType;

	VectorQuantileProvider(RegionGraphType& regionGraph) :
		_values(regionGraph) {}

	inline void addAffinity(EdgeIdType e, ValueType affinity) {

		if (InitWithMax) {

			if (_values[e].size() == 1) {

				if (_values[e][0] < affinity)
					_values[e][0] = affinity;

				return;
			}
		}

		_values[e].push_back(affinity);
	}

	inline bool notifyEdgeMerge(EdgeIdType from, EdgeIdType to) {

		_values[to].reserve(_values[to].size() + _values[from].size());

		auto otherQuantile = getQuantileIterator(_values[from].begin(), _values[from].end(), Q);
		_values[to].insert(_values[to].begin(), _values[from].begin(), otherQuantile);
		_values[to].insert(_values[to].end(), otherQuantile, _values[from].end());

		auto quantile = getQuantileIterator(_values[to].begin(), _values[to].end(), Q);
		std::nth_element(_values[to].begin(), quantile, _values[to].end());

		_values[from].clear();

		return true;
	}

	inline ValueType operator[](EdgeIdType e) const {

		auto quantile = getQuantileIterator(_values[e].begin(), _values[e].end(), Q);
		return *quantile;
	}

private:

	template <typename It>
	inline It getQuantileIterator(It begin, It end, int q) const {

		size_t size = end - begin;
		if (size == 0) {

			std::cerr << "quantile provider is empty" << std::endl;
			throw std::exception();
		}

		int pivot = q*size/100;
		if (pivot == size)
			pivot--;

		return begin + pivot;
	}

	typename RegionGraphType::template EdgeMap<std::vector<Precision>> _values;
};

#endif // WATERZ_VECTOR_QUANTILE_PROVIDER_H__

