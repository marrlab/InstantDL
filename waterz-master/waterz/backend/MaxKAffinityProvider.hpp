#ifndef WATERZ_MAX_K_AFFINITY_PROVIDER_H__
#define WATERZ_MAX_K_AFFINITY_PROVIDER_H__

#include "MaxKValues.hpp"
#include "StatisticsProvider.hpp"

template <typename RegionGraphType, int K, typename Precision>
class MaxKAffinityProvider : public StatisticsProvider {

public:

	typedef const MaxKValues<Precision,K>& ValueType;
	typedef typename RegionGraphType::EdgeIdType EdgeIdType;

	MaxKAffinityProvider(RegionGraphType& regionGraph) :
		_maxKValues(regionGraph) {}

	inline void addAffinity(EdgeIdType e, Precision affinity) {

		_maxKValues[e].push(affinity);
	}

	inline bool notifyEdgeMerge(EdgeIdType from, EdgeIdType to) {

		_maxKValues[to].merge(_maxKValues[from]);
		return true;
	}

	inline const MaxKValues<Precision,K>& operator[](EdgeIdType e) const {

		return _maxKValues[e];
	}

private:

	typename RegionGraphType::template EdgeMap<MaxKValues<Precision,K>> _maxKValues;
};



#endif // WATERZ_MAX_K_AFFINITY_PROVIDER_H__

