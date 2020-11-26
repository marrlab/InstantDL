#include "StatisticsProvider.hpp"

template <typename RegionGraphType, typename Precision>
class MeanAffinityProvider : public StatisticsProvider {

public:

	typedef Precision ValueType;
	typedef typename RegionGraphType::EdgeIdType EdgeIdType;

	MeanAffinityProvider(RegionGraphType& regionGraph) :
		_numValues(regionGraph),
		_meanAffinities(regionGraph) {}

	inline void notifyNewEdge(EdgeIdType e) {

		_numValues[e] = 0;
		_meanAffinities[e] = 0;
	}

	inline void addAffinity(EdgeIdType e, ValueType affinity) {
	
		size_t n = _numValues[e];
		Precision mean = _meanAffinities[e];

		_meanAffinities[e] = (affinity + mean*n)/(n+1);
		_numValues[e]++;
	}

	inline bool notifyEdgeMerge(EdgeIdType from, EdgeIdType to) {

		size_t fromN = _numValues[from];
		size_t toN = _numValues[to];
		Precision fromMean = _meanAffinities[from];
		Precision toMean = _meanAffinities[to];

		_meanAffinities[to] = (fromMean*fromN + toMean*toN)/(fromN + toN);
		_numValues[to] = fromN + toN;

		// score changed
		return true;
	}

	inline ValueType operator[](EdgeIdType e) const {

		return _meanAffinities[e];
	}

private:

	typename RegionGraphType::template EdgeMap<size_t> _numValues;
	typename RegionGraphType::template EdgeMap<ValueType> _meanAffinities;
};

