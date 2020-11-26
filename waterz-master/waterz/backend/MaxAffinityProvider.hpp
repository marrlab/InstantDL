#include "StatisticsProvider.hpp"

template <typename RegionGraphType, typename Precision>
class MaxAffinityProvider : public StatisticsProvider {

public:

	typedef Precision ValueType;
	typedef typename RegionGraphType::EdgeIdType EdgeIdType;

	MaxAffinityProvider(RegionGraphType& regionGraph) :
		_maxAffinities(regionGraph) {}

	inline void notifyNewEdge(EdgeIdType e) {

		_maxAffinities[e] = 0;
	}

	inline void addAffinity(EdgeIdType e, ValueType affinity) {
	
		_maxAffinities[e] = std::max(_maxAffinities[e], affinity);
	}

	inline bool notifyEdgeMerge(EdgeIdType from, EdgeIdType to) {

		if (_maxAffinities[to] >= _maxAffinities[from])
			// no change
			return false;

		_maxAffinities[to] = _maxAffinities[from];

		// score changed
		return true;
	}

	inline ValueType operator[](EdgeIdType e) const {

		return _maxAffinities[e];
	}

private:

	typename RegionGraphType::template EdgeMap<ValueType> _maxAffinities;
};
