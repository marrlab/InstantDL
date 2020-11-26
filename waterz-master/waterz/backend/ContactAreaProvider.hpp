#include "StatisticsProvider.hpp"

template <typename RegionGraphType>
class ContactAreaProvider : public StatisticsProvider {

public:

	typedef size_t ValueType;
	typedef typename RegionGraphType::EdgeIdType EdgeIdType;

	ContactAreaProvider(RegionGraphType& regionGraph) :
		_contactArea(regionGraph) {}

	inline void notifyNewEdge(EdgeIdType e) {

		_contactArea[e] = 0;
	}

	template<typename ScoreType>
	inline void addAffinity(EdgeIdType e, ScoreType affinity) {

		_contactArea[e]++;
	}

	inline bool notifyEdgeMerge(EdgeIdType from, EdgeIdType to) {

		_contactArea[to] += _contactArea[from];

		// score changed
		return true;
	}

	inline ValueType operator[](EdgeIdType e) const {

		return _contactArea[e];
	}

private:

	typename RegionGraphType::template EdgeMap<ValueType> _contactArea;
};

