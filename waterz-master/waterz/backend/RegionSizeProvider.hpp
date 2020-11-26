#include "StatisticsProvider.hpp"

template <typename RegionGraphType>
class RegionSizeProvider : public StatisticsProvider {

public:

	typedef size_t ValueType;
	typedef typename RegionGraphType::NodeIdType NodeIdType;

	RegionSizeProvider(RegionGraphType& regionGraph) :
		_regionSizes(regionGraph) {}

	inline void addVoxel(NodeIdType n, std::size_t x, std::size_t y, std::size_t z) {

		_regionSizes[n]++;
	}

	inline bool notifyNodeMerge(NodeIdType from, NodeIdType to) {

		_regionSizes[to] += _regionSizes[from];
		_regionSizes[from] = 0;

		return true;
	}

	inline ValueType operator[](NodeIdType n) const {

		return _regionSizes[n];
	}

private:

	typename RegionGraphType::template NodeMap<ValueType> _regionSizes;
};
