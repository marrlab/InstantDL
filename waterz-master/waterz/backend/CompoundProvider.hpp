#ifndef WATERZ_COMPOUND_PROVIDER_H__
#define WATERZ_COMPOUND_PROVIDER_H__

/**
 * Combines statistics providers into a single provider, which inherits from all 
 * the other ones.
 */

// inherits from all given types
template <typename Head, typename ... Tail>
class CompoundProvider : public Head, public CompoundProvider<Tail...> {

public:

	typedef Head HeadType;
	typedef CompoundProvider<Tail...> Parent;

	template <typename RegionGraphType>
	CompoundProvider(RegionGraphType& regionGraph) :
		Head(regionGraph),
		Parent(regionGraph) {}

	template <typename EdgeIdType>
	inline void notifyNewEdge(EdgeIdType e) {
	
		Head::notifyNewEdge(e);
		Parent::notifyNewEdge(e);
	}

	template <typename EdgeIdType, typename ScoreType>
	inline void addAffinity(EdgeIdType e, ScoreType affinity) {

		Head::addAffinity(e, affinity);
		Parent::addAffinity(e, affinity);
	}

	template <typename NodeIdType>
	inline void addVoxel(NodeIdType n, std::size_t x, std::size_t y, std::size_t z) {

		Head::addVoxel(n, x, y, z);
		Parent::addVoxel(n, x, y, z);
	}

	template<typename NodeIdType>
	inline bool notifyNodeMerge(NodeIdType from, NodeIdType to) {

		return (
				Head::notifyNodeMerge(from, to) ||
				Parent::notifyNodeMerge(from, to));
	}

	template<typename EdgeIdType>
	inline bool notifyEdgeMerge(EdgeIdType from, EdgeIdType to) {

		return (
				Head::notifyEdgeMerge(from, to) ||
				Parent::notifyEdgeMerge(from, to));
	}
};


// end of recursion

struct EndOfCompound {};

template <typename Head>
class CompoundProvider<Head> : public Head {
public:

	typedef Head HeadType;
	typedef EndOfCompound Parent;

	template <typename RegionGraphType>
	CompoundProvider(RegionGraphType& regionGraph) :
		Head(regionGraph) {}
};

#endif // WATERZ_COMPOUND_PROVIDER_H__

