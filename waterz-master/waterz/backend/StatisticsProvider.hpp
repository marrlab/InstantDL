#ifndef WATERZ_STATISTICS_PROVIDER_H__
#define WATERZ_STATISTICS_PROVIDER_H__

/**
 * Base class for statistics providers with fallback implementations.
 */
class StatisticsProvider {

public:

	/**
	 * Callback for adding edges to the RAG.
	 */
	template <typename EdgeIdType>
	inline void notifyNewEdge(EdgeIdType e) {}

	/**
	 * Callback for adding voxel-level affinities to an edge. Will be called 
	 * after notifyNewEdge().
	 */
	template <typename EdgeIdType, typename ScoreType>
	inline void addAffinity(EdgeIdType e, ScoreType affinity) {}

	template <typename NodeIdType>
	inline void addVoxel(NodeIdType n, std::size_t x, std::size_t y, std::size_t z) {}

	/**
	 * Callback for node merges: 'from' will be merged into 'to'. Return true, 
	 * if this changed the statistics of this provider.
	 */
	template<typename NodeIdType>
	inline bool notifyNodeMerge(NodeIdType from, NodeIdType to) { return false; }

	/**
	 * Callback for edge merges: 'from' will be merged into 'to'. Return true, 
	 * if this changed the statistics of this provider.
	 */
	template<typename EdgeIdType>
	inline bool notifyEdgeMerge(EdgeIdType from, EdgeIdType to) { return false; }
};

#endif // WATERZ_STATISTICS_PROVIDER_H__

