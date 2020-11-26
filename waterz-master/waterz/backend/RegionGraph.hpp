#ifndef REGION_GRAPH_H__
#define REGION_GRAPH_H__

#include <algorithm>
#include <vector>
#include <limits>
#include <cassert>

template <typename ID>
struct RegionGraphEdge {

	typedef ID NodeIdType;

	NodeIdType u;
	NodeIdType v;

	RegionGraphEdge() : u(0), v(0) {}
	RegionGraphEdge(NodeIdType u_, NodeIdType v_) : u(u_), v(v_) {}
};

// forward declaration
template <typename ID>
class RegionGraph;

template<typename ID>
class RegionGraphNodeMapBase {

public:

	typedef RegionGraph<ID> RegionGraphType;

	RegionGraphType& getRegionGraph() { return _regionGraph; }

	const RegionGraphType& getRegionGraph() const { return _regionGraph; }

protected:

	RegionGraphNodeMapBase(RegionGraphType& regionGraph) :
		_regionGraph(regionGraph) {

		_regionGraph.registerNodeMap(this);
	}

	virtual ~RegionGraphNodeMapBase() {

		_regionGraph.deregisterNodeMap(this);
	}

private:

	friend RegionGraphType;

	virtual void onNewNode(ID id) = 0;

	RegionGraphType& _regionGraph;
};

template<typename ID, typename T, typename Container>
class RegionGraphNodeMap : public RegionGraphNodeMapBase<ID> {

public:

	typedef T ValueType;

	typedef RegionGraph<ID> RegionGraphType;

	RegionGraphNodeMap(RegionGraphType& regionGraph) :
		RegionGraphNodeMapBase<ID>(regionGraph),
		_values(regionGraph.numNodes()) {}

	RegionGraphNodeMap(RegionGraphType& regionGraph, Container&& values) :
		RegionGraphNodeMapBase<ID>(regionGraph),
		_values(std::move(values)) {}

	inline typename Container::const_reference operator[](ID i) const { return _values[i]; }
	inline typename Container::reference operator[](ID i) { return _values[i]; }

private:

	void onNewNode(ID id) {

		_values.push_back(T());
	}

	Container _values;
};

template<typename ID>
class RegionGraphEdgeMapBase {

public:

	typedef RegionGraph<ID> RegionGraphType;

	RegionGraphType& getRegionGraph() { return _regionGraph; }

	const RegionGraphType& getRegionGraph() const { return _regionGraph; }

protected:

	RegionGraphEdgeMapBase(RegionGraphType& regionGraph) :
		_regionGraph(regionGraph) {

		_regionGraph.registerEdgeMap(this);
	}

	virtual ~RegionGraphEdgeMapBase() {

		_regionGraph.deregisterEdgeMap(this);
	}

private:

	friend RegionGraphType;

	virtual void onNewEdge(std::size_t id) = 0;

	RegionGraphType& _regionGraph;
};

template<typename ID, typename T, typename Container>
class RegionGraphEdgeMap : public RegionGraphEdgeMapBase<ID> {

public:

	typedef T ValueType;

	typedef RegionGraph<ID> RegionGraphType;

	RegionGraphEdgeMap(RegionGraphType& regionGraph) :
		RegionGraphEdgeMapBase<ID>(regionGraph),
		_values(regionGraph.edges().size()) {}

	inline typename Container::const_reference operator[](std::size_t i) const { return _values[i]; }
	inline typename Container::reference operator[](std::size_t i) { return _values[i]; }

private:

	void onNewEdge(std::size_t id) {

		_values.push_back(T());
	}

	Container _values;
};

template <typename ID>
class RegionGraph {

public:

	typedef ID                          NodeIdType;
	typedef std::size_t                 EdgeIdType;

	typedef RegionGraphEdge<NodeIdType> EdgeType;

	template <typename T, typename Container = std::vector<T>>
	using NodeMap = RegionGraphNodeMap<ID, T, Container>;

	template <typename T, typename Container = std::vector<T>>
	using EdgeMap = RegionGraphEdgeMap<ID, T, Container>;

	static const EdgeIdType NoEdge = std::numeric_limits<EdgeIdType>::max();

	RegionGraph(ID numNodes = 0) :
		_numNodes(numNodes),
		_incEdges(numNodes) {}

	ID numNodes() const { return _numNodes; }

	std::size_t numEdges() const { return _edges.size(); }

	ID addNode() {

		NodeIdType id = _numNodes;
		_numNodes++;
		_incEdges.emplace_back();

		for (RegionGraphNodeMapBase<ID>* map : _nodeMaps)
			map->onNewNode(id);

		return id;
	}

	EdgeIdType addEdge(NodeIdType u, NodeIdType v) {

		EdgeIdType id = _edges.size();
		_edges.push_back(EdgeType(std::min(u, v), std::max(u, v)));

		_incEdges[u].push_back(id);
		_incEdges[v].push_back(id);

		for (RegionGraphEdgeMapBase<ID>* map : _edgeMaps)
			map->onNewEdge(id);

		return id;
	}

	void removeEdge(EdgeIdType e) {

		removeIncEdge(_edges[e].u, e);
		removeIncEdge(_edges[e].v, e);
	}

	void moveEdge(EdgeIdType e, NodeIdType u, NodeIdType v) {

		// three possibilities:
		//
		//   1. nothing changed (unlikely, callers responsibility)
		//   2. only u or v changed
		//      order independent, four subcases
		//   3. u and v changed

		NodeIdType pu = _edges[e].u;
		NodeIdType pv = _edges[e].v;

		// is pu already one of the new nodes?
		if (pu == u) {

			// keep pu, update pv -> v
			moveEdgeNodeV(e, v);

		} else if (pu == v) {

			// keep pu, update pv -> u
			moveEdgeNodeV(e, u);

		} else {

			// is pv already one of the new nodes?
			if (pv == v) {

				// keep pv, update pu -> u
				moveEdgeNodeU(e, u);

			} else if (pv == u) {

				// keep pv, update pu -> u
				moveEdgeNodeU(e, v);

			} else {

				// none of them is equal to the new nodes
				moveEdgeNodeU(e, u);
				moveEdgeNodeV(e, v);
			}
		}

		// ensure new ids are sorted
		if (_edges[e].u > _edges[e].v)
			std::swap(_edges[e].u, _edges[e].v);

		assert(std::min(u, v) == _edges[e].u);
		assert(std::max(u, v) == _edges[e].v);
		assert(findEdge(u, v) == e);
		assert(std::find(incEdges(u).begin(), incEdges(u).end(), e) != incEdges(u).end());
		assert(std::find(incEdges(v).begin(), incEdges(v).end(), e) != incEdges(v).end());
	}

	inline const EdgeType& edge(EdgeIdType e) const { return _edges[e]; }

	inline const std::vector<EdgeType>& edges() const { return _edges; }

	inline const std::vector<EdgeIdType>& incEdges(ID node) const { return _incEdges[node]; }

	inline NodeIdType getOpposite(NodeIdType n, EdgeIdType e) const {

		return (_edges[e].u == n ? _edges[e].v : _edges[e].u);
	}

	/**
	 * Find the edge connecting u and v. Returns NoEdge, if there is none.
	 */
	inline EdgeIdType findEdge(NodeIdType u, NodeIdType v) {

		return findEdge(u, v, (_incEdges[u].size() < _incEdges[v].size() ? _incEdges[u] : _incEdges[v]));
	}

	/**
	 * Same as findEdge(u, v), but restricted to edges in pool.
	 */
	inline EdgeIdType findEdge(NodeIdType u, NodeIdType v, const std::vector<EdgeIdType>& pool) {

		NodeIdType min = std::min(u, v);
		NodeIdType max = std::max(u, v);

		for (EdgeIdType e : pool)
			if (std::min(_edges[e].u, _edges[e].v) == min &&
				std::max(_edges[e].u, _edges[e].v) == max)
				return e;

		return NoEdge;
	}

private:

	friend RegionGraphNodeMapBase<ID>;
	friend RegionGraphEdgeMapBase<ID>;

	void registerNodeMap(RegionGraphNodeMapBase<ID>* nodeMap) {

		_nodeMaps.push_back(nodeMap);
	}

	void deregisterNodeMap(RegionGraphNodeMapBase<ID>* nodeMap) {

		auto it = std::find(_nodeMaps.begin(), _nodeMaps.end(), nodeMap);
		if (it != _nodeMaps.end())
			_nodeMaps.erase(it);
	}

	void registerEdgeMap(RegionGraphEdgeMapBase<ID>* edgeMap) {

		_edgeMaps.push_back(edgeMap);
	}

	void deregisterEdgeMap(RegionGraphEdgeMapBase<ID>* edgeMap) {

		auto it = std::find(_edgeMaps.begin(), _edgeMaps.end(), edgeMap);
		if (it != _edgeMaps.end())
			_edgeMaps.erase(it);
	}

	inline void moveEdgeNodeV(EdgeIdType e, NodeIdType v) {

		removeIncEdge(_edges[e].v, e);
		_incEdges[v].push_back(e);
		_edges[e].v = v;
	}

	inline void moveEdgeNodeU(EdgeIdType e, NodeIdType u) {

		removeIncEdge(_edges[e].u, e);
		_incEdges[u].push_back(e);
		_edges[e].u = u;
	}

	inline void removeIncEdge(NodeIdType n, EdgeIdType e) {

		auto it = std::find(_incEdges[n].begin(), _incEdges[n].end(), e);
		assert(it != _incEdges[n].end());
		_incEdges[n].erase(it);
		assert(std::find(_incEdges[n].begin(), _incEdges[n].end(), e) == _incEdges[n].end());
	}

	ID _numNodes;

	std::vector<EdgeType> _edges;

	std::vector<std::vector<EdgeIdType>> _incEdges;

	std::vector<RegionGraphNodeMapBase<ID>*> _nodeMaps;
	std::vector<RegionGraphEdgeMapBase<ID>*> _edgeMaps;
};

#endif // REGION_GRAPH_H__

