#pragma once

#include <boost/multi_array.hpp>
#include <boost/multi_array/types.hpp>

#include <memory>
#include <vector>
#include <tuple>

#include "RegionGraph.hpp"

template < typename T > struct watershed_traits;

template <> struct watershed_traits<uint32_t>
{
    static const uint32_t high_bit = 0x80000000;
    static const uint32_t mask     = 0x7FFFFFFF;
    static const uint32_t visited  = 0x00001000;
    static const uint32_t dir_mask = 0x0000007F;
    
    static const uint32_t high_bit_2 = 0x40000000;
    static const uint32_t mask_2     = 0xBFFFFFFF;
	static const uint32_t mask_3     = 0x3FFFFFFF;
	static const uint32_t mask_high  = 0xC0000000;
};

template <> struct watershed_traits<uint64_t>
{
    static const uint64_t high_bit = 0x8000000000000000LL;
    static const uint64_t mask     = 0x7FFFFFFFFFFFFFFFLL;
    static const uint64_t visited  = 0x0000000000001000LL;
    static const uint64_t dir_mask = 0x000000000000007FLL;
    
    static const uint64_t high_bit_2 = 0x40000000;
    static const uint64_t mask_2     = 0xBFFFFFFF;
	static const uint64_t mask_3     = 0x3FFFFFFF;
	static const uint64_t mask_high  = 0xC0000000;
};

template < typename T >
using volume = boost::multi_array<T,3>;

template < typename T >
using volume_ref = boost::multi_array_ref<T,3>;

template < typename T >
using volume_const_ref = boost::const_multi_array_ref<T,3>;

template < typename T >
using affinity_graph = boost::multi_array<T,4>;

template < typename T >
using affinity_graph_ref = boost::const_multi_array_ref<T,4>;

template < typename T >
using volume_ptr = std::shared_ptr<volume<T>>;

template < typename T >
using volume_ref_ptr = std::shared_ptr<volume_ref<T>>;

template < typename T >
using volume_const_ref_ptr = std::shared_ptr<volume_const_ref<T>>;

template < typename T >
using affinity_graph_ptr = std::shared_ptr<affinity_graph<T>>;

template < typename T >
using affinity_graph_ref_ptr = std::shared_ptr<affinity_graph_ref<T>>;

template< typename ID >
using region_graph_ptr = std::shared_ptr<RegionGraph<ID>>;

template < typename T >
using counts_t = std::vector<T>;

template < typename T >
using counts_ptr = std::shared_ptr<counts_t<T>>;
