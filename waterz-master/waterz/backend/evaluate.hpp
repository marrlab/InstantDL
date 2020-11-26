#ifndef WATERZ_EVALUATE_H__
#define WATERZ_EVALUATE_H__

#include <iostream>
#include <tuple>
#include <map>
#include <math.h> 

using namespace std;

template <typename V1, typename V2>
std::tuple<double,double,double,double>
compare_volumes(
				 const V1& gt,
				 const V2& ws){

	size_t dimX = gt.shape()[0];
	size_t dimY = gt.shape()[1];
	size_t dimZ = gt.shape()[2];

	double total = 0;

	// number of co-occurences of label i and j
	std::map<uint64_t, std::map<uint64_t, double>> p_ij;

	// number of occurences of label i and j in the respective volumes
	std::map<uint64_t, double> s_i, t_j;

	for ( std::ptrdiff_t z = 0; z < dimZ; ++z )
		for ( std::ptrdiff_t y = 0; y < dimY; ++y )
			for ( std::ptrdiff_t x = 0; x < dimX; ++x )
			{
				uint64_t wsv = ws[x][y][z];
				uint64_t gtv = gt[x][y][z];

				if ( gtv )
				{
					++total;

					++p_ij[gtv][wsv];
					++s_i[wsv];
					++t_j[gtv];
				}
			}

	// sum of squares in p_ij
	double sum_p_ij = 0;
	for ( auto& a: p_ij )
		for ( auto& b: a.second )
			sum_p_ij += b.second * b.second;

	// sum of squares in t_j
	double sum_t_k = 0;
	for ( auto& a: t_j )
		sum_t_k += a.second * a.second;

	// sum of squares in s_i
	double sum_s_k = 0;
	for ( auto& a: s_i )
		sum_s_k += a.second * a.second;

	// we have everything we need for RAND, normalize histograms for VOI

	for ( auto& a: p_ij )
		for ( auto& b: a.second )
			b.second /= total;

	for ( auto& a: t_j )
		a.second /= total;

	for ( auto& a: s_i )
		a.second /= total;

	// compute entropies

	// H(s,t)
	double H_st = 0;
	for ( auto& a: p_ij )
		for ( auto& b: a.second )
			if(b.second)
				H_st -= b.second * log2(b.second);

	// H(t)
	double H_t = 0;
	for ( auto& a: t_j )
		if(a.second)
			H_t -= a.second * log2(a.second);

	// H(s)
	double H_s = 0;
	for ( auto& a: s_i )
		if(a.second)
			H_s -= a.second * log2(a.second);

	double rand_split = sum_p_ij/sum_t_k;
	double rand_merge = sum_p_ij/sum_s_k;

	// H(s|t)
	double voi_split = H_st - H_t;
	// H(t|s)
	double voi_merge = H_st - H_s;

	std::cout << "\tRand split: " << rand_split << "\n";
	std::cout << "\tRand merge: " << rand_merge << "\n";
	std::cout << "\tVOI split: " << voi_split << "\n";
	std::cout << "\tVOI merge: " << voi_merge << "\n";

	return std::make_tuple(
			rand_split,
			rand_merge,
			voi_split,
			voi_merge);
}

#endif // WATERZ_EVALUATE_H__

