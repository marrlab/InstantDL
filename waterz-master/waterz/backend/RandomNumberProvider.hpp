#ifndef WATERZ_BACKEND_RANDOM_NUMBER_PROVIDER_H__
#define WATERZ_BACKEND_RANDOM_NUMBER_PROVIDER_H__

#include <random>
#include "StatisticsProvider.hpp"

/**
 * Provides a random number between 0 and 1, whenever the score function is 
 * re-evaluated. Does not indicate score changes on node nor edge merge.
 */
class RandomNumberProvider : public StatisticsProvider {

public:

	typedef float ValueType;

	template <typename RegionGraphType>
	RandomNumberProvider(RegionGraphType&) {}

	inline ValueType operator()() const {

		return ValueType(rand())/RAND_MAX;
	}
};

#endif // WATERZ_BACKEND_RANDOM_NUMBER_PROVIDER_H__

