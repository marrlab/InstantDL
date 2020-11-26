#ifndef WATERZ_BACKEND_CONSTANT_NUMBER_PROVIDER_H__
#define WATERZ_BACKEND_CONSTANT_NUMBER_PROVIDER_H__

#include "StatisticsProvider.hpp"

template <int C>
class ConstantProvider : public StatisticsProvider {

public:

	typedef int ValueType;

	template <typename RegionGraphType>
	ConstantProvider(RegionGraphType&) {}

	inline ValueType operator()() const {

		return C;
	}
};

#endif // WATERZ_BACKEND_CONSTANT_NUMBER_PROVIDER_H__

