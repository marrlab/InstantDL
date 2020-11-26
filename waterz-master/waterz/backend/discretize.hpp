#ifndef WATERZ_DISCRETIZE_H__
#define WATERZ_DISCRETIZE_H__

template <typename To, typename From, typename LevelsType>
inline To discretize(From value, LevelsType levels) {

	return std::min((To)(value*levels), (To)(levels-1));
}

template <typename To, typename From, typename LevelsType>
inline To undiscretize(From value, LevelsType levels) {

	return ((To)value + 0.5)/levels;
}

#endif // WATERZ_DISCRETIZE_H__

