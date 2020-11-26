#ifndef HISTOGRAM_H__
#define HISTOGRAM_H__

#include <vector>

template <int Bins, typename T = int>
class Histogram {

public:

	Histogram() { clear(); }

	Histogram operator+(const Histogram& other) {

		Histogram result(*this);
		result += other;
		return result;
	}

	Histogram& operator+=(const Histogram& other) {

		for (int i = 0; i < Bins; i++)
			_bins[i] += other._bins[i];
		_sum += other._sum;
		_lowestBin = std::min(_lowestBin, other._lowestBin);
		return *this;
	}

	void inc(int i) {

		_bins[i]++;
		_sum++;
		_lowestBin = std::min(_lowestBin, i);
	}

	const T& operator[](int i) const { return _bins[i]; }

	T sum() const { return _sum; }

	void clear() {

		_sum = 0;
		for (int i = 0; i < Bins; i++)
			_bins[i] = 0;
		_lowestBin = Bins;
	}

	/**
	 * Get the lowest non-empty bin. Returns Bins, if all bins are empty.
	 */
	T lowestBin() const { return _lowestBin; }

private:

	T _bins[Bins];
	T _sum;
	T _lowestBin;
};

#endif // HISTOGRAM_H__

