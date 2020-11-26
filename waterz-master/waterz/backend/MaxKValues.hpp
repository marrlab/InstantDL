#ifndef WATERZ_MAX_K_VALUES_H__
#define WATERZ_MAX_K_VALUES_H__

template <typename T, int K>
class MaxKValues {

public:

	MaxKValues() {

		for (auto& v : _values)
			v = std::numeric_limits<T>::lowest();
	}

	void push(T value) {

		for (int k = 0; k < K; k++) {

			if (_values[k] == std::numeric_limits<T>::lowest()) {

				_values[k] = value;
				break;
			}

			if (_values[k] < value) {

				// move values down
				for (int i = K-1; i > k; i--)
					_values[i] = _values[i-1];
				_values[k] = value;
				break;
			}
		}
	}

	void merge(const MaxKValues& other) {

		T a[K];
		std::copy(_values, _values+K, a);
		const auto& b = other._values;

		int i = 0;
		int j = 0;
		for (int k = 0; k < K; k++) {

			if (a[i] > b[j]) {

				_values[k] = a[i];
				i++;

			} else {

				_values[k] = b[j];
				j++;
			}
		}
	}

	T average() const {

		T sum = 0;
		int k;
		for (k = 0; k < K; k++) {
			if (_values[k] == std::numeric_limits<T>::lowest())
				break;
			sum += _values[k];
		}

		if (k == 0)
			return std::numeric_limits<T>::signaling_NaN();

		return sum/k;
	}

private:

	T _values[K];
};

#endif // WATERZ_MAX_K_VALUES_H__

