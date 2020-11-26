#ifndef WATERZ_PRIORITY_QUEUE_H__
#define WATERZ_PRIORITY_QUEUE_H__

template <typename T, typename ScoreType>
class PriorityQueue {

public:

	PriorityQueue() {}

	void push(const T& element, ScoreType score) {

		_queue.push({element, score});
	}

	const T& top() const {

		return _queue.top().element;
	}

	void pop() {

		_queue.pop();
	}

	bool empty() const {

		return _queue.empty();
	}

	size_t size() const {

		return _queue.size();
	}

private:

	struct Entry {

		T element;
		ScoreType score;

		bool operator>(const Entry& other) const {
			return score > other.score;
		}
	};

	std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> _queue;
};


#endif // WATERZ_PRIORITY_QUEUE_H__

