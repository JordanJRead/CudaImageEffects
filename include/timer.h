#ifndef MY_TIMER_H
#define MY_TIMER_H

#include <chrono>
#include <string>
#include <fstream>
#include <iostream>

typedef std::chrono::steady_clock::time_point TIME;

template <int ActivityCount>
class Timer {
public:

	void start(int activityIndex) {
		startTimes[activityIndex] = std::chrono::steady_clock::now();
	}

	void end(int activityIndex) {
		TIME now{ std::chrono::steady_clock::now() };

		totalTimeRan[activityIndex] += std::chrono::duration_cast<std::chrono::milliseconds>(now - startTimes[activityIndex]).count();
		++timesRan[activityIndex];
	}

	void outputToFile(const std::string& fileName) {
		std::ofstream file{ fileName };

		if (!file.is_open()) {
			std::cerr << "Timer error: could not open file " << fileName << "\n";
			return;
		}

		for (int i{ 0 }; i < ActivityCount; ++i) {
			if (timesRan[i] == 0)
				continue;

			double averageTime{ totalTimeRan[i] / timesRan[i] };
			file << "Activity " << i << ": " << averageTime << "ms" << "\n";
		}

		file.close();
	}

private:
	std::array<double, ActivityCount> totalTimeRan{};
	std::array<int, ActivityCount> timesRan{};
	std::array<TIME, ActivityCount> startTimes{};
};

#endif

