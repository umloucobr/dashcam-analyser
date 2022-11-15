#pragma once
#ifndef LANE_DETECTOR_HPP
#define LANE_DETECTOR_HPP
#include "dashcam-analyser.hpp"
namespace dashan {
	namespace constants {
		extern const cv::Mat lowerYellowThreshold;
		extern const cv::Mat upperYellowThreshold;
		extern const cv::Mat lowerWhiteThreshold;
		extern const cv::Mat upperWhiteThreshold;
	}

	cv::Mat laneDetector (cv::Mat& input, bool drawContours);
}
#endif

