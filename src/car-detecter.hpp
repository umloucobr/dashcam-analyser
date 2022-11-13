#pragma once
#ifndef CAR_DETECTOR_H
#define CAR_DETECTOR_H
#include "dashcam-analyser.hpp"
namespace dashan {
	namespace constants {
		extern const float scoreThreshold;
		extern const float NMSThreshold;
		extern const float confidenceThreshold;
		extern const float inputHeight;
		extern const float inputWidth;
	}

	struct Detection
	{
		int classId;
		float confidence;
		cv::Rect box;
	};

	void configureNet(cv::dnn::Net& net, bool isCuda);

	cv::Mat formatYolov5(const cv::Mat& source);

	cv::Mat objectDetector(cv::Mat& image, cv::dnn::Net& net, const std::vector<std::string>& className);
}
#endif

