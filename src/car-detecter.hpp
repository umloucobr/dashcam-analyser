#pragma once
#ifndef CAR_DETECTOR_H
#define CAR_DETECTOR_H
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <format>
#include <fstream>
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

	void detect(cv::Mat& image, cv::dnn::Net& net, std::vector<Detection>& output, const std::vector<std::string>& className);
}
#endif

