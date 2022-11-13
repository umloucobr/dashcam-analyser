#include "lane-detector.hpp"

namespace dashan {
	namespace constants {
		extern const cv::Mat lowerYellowThreshold {20, 100, 100};
		extern const cv::Mat upperYellowThreshold {20, 255, 255};
		extern const cv::Mat lowerWhiteThreshold {0, 0, 230};
		extern const cv::Mat upperWhiteThreshold {180, 50, 255};
	}
	
	cv::Mat laneDetector(cv::Mat& input, bool drawContours) {
		cv::Mat roiImage;
		cv::Mat imageHsv;
		cv::Mat imageGray;

		cv::Mat maskYellow;
		cv::Mat maskWhite;
		cv::Mat maskBoth;
		cv::Mat finalMask;
		cv::Mat canny;
		std::vector<cv::Vec4i> lines;

		cv::GaussianBlur(input, imageHsv, cv::Size(9, 9), 3, 3);
		cv::cvtColor(imageHsv, imageGray, cv::COLOR_BGR2GRAY);
		cv::cvtColor(imageHsv, imageHsv, cv::COLOR_BGR2HSV);

		//Gets colors white and yellow from the HSV image.
		cv::inRange(imageHsv, dashan::constants::lowerYellowThreshold, dashan::constants::upperYellowThreshold, maskYellow);
		cv::inRange(imageHsv, dashan::constants::lowerWhiteThreshold, dashan::constants::upperWhiteThreshold, maskWhite);

		//Combine both thresholds.
		cv::bitwise_or(maskYellow, maskWhite, maskBoth);
		cv::bitwise_and(imageGray, maskBoth, finalMask);

		int x0 {0};						//xy1	---------- xy2
		int y0 {1080};					//	   /		  \ /
		int x1 {800};					//	  /			   \ /
		int y1 {600};					//xy0 ______________ xy3
		int x2 {1280};
		int y2 {600};
		int x3 {1920};
		int y3 {1080};
		cv::Mat polygonMask {cv::Mat::zeros(input.size(), input.type())};
		std::vector<cv::Point> fillContAll {cv::Point(x0, y0), cv::Point(x1, y1), cv::Point(x2, y2), cv::Point(x3, y3)};

		cv::fillPoly(polygonMask, fillContAll, cv::Scalar(255, 255, 255));

		//Bitwise_and is a little bitch and won't function without this useless conversion.
		cv::cvtColor(finalMask, finalMask, cv::COLOR_GRAY2BGR);
		cv::bitwise_and(polygonMask, finalMask, roiImage);

		cv::Canny(roiImage, canny, 50, 200, 3, true);
		cv::HoughLinesP(canny, lines, 2, CV_PI / 180, 100, 40.0, 5.0);
		
		//Draw the HoughLinesP findings.
		for (size_t i = 0; i < lines.size(); i++) {
			cv::line(input, cv::Point(lines[i][0], lines[i][1]),
			cv::Point(lines[i][2], lines[i][3]), cv::Scalar(0, 0, 255), 3, 8);
		}

		if (drawContours) {
			std::vector<std::vector<cv::Point> > contours;
			std::vector<cv::Vec4i> hierarchy;
			findContours(canny, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

			for (size_t i = 0; i < contours.size(); i++)
			{
				cv::Scalar color = cv::Scalar(0, 0, 255);
				cv::drawContours(input, contours, (int)i, color, 2, cv::LINE_8, hierarchy, 0);
			}
		}
		return input;
	}
}