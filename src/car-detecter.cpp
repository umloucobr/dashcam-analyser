#include "car-detecter.hpp"

namespace dashan {
    namespace constants {
        extern const float scoreThreshold {0.2};
        extern const float NMSThreshold {0.45};
        extern const float confidenceThreshold {0.45};
    }
	//Part of this code comes from learnopencv.com but I have changed it to match good C++ practices.

	//Put the image in a big enough square (Resize to the biggest member).
    cv::Mat formatYolov5(const cv::Mat& source) {
		int col {source.cols};
        int row {source.rows};
        int biggest = std::max(col, row);

        cv::Mat resized = cv::Mat::zeros(biggest, biggest, CV_8UC3);
        source.copyTo(resized(cv::Rect(0, 0, col, row)));

        return resized;
    }

	void detect(cv::Mat& image, cv::dnn::Net& net, std::vector<Detection>& output, const std::vector<std::string>& className) {
		cv::Mat blob;

		auto inputImage {formatYolov5(image)};

		cv::dnn::blobFromImage(inputImage, blob, 1. / 255., cv::Size(640, 640), cv::Scalar(), true, false);

		net.setInput(blob);
		std::vector<cv::Mat> outputs;
		net.forward(outputs, net.getUnconnectedOutLayersNames());

		//This makes the boxes match the resolutionof the input image.

		float xFactor {inputImage.cols / 640.0f};
		float yFactor {inputImage.rows / 640.0f};

		float* data {reinterpret_cast<float*>(outputs[0].data)};

		const int dimensions {85}; //85 = 80 COCO classes + 1 confidence + xywh.
		const int rows {23200}; //23200 is the standart for 640x640.

		std::vector<int> classIds;
		std::vector<float> confidences;
		std::vector<cv::Rect> boxes;

		for (int i = 0; i < rows; ++i) {

			float confidence {data[4]};
			if (confidence >= dashan::constants::confidenceThreshold) {

				float* classesScores {data + 5};

				cv::Mat scores(1, className.size(), CV_32FC1, classesScores);
				cv::Point classId;
				double maxClassScore;

				cv::minMaxLoc(scores, 0, &maxClassScore, 0, &classId);
				if (maxClassScore > dashan::constants::scoreThreshold) {

					confidences.push_back(confidence);
					classIds.push_back(classId.x);

					float x {data[0]};
					float y {data[1]};
					float w {data[2]};
					float h {data[3]};
					int left {int((x - 0.5 * w) * xFactor)};
					int top {int((y - 0.5 * h) * yFactor)};
					int width {int(w * xFactor)};
					int height {int(h * yFactor)};

					boxes.push_back(cv::Rect(left, top, width, height));
				}
			}
			data += 85;
		}
		//Remove overlapping boxes.

		std::vector<int> nmsResult;
		cv::dnn::NMSBoxes(boxes, confidences, dashan::constants::scoreThreshold, dashan::constants::NMSThreshold, nmsResult);
		for (int i = 0; i < nmsResult.size(); i++) {
			int idx = nmsResult[i];
			Detection result;
			result.classId = classIds[idx];
			result.confidence = confidences[idx];
			result.box = boxes[idx];
			output.push_back(result);
		}
	}
}