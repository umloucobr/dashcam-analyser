#include "car-detecter.hpp"

namespace dashan {
    namespace constants {
        extern const float scoreThreshold {0.2f};
        extern const float NMSThreshold {0.45f};
        extern const float confidenceThreshold {0.45f};
		extern const float inputHeight {640.0};
		extern const float inputWidth {640.0};
    }
	void configureNet(cv::dnn::Net& net, bool isCuda) {
		if (isCuda)
		{
			net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
			net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
		}
	}
	//Part of this code comes from https://github.com/doleron/yolov5-opencv-cpp-python but I have changed it to match good C++ practices.

	//Put the image in a big enough square (Resize to the biggest member).
    cv::Mat formatYolov5(const cv::Mat& source) {
		int col {source.cols};
        int row {source.rows};
		int biggest {std::max(col, row)};

		cv::Mat resized{cv::Mat::zeros(biggest, biggest, CV_8UC3)};
        source.copyTo(resized(cv::Rect(0, 0, col, row)));

        return resized;
    }

	cv::Mat objectDetector(cv::Mat& image, cv::dnn::Net& net, const std::vector<std::string>& classList) {
		cv::Mat blob;
		std::vector<Detection> output;

		auto inputImage {formatYolov5(image)};

		cv::dnn::blobFromImage(inputImage, blob, 1. / 255., cv::Size(dashan::constants::inputHeight, dashan::constants::inputWidth), cv::Scalar(), true, false);

		net.setInput(blob);
		std::vector<cv::Mat> outputs;
		net.forward(outputs, net.getUnconnectedOutLayersNames());

		//This makes the boxes match the resolution of the input image.

		float xFactor {inputImage.cols / dashan::constants::inputHeight};
		float yFactor {inputImage.rows / dashan::constants::inputWidth};

		float* data {reinterpret_cast<float*>(outputs[0].data)};

		const int dimensions {85}; //85 = 80 COCO classes + 1 confidence + xywh.
		const int rows {23200}; //23200 is the standart for 640x640.

		std::vector<int> classIds;
		std::vector<float> confidences;
		std::vector<cv::Rect> boxes;

		for (int i = 0; i <= rows; ++i) {
			float confidence {data[4]};
			if (confidence >= dashan::constants::confidenceThreshold) {

				float* classesScores {data + 5};

				cv::Mat scores (1, classList.size(), CV_32FC1, classesScores);
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

		int detections = output.size();

		for (int i = 0; i < detections; ++i) {
			auto detection{ output[i] };
			auto box{ detection.box };
			auto classId{ detection.classId };
			auto confidence{ detection.confidence };

			cv::rectangle(image, box, cv::Scalar(0, 0, 255), 3);
			cv::rectangle(image, cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), cv::Scalar(0, 0, 255), cv::FILLED);
			cv::putText(image, classList[classId].c_str(), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
		}
		return image;
	}
}