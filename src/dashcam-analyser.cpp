#include "dashcam-analyser.hpp"
#include "car-detecter.hpp"
#include "lane-detector.hpp"

int main(int argc, char* argv[])
{
    cv::String pathToVideo {"c.mp4"};
    bool useCuda {true};

	cv::namedWindow("Dashcam Analyser", cv::WINDOW_AUTOSIZE);
	cv::namedWindow("Dashcam Analyser 2", cv::WINDOW_AUTOSIZE);

    if (argc > 1)
    {
        const cv::String keys {
            "{help h usage ? |      | Print this message}"
            "{@path			 |<none>| Path of the video/camera}"
            "{cu cuda		 |      | Use CUDA}"};

        cv::CommandLineParser parser(argc, argv, keys);
        parser.about("Dashcam Analyser V0.2");

        pathToVideo = parser.get<cv::String>(0);

        if (parser.has("help"))
        {
            parser.printMessage();
            return 0;
        }

        if (parser.has("cu"))
        {
            useCuda = true;
        }
    }

    std::vector<std::string> classList;
    std::ifstream ifs("coco.names");
    std::string line;
    while (getline(ifs, line)) {
        classList.push_back(line);
    }

    cv::VideoCapture video {pathToVideo};

    if (!video.isOpened()) {
        std::cerr << "Error when opening the video." << std::endl;
        return -1;
    }

    auto net = cv::dnn::readNet("yolov5x6.onnx");
    dashan::configureNet(net, useCuda);

    cv::Mat frame;
   
	while (true) {
        video >> frame;
        if (frame.empty()) {
            return 0;
        }

        dashan::objectDetector(frame, net, classList);
        dashan::laneDetector(frame, true);
        cv::imshow("Dashcam Analyser 2", frame);

        if (static_cast<int>(cv::waitKey(33)) == 27) {
            cv::destroyAllWindows();
            return 0;
        }       
	}
	return 0;
}
