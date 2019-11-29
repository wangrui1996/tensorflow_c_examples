#include "Network.hpp"
void* callback_func(std::vector<TF_Tensor*> TF_tensors) {
    std::cout << "ff" << std::endl;
    auto detection_classes = static_cast<float_t*>(TF_TensorData(TF_tensors[0]));
    auto detection_scores = static_cast<float_t*>(TF_TensorData(TF_tensors[1]));
    auto detection_boxes = static_cast<float_t*>(TF_TensorData(TF_tensors[2]));
    auto detection_num = static_cast<float_t*>(TF_TensorData(TF_tensors[3]));
    std::cout << detection_scores[0] << " " << detection_classes[0] << " " << detection_boxes[0] << " " << detection_num[0]<< std::endl;
    for (int i = 0; i < int(*detection_num); ++i) {
        int idx = i*4;
        std::cout << detection_boxes[idx +1] << " " << detection_boxes[idx]
        << " " << detection_boxes[idx+3] << " " << detection_boxes[idx+2];
    }



    void * return_values = &TF_tensors;
    return return_values;
}
int main()
{
    std::string modelPath = "frozen_inference_graph.pb";
    std::string imagePath = "/home/rui/demo.jpeg";

    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    input_names.emplace_back("image_tensor");
    output_names.emplace_back("detection_classes");
    output_names.emplace_back("detection_scores");
    output_names.emplace_back("detection_boxes");
    output_names.emplace_back("num_detections");
    Network ssd_network(input_names, output_names);
    ssd_network.LoadGraph(modelPath);

    cv::Mat image;
    image = cv::imread(imagePath);
    cv::resize(image, image, cv::Size(224, 224));
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    //cv::imshow("test", image);
    //cv::waitKey();
    image.convertTo(image, CV_8UC3);
    //std::cout << image.channels;

    clock_t start,end;
int t;



//To do
for (t = 0;  ; t++) {
    start=clock();
    std::cout << std::endl;
    std::cout << t << "s number of time: ";
    ssd_network.Run(image, callback_func);
    end = clock();
    std::cout << "total time " <<  (float) (end - start) * 1000 / CLOCKS_PER_SEC << std::endl;
}
}