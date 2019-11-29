//
// Created by rui on 19-11-13.
//

#ifndef HELLO_TF_NETWORK_HPP
#define HELLO_TF_NETWORK_HPP


#include <iostream>
#include <vector>

#include <c_api.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>

#define UNUSED(x) (void)(x)
typedef void *(*callback)(std::vector<TF_Tensor*>);

class Network{

public:
    void LoadGraph(std::string modelPath);
    void Run(cv::Mat  image, callback func);
    static void Deallocator(void* data, size_t length, void* arg);
    Network(std::vector<std::string> input_names, std::vector<std::string> output_names);

private:
    TF_Session* session;
    TF_Graph* graph;
    std::vector<std::string> input_names, output_names;
};


#endif //HELLO_TF_NETWORK_HPP
