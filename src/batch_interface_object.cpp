#include "Network.hpp"



void DeallocateBuffer(void* data, size_t) {
    std::free(data);
}

TF_Buffer* ReadBufferFromFile(std::string file) {
    std::ifstream f(file, std::ios::binary);
    if (f.fail() || !f.is_open()) {
        return nullptr;
    }

    f.seekg(0, std::ios::end);
    const auto fsize = f.tellg();
    f.seekg(0, std::ios::beg);

    if (fsize < 1) {
        f.close();
        return nullptr;
    }

    char* data = static_cast<char*>(std::malloc(fsize));
    f.read(data, fsize);
    f.close();

    TF_Buffer* buf = TF_NewBuffer();
    buf->data = data;
    buf->length = fsize;
    buf->data_deallocator = DeallocateBuffer;
    return buf;
}



Network::Network(std::vector<std::string> input_names, std::vector<std::string> output_names)
{
    this->input_names = input_names;
    this->output_names = output_names;
}

void Network::LoadGraph(std::string modelPath)
{
    TF_Buffer* buffer = ReadBufferFromFile(modelPath);
    if (buffer == nullptr) {
        throw std::invalid_argument("Error creating the session from the given model path %s !");
    }
    TF_Status* status = TF_NewStatus();
    TF_ImportGraphDefOptions* opts = TF_NewImportGraphDefOptions();

    graph = TF_NewGraph();
    TF_GraphImportGraphDef(graph, buffer, opts, status);
    TF_DeleteImportGraphDefOptions(opts);
    TF_DeleteBuffer(buffer);
    if (TF_GetCode(status) != TF_OK) {
        TF_DeleteGraph(graph);
        graph = nullptr;
    }
    TF_DeleteStatus(status);

    // create session from graph
    status = TF_NewStatus();
    TF_SessionOptions* options = TF_NewSessionOptions();
    session = TF_NewSession(graph, options, status);
    TF_DeleteSessionOptions(options);
}

void Network::Run(cv::Mat image, callback func)
{
    std::vector<TF_Output> 	input_tensors, output_tensors;
    std::vector<TF_Tensor*> input_values, output_values;

    //input tensor shape.
    int num_dims = 4;
    std::int64_t input_dims[4] = {1, image.rows, image.cols, 3}; //1 is number of batch, and 3 is the no of channels.
    int num_bytes_in = image.cols * image.rows * 3; //3 is the number of channels.
    input_tensors.push_back({TF_GraphOperationByName(graph, input_names[0].c_str()),0});
    input_values.push_back(TF_NewTensor(TF_UINT8, input_dims, num_dims, image.data, num_bytes_in, &Deallocator, 0));
    size_t i;
    for (i = 0; i < output_names.size(); ++i) {
        output_tensors.push_back({ TF_GraphOperationByName(graph, output_names[i].c_str()),0 });
        output_values.push_back(nullptr);
    }

    TF_Status* status = TF_NewStatus();
    TF_SessionRun(session, nullptr,
                  &input_tensors[0], &input_values[0], input_values.size(),
                  &output_tensors[0], &output_values[0], 4, //3 is the number of outputs count..
                  nullptr, 0, nullptr, status
    );
    if (TF_GetCode(status) != TF_OK)
    {
        printf("ERROR: SessionRun: %s", TF_Message(status));
    }

    (*func)(output_values);

}

void Network::Deallocator(void* data, size_t length, void* arg)
{
    UNUSED(length);
    UNUSED(arg);
    UNUSED(data);
}

