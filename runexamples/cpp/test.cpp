#include "./onnx/include/onnxruntime_cxx_api.h"
// string
#include <string>
// nullable
#include <optional>
#include <map>
#include <functional>
// std::pair
#include <utility>
#include <vector>
#include <iostream>
#include "tokenizer/tokenizer.hpp"
#include "sampler/sample.hpp"


struct State {
    std::vector<Ort::Value>& state;
    std::vector<std::string>& keys;
    std::vector<std::string>& outputkeys;
};

struct StateProbs {
    float* probs;
    State state;
};

struct StateToken {
    std::vector<int32_t> token;
    State state;
};


class RWKV {
    private:
        Ort::Session* session = nullptr;
        Ort::Env env =  Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntimeExample");
        Ort::SessionOptions sessionOptions;
        Ort::AllocatorWithDefaultOptions allocator;

        size_t layers = 0;
        size_t hidden_size = 0;
        size_t num_heads = 0;
        size_t vocab_size = pow(2, 16);
        std::string outputName = "output0";
        
        State* globalState = nullptr;
        

    public:
        RWKV(std::string model_path) {

            // Set up options for the session
             
            sessionOptions.SetIntraOpNumThreads(2);
            sessionOptions.SetInterOpNumThreads(16);
            sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

            // Create a session with the model file
            session = new Ort::Session(env, model_path.c_str(), sessionOptions);
            
            // Print some information about the model
            size_t numInputNodes = session->GetInputCount();
            size_t numOutputNodes = session->GetOutputCount();

            layers = (numInputNodes - 1) / 3;
            hidden_size = session->GetInputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape()[1];
            int stateindex = 0;
            while (session->GetInputTypeInfo(stateindex).GetTensorTypeAndShapeInfo().GetShape().size() < 3) {
                stateindex++;
            }

            for (int i = 0; i < layers*3+1; i++) {
                auto type = session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType();
                
                std::cout << "Output name: " << outputName << std::endl;
                if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
                    auto name = session->GetOutputNameAllocated(0, allocator);
                    outputName = std::string(name.get());
                    std::cout << "Output name: " << outputName << std::endl;
                    break;
                }
            }

            num_heads = session->GetInputTypeInfo(stateindex).GetTensorTypeAndShapeInfo().GetShape()[1];

            std::cout << "Number of inputs: " << numInputNodes << std::endl;
            std::cout << "Number of outputs: " << numOutputNodes << std::endl;
            std::cout << "layers: " << layers << std::endl;
            std::cout << "hidden_size: " << hidden_size << std::endl;
            std::cout << "num_heads: " << num_heads << std::endl;

            
            globalState = newState();
        }
        ~RWKV() {
        }

        State* newState(int32_t batch_size = 1) {
        
        std::vector<Ort::Value>* inputTensors = new std::vector<Ort::Value>();
        std::vector<std::string>* inputNames = new std::vector<std::string>();
        std::vector<std::string>* outputNames = new std::vector<std::string>();

        for(int i = 0; i < layers*2; i++) {
            const auto meminfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            const int64_t size[2] = {batch_size, hidden_size};
            
            auto key = std::string("state" + std::to_string(i));
            //  OrtMemoryInfo* info, T* p_data, size_t p_data_element_count,  int64_t* shape, size_t shape_len
            auto key2 = std::string("state" + std::to_string(i) + "out");
            

            inputNames->push_back(key);
            outputNames->push_back(key2);
            inputTensors->push_back(Ort::Value::CreateTensor<float>(allocator,(const int64_t*)size, size_t(2)));
            // ("float32",new Float32Array(this.embed),[1,this.embed])
            std::fill_n(inputTensors->back().GetTensorMutableData<float>(), hidden_size, 0);
        }   

        for (int i = 0; i < layers; i++) {
            const int64_t size[4] = {batch_size, num_heads, hidden_size / num_heads, hidden_size / num_heads};
            
            auto key = std::string("statewkv" + std::to_string(i));
            // const OrtMemoryInfo* info, T* p_data, size_t p_data_element_count, const int64_t* shape, size_t shape_len
            auto key2 = std::string("statewkv" + std::to_string(i) + "out");
            

            inputNames->push_back(key);
            outputNames->push_back(key2);
            inputTensors->push_back(Ort::Value::CreateTensor<float>(allocator, (const int64_t*)size, size_t(4)));
            // fill with zeros
            std::fill_n(inputTensors->back().GetTensorMutableData<float>(), hidden_size * hidden_size / num_heads, 0);
            
        }

        outputNames->push_back(outputName);
        inputNames->push_back("input0");

        const int64_t size[1] = {batch_size};
        inputTensors->push_back(Ort::Value::CreateTensor<int32_t>(allocator, (const int64_t*)size, size_t(1)));

        

        State* result = new State({
            .state = *inputTensors,
            .keys = *inputNames,
            .outputkeys = *outputNames
        });

        return result;
    }

    void digest(std::vector<State*> input){
        auto runOptions = Ort::RunOptions{nullptr}; 

        auto inputnames = globalState->keys;
        auto outputnames = globalState->outputkeys;

        for (int i = 0; i < layers*3 + 1; i++){
            for (int j = 0; j < input.size(); j++) {
                std::copy(input[j]->state[i].GetTensorMutableData<float>(), input[j]->state[i].GetTensorMutableData<float>() + input[j]->state[i].GetTensorTypeAndShapeInfo().GetElementCount(), globalState->state[i].GetTensorMutableData<float>() + j * input[j]->state[i].GetTensorTypeAndShapeInfo().GetElementCount() * sizeof(float));
            }
        }

        const char** inputNamesChar = new const char*[inputnames.size()];
        for (int i = 0; i < inputnames.size(); i++) {
            inputNamesChar[i] = inputnames[i].c_str();
        }

        const char** outputNamesChar = new const char*[outputnames.size()];
        for (int i = 0; i < outputnames.size(); i++) {
            outputNamesChar[i] = outputnames[i].c_str();
        }

        
        std::vector<Ort::Value> outTensors = session->Run(runOptions, inputNamesChar, globalState->state.data(), layers*3 + 1, outputNamesChar, outputnames.size());

        for (int i = 0; i < layers*3; i++) {
            for (int j = 0; j < input.size(); j++) {
                std::copy(outTensors[i].GetTensorMutableData<float>() + j * input[j]->state[i].GetTensorTypeAndShapeInfo().GetElementCount(), outTensors[i].GetTensorMutableData<float>() + (j+1) * input[j]->state[i].GetTensorTypeAndShapeInfo().GetElementCount(), input[j]->state[i].GetTensorMutableData<float>());
            }    
        }
        // std::vector<Ort::Value ns = std::vector<Ort::Value>(outTensors.begin(), outTensors.begin() + layers*3);
        // slice(0,-1)

        float* probs = outTensors[layers*3].GetTensorMutableData<float>();
        auto samp = typical(input.size(),probs, 0.9, 0.9);
        for(int i = 0; i < input.size(); i++) {
            input[i]->state[layers*3].GetTensorMutableData<int32_t>()[0] = samp[i];
        }
    }
};

int main() {
    std::cout << "Loading ONNX model..." << std::endl;
    // Initialize ONNX Runtime (optional, depending on the version of ONNX Runtime)
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntimeExample");

    // Set up options for the session
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(6);
    sessionOptions.SetInterOpNumThreads(6);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);


    try {
        // Create a session with the model file
        auto file = "RWKV_32_2560_32_17_QUInt8-pc-norr-ext.onnx";
        RWKV* rwkv = new RWKV(file);

        std::cout << "Creating state..." << std::endl;
        State* state = rwkv->newState();

        std::cout << "Creating token..." << std::endl;

        auto context = worldTokenizer.encode("### Instruction:\nWrite a story about a man going to fight in the war against the puppies, showcasing the atrocities of war.\n### Response:\n");

        state->state.back().GetTensorMutableData<int32_t>()[0] = context[0];
       

        std::cout << "Digesting token..." << std::endl;
        std::vector<State*> input = {state};
        int i = 0;
        while(1){
            rwkv->digest(input);
            if(i < context.size()){
                state->state.back().GetTensorMutableData<int32_t>()[0] = context[i];
                i++;
            }
            std::cout << worldTokenizer.decode({state->state.back().GetTensorMutableData<int32_t>()[0]}) << std::flush;
        }

        std::cout << "Done!" << std::endl;
        

    } catch (const Ort::Exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}