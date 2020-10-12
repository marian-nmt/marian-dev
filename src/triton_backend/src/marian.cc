// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <chrono>
#include <string>
#include <thread>
#include <algorithm>
#include <string.h>

#include "src/core/model_config.h"
#include "src/core/model_config.pb.h"
#include "src/custom/sdk/custom_instance.h"
#include "src/custom/marian/marian.h"

#ifdef TRTIS_ENABLE_GPU
#include <cuda_runtime_api.h>
#endif // TRTIS_ENABLE_GPU

#define LOG_ERROR std::cerr
#define LOG_INFO std::cout

// This custom backend returns the translate results from marian framework.

namespace nvidia { namespace inferenceserver { namespace custom {
namespace marian {

// Context object. All state must be kept in this object.
class Context : public CustomInstance {
public:
    Context(
        const std::string& instance_name, const ModelConfig& config,
        const int gpu_device);
    ~Context() = default;

    // Validate the model configuration for the derived backend instance
    int Init();

#ifdef TRTIS_ENABLE_GPU
    // Version 2 interface may need to deal with data in GPU memory,
    // which requires CUDA support.
    int Execute(
        const uint32_t payload_cnt, CustomPayload* payloads,
        CustomGetNextInputV2Fn_t input_fn,
        CustomGetOutputV2Fn_t output_fn) override;
#else
    int Execute(
        const uint32_t payload_cnt, CustomPayload* payloads,
        CustomGetNextInputFn_t input_fn, CustomGetOutputFn_t output_fn) override;
#endif  // TRTIS_ENABLE_GPU

private:
#ifdef TRTIS_ENABLE_GPU
    int GetInputTensor(
        CustomGetNextInputV2Fn_t input_fn, void* input_context, const char* name,
        const uint32_t batch_size, std::vector<std::string>& input);
#else
    int GetInputTensor(
        CustomGetNextInputFn_t input_fn, void* input_context, const char* name,
        const uint32_t batch_size, std::vector<std::string>& input);
#endif

    int Process(const char* input, char** output);

    // Accumulators maintained by this context, one for each batch slot.
    std::vector<int32_t> accumulator_;

    // The shape of marian's output
    std::vector<int64_t> output_shape_;

    // The data type of marian's output
    DataType output_type_;

    void* marian_ = nullptr;

    std::string marian_config_path_;

#ifdef TRTIS_ENABLE_GPU
    cudaStream_t stream_;
#endif  // TRTIS_ENABLE_GPU

    // local Error Codes
    const int kSequenceBatcher = RegisterError("model configuration must configure sequence batcher");
    const int kInputName = RegisterError("model input must be named 'INPUT'");
    const int kOutputName = RegisterError("model output must be named 'OUTPUT'");
    const int kInputOutput = RegisterError("model input/output pairs mut be name 'INPUTn' and 'OUTPUTn'");
    const int kInputOutputDataType =
        RegisterError("model input and output must have TYPE_STRING data-type");
    const int kInputContents = RegisterError("unable to get input tensor values");
    const int kInputSize = RegisterError("unexpected size for input tensor");
    const int kInputBuffer = RegisterError("unable to get buffer for input tensor values");
    const int kOutputBuffer = RegisterError("unable to get buffer for output tensor values");
    const int kBatchTooBig = RegisterError("unable to execute batch larger than max-batch-size");
};

Context::Context(
    const std::string& instance_name, const ModelConfig& model_config,
    const int gpu_device)
    : CustomInstance(instance_name, model_config, gpu_device)
{
    if (model_config_.parameters_size() > 0) {
        const auto itr = model_config_.parameters().find("config_filepath");
        if (itr != model_config_.parameters().end()) {
            std::string config_path("/var/azureml-app/");
            config_path.append(std::getenv("AZUREML_MODEL_DIR"));
            config_path.append("/nlxseq2seq/triton/nlxseq2seq/1/data/model/");
            config_path.append(itr->second.string_value());
            marian_config_path_ = config_path;
        }
    }

    accumulator_.resize(std::max(1, model_config_.max_batch_size()));

#ifdef TRTIS_ENABLE_GPU
    int device_cnt;
    auto cuerr = cudaGetDeviceCount();
    // Do nothing if there is no CUDA device since all data transfer will be done
    // within CPU memory
    if ((cuerr != cudaErrorNoDevice) && (cuerr != cudaErrorInsufficienDriver)) {
        if (cuerr == cudaSuccess) {
            cuerr = cudaStreamCreate(&stream_);
        }
        if (cuerr != cudaSuccess) {
            stream_ = nullptr;
        }
    }
#endif  // TRTIS_ENABLE_GPU
}

int
Context::Init()
{
    if (model_config_.input(0).name() != "INPUT") {
        return kInputName;
    }
    if (model_config_.input(0).data_type() != DataType::TYPE_STRING) {
        return kInputOutputDataType;
    }
    if (model_config_.output(0).name() != "OUTPUT") {
        return kOutputName;
    }
    if (model_config_.output(0).data_type() != DataType::TYPE_STRING) {
        return kInputOutputDataType;
    }
    output_type_ = model_config_.output(0).data_type();

    marian_ = init(const_cast<char*>(marian_config_path_.c_str()));

    return ErrorCodes::Success;
}

#ifdef TRTIS_ENABLE_GPU
int
Context::GetInputTensor(
    CustomGetNextInputV2Fn_t input_fn, void* input_context, const char* name,
    const uint32_t batch_size, std::vector<std::string>& input)
{
    return ErrorCodes::Success;
}
#else
int
Context::GetInputTensor(
    CustomGetNextInputFn_t input_fn, void* input_context, const char* name,
    const uint32_t batch_size, std::vector<std::string>& input)
{
    // The size of a STRING data type can only be obtained from the data
    // (convention: first 4 bytes stores the size of the actual data)
    uint32_t content_size = 0;
    uint32_t byte_read = 0;
    std::vector<char> size_buffer;
    std::vector<char> content_buffer;

    while (true) {
        const void* content;
        // Get all content out
        uint64_t content_byte_size = -1;
        if (!input_fn(input_context, name, &content, &content_byte_size)) {
            return kInputBuffer;
        }

        // If 'content' returns nullptr we have all the input.
        if (content == nullptr) {
            break;
        }

        // Keep consuming the content because we want to decompose the batched input
        while (content_byte_size > 0) {
            // If there are input and 'content_size' is not set, try to read content_size
            if (content_size == 0) {
                // Make sure we have enough bytes to read as 'content_size'
                uint64_t byte_to_append = 4 - size_buffer.size();
                byte_to_append = (byte_to_append < content_byte_size)
                                    ? byte_to_append
                                    : content_byte_size;
                size_buffer.insert(
                    size_buffer.end(), static_cast<const char*>(content),
                    static_cast<const char*>(content) + byte_to_append);

                // modify position to unread content
                content = static_cast<const char*>(content) + byte_to_append;
                content_byte_size -= byte_to_append;
                if (size_buffer.size() == 4) {
                    content_size = *(uint32_t*)(&size_buffer[0]);
                    byte_read = 0;
                    size_buffer.clear();
                } else {
                    break;
                }
            }

            uint32_t byte_to_read = content_size - byte_read;
            byte_to_read =
                (byte_to_read < content_byte_size) ? byte_to_read : content_byte_size;

            content_buffer.insert(
                content_buffer.end(), static_cast<const char*>(content),
                static_cast<const char*>(content) + byte_to_read);

            // modify position to unread content
            content = static_cast<const char*>(content) + byte_to_read;
            content_byte_size -= byte_to_read;
            byte_read += byte_to_read;
            if (byte_read == content_size) {
                std::string s(content_buffer.begin(), content_buffer.end());
                input.push_back(s);
                content_size = 0;
                content_buffer.clear();
            }
        }
    }

    // Make sure we end up with exactly the amount of input we expect.
    if (batch_size != input.size()) {
        return kInputSize;
    }

    return ErrorCodes::Success;
}
#endif

#ifdef TRTIS_ENABLE_GPU
int
Context::Execute(
    const uint32_t payload_cnt, CustomPayload* payloads,
    CustomGetNextInputV2Fn_t input_fn, CustomGetOutputV2Fn_t output_fn)
{
    return ErrorCodes::Success;
}
#else
int
Context::Execute(
    const uint32_t payload_cnt, CustomPayload* payloads,
    CustomGetNextInputFn_t input_fn, CustomGetOutputFn_t output_fn)
{
    if (payload_cnt > accumulator_.size()) {
        return kBatchTooBig;
    }

    // Do the real batch
    std::string input_strings;
    for (size_t idx = 0; idx < payload_cnt; idx++) {
        CustomPayload& payload = payloads[idx];
        // If output wasn't requested just do nothing.
        if (payload.output_cnt == 0) {
            continue;
        }

        // Reads input
        uint32_t batch_size =
            (payload.batch_size == 0) ? 1 : payload.batch_size;
        std::vector<std::string> input;
        int err = GetInputTensor(
            input_fn, payload.input_context, "INPUT", batch_size, input);
        if (err != ErrorCodes::Success) {
            payload.error_code = err;
            continue;
        }

        // Correct batch size of the payload according to the '\n' count.
        int count = std::count(input[0].begin(), input[0].end(), '\n');
        payload.batch_size = count + 1;

        // Should be only one item.
        for (auto& item : input) {
            if (input_strings.empty()) {
                input_strings = item;
            } else {
                input_strings.append("\n");
                input_strings.append(item);
            }
        }
    }

    char* result;
    int err = Process(input_strings.c_str(), &result);
    if (err != ErrorCodes::Success) {
        return err;
    }
    char* pos = result;

    for (size_t idx = 0; idx < payload_cnt; idx++) {
        CustomPayload& payload = payloads[idx];
        // Find current output content
        char *output_content = nullptr;
        uint32_t byte_size = 0;
        while (payload.batch_size > 0) {
            char* p = strchr(pos, '\n');
            if (p != nullptr) {
                *p = '\0';
            }
            if (output_content == nullptr) {
                output_content = pos;
            } else {
                strcat(output_content, "\n");
                strcat(output_content, pos);
            }
            byte_size = strlen(output_content);
            // Move to next output content
            if (p != nullptr) {
                pos = p + 1;
            }
            payload.batch_size--;
        }

        // Obtain the output buffer for the whole batch
        payload.batch_size = 1; // hard code here, batch_size of every request is one.
        std::vector<int64_t> output_shape = output_shape_;
        output_shape.insert(output_shape.begin(), payload.batch_size);

        void *obuffer;
        if (!output_fn(
                payload.output_context,
                payload.required_output_names[0], output_shape.size(),
                &output_shape[0], byte_size + 4,
                &obuffer)) {
            payload.error_code = kOutputBuffer;
            continue;
        }

        // If there is no error but the 'obuffer' is returned as nullptr,
        // then skip writing this output.
        if (obuffer != nullptr) {
            memcpy(obuffer, reinterpret_cast<const char*>(&byte_size), 4);
            memcpy(static_cast<char*>(obuffer) + 4, output_content, byte_size);
        }
    }
    free_result(result);

    return ErrorCodes::Success;
}
#endif

int
Context::Process(const char* input, char** output)
{
    *output = translate(marian_, const_cast<char*>(input));

    return ErrorCodes::Success;
}

}  // namespace marian

// Creates a new Marian context instance
int
CustomInstance::Create(
    CustomInstance** instance, const std::string& name,
    const ModelConfig& model_config, int gpu_device,
    const CustomInitializeData* data)
{
    marian::Context* ctx =
        new marian::Context(name, model_config, gpu_device);

    *instance = ctx;

    if (ctx == nullptr) {
        return ErrorCodes::CreationFailure;
    }

    return ctx->Init();
}

/////////////

extern "C" {

uint32_t
CustomVersion()
{
#ifdef TRTIS_ENABLE_GPU
    return 2;
#else
    return 1;
#endif  // TRTIS_ENABLE_GPU
}

}  // extern "C"

}}}  // namespace nvidia::inferenceserver::custom
