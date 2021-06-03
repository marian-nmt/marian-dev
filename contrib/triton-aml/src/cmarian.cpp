#include "marian.h"
#include "translator/beam_search.h"
#include "translator/translator.h"
#include "common/utils.h"

#include<stdio.h>
#include<string.h>
#include<iostream>
#include <string>

#ifdef _WIN32
    #define DLLEXPORT extern "C" __declspec(dllexport)
#else
    #define DLLEXPORT extern "C"
#endif

using namespace marian;
typedef void (*callbackFunc)(int, const char*, void*);

class CMarian {
private:
    Ptr<Options> options_;
    char* configPath_;
    Ptr<TranslateService<BeamSearch>> task_;
    Ptr<TranslateServiceAsync<BeamSearch>> taskAsync_;

public:
    CMarian(char* configPath, int device_num) : configPath_(configPath) {
        int argc = 5;
        char** argv = new char*[argc];
        argv[0] = new char[20];
        strcpy(argv[0], "./marian-decoder");
        argv[1] = new char[12];
        strcpy(argv[1], "--config");
        argv[2] = configPath_;
        argv[3] = new char[12];
        strcpy(argv[3], "--devices");
        argv[4] = new char[sizeof(device_num) + 1];
        strcpy(argv[4], std::to_string(device_num).c_str());

        options_ = marian::parseOptions(argc, argv, cli::mode::translation, true);
        delete[] argv[0];
        delete[] argv[1];
        delete[] argv[3];
        delete[] argv[4];
    }

    /**
     * @brief Exposes Marian translation capabilities based on the loaded YAML config associated with this class.
     * @param sent The sentence to run inference on.
     * @return A string delimited by ||| with newlines separating beams.
     */
    char* translate(char* sent) {
        // Lazy initialize so we can use either the sync or async version with this repo
        if(!task_) {
            task_ = New<TranslateService<BeamSearch>>(options_);
        }
        std::string strSent(sent);
        auto outputText = task_->run(strSent);
        char* ret = (char*) malloc(outputText.length() + 1);
        snprintf(ret, outputText.length() + 1, "%s", outputText.c_str());
        return ret;
    }

    /**
     * @brief Exposes Marian translation capabilities based on the loaded YAML config associated with this class.
     * @param sent The sentence to run inference on.
     * @param callback the function to run on the translated sentence. It takes the batchID of the source sentence 
     *                   along with the translated sentence and processes those.
     * @param userData A pointer to any state the user needs to use in their callback
     */
    void translate_async(char* sent, callbackFunc callback, void* userData) {
        // Lazy initialize so we can use either the sync or async version with this repo
        if (!taskAsync_) {
            taskAsync_ = New<TranslateServiceAsync<BeamSearch>>(options_);
        }
        std::string strSent(sent);
        taskAsync_->run(strSent, callback, userData);
    }
};

DLLEXPORT void* init(char* path, int device_num) {
    CMarian* m = new CMarian(path, device_num);
    return (void*)m;
}

DLLEXPORT char* translate(void* marian, char* sent) {
    CMarian* m = static_cast<CMarian*>(marian);
    return m->translate(sent);
}

DLLEXPORT void translate_async(void* marian, char* sent, callbackFunc callback, void* userData) {
    CMarian* m = static_cast<CMarian*>(marian);
    m->translate_async(sent, callback, userData);
}

DLLEXPORT void free_result(char* to_free) {
    free(to_free);
}
