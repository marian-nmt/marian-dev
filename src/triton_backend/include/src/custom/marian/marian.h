#pragma once

#ifdef _WIN32
    #define DLLEXPORT extern "C" __declspec(dllexport)
#else
    #define DLLEXPORT extern "C"
#endif

DLLEXPORT void* init(char* path);
DLLEXPORT char* translate(void* marian, char* sent);
DLLEXPORT void free_result(char* to_free);
