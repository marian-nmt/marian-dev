#pragma once
#include <stdio.h>
#include <execinfo.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>

#ifdef _WIN32
    #define DLLEXPORT extern "C" __declspec(dllexport)
#else
    #define DLLEXPORT extern "C"
#endif

// This is gross but necessary since exporting a C interface. The callback function
// takes an integer which corresponds to the sentence id in the batch along with
// a char* which is the translated version of the sentence at the corresponding batch id.
// The callback function is free to do whatever it pleases with this, including immediately
// calling send response.
DLLEXPORT void translate_async(void* marian, char* sent, void(*callback)(int, const char*, void*), void* userData);

DLLEXPORT void* init(char* path, int device_num);
DLLEXPORT char* translate(void* marian, char* sent);
DLLEXPORT void free_result(char* to_free);
