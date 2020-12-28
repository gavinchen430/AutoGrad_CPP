#pragma once

#include <stdio.h>
#include <stdarg.h>
#include <sys/syscall.h>
#include <unistd.h>

#define MVM_VERBOSE 1
#define MVM_DEBUG 2
#define MVM_INFO 3
#define MVM_WARN 4
#define MVM_ERROR 5
#define MVM_FATAL 6

inline void mvm_printf(int level, const char* file,
        const char* func, int lineNum, const char* fmt, ...) {
    if (fmt == NULL) {
        return;
    }
    char buf[256 * 2];

    va_list arg;
    va_start(arg, fmt);
    vsprintf(buf, fmt, arg);
    va_end(arg);
    printf("%s--%s:%d | %s\n", file, func, lineNum, buf);
}

#define ALOG(level, ...) mvm_printf(level, __FILE__,  __FUNCTION__, (int)__LINE__, __VA_ARGS__);

#define ALOGV(...) ALOG(MVM_VERBOSE, __VA_ARGS__);
#define ALOGD(...) ALOG(MVM_DEBUG, __VA_ARGS__);
#define ALOGI(...) ALOG(MVM_INFO, __VA_ARGS__);
#define ALOGW(...) ALOG(MVM_WARN, __VA_ARGS__);
#define ALOGE(...) ALOG(MVM_ERROR, __VA_ARGS__);
#define ALOGF(...) ALOG(MVM_FATAL, __VA_ARGS__);
