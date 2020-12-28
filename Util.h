#pragma once

#include "Log.h"

#define unlikely(x) __builtin_expect(!!(x), 0)
#define likely(x) __builtin_expect(!!(x), 1)

#define MVM_CHECK(condition, ...)      \
    do {                               \
        if (unlikely(!(condition))) {  \
            ALOGE(__VA_ARGS__);        \
            abort();                   \
        }                              \
    } while(0)                         \

#define MVM_CHECK_NULL(ptr)            \
    do {                               \
        if (unlikely(NULL == ptr)) {   \
            ALOGE(#ptr " cannot be null");   \
            abort();                   \
        }                              \
    } while(0)                         \

#define MVM_UNUSED(para) (void)para

#define MVM_ABORT(...) \
    do { \
        ALOGF(__VA_ARGS__); \
        abort(); \
    } while(0) \

