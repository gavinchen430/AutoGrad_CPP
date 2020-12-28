#include <stdio.h>
#include <string.h>
#include <memory>
#include <set>
#include <stack>
#include <cmath>
#include <sstream>
#include "FuncWrapper.h"


template<typename ValueType>
ValueType func(ValueType x) {
    return 1 + x / 2;
}

template<typename ValueType>
ValueType func1(ValueType x) {
    return 1 + x * x;
}

template<typename ValueType>
ValueType func2(ValueType x) {
    // 1. 3*x*x
    // 2. 6*x
    // 3. 6
    return 1 + x * x * x;
}

template<typename ValueType>
ValueType func3(ValueType x) {
    return x * log(x);
}

template<typename ValueType>
ValueType func4(ValueType x) {
    return x*exp(x);
}

template<typename ValueType>
ValueType func5(ValueType x) {
    return (1 - exp(-x)) / (1 + exp(-x));
}

int main() {
    printf("grad(1 + x / 2) 0.5: %f \n", (float)grad(func<Box<float>>)(100));
    printf("grad(grad(1 + x * x)) 2: %d\n", (int)grad(grad(func1<Box<int>>))(100));// 2
    printf("grad(grad(grad(1 + x * x * x))) 6: %d\n", (int)grad(grad(grad(func2<Box<int>>)))(100));// 6
    printf("grad(x * log(x)) 5.60517019: %f\n", (float)grad(func3<Box<float>>)(100));// 5.60517019
    printf("grad(grad(x * log(x))) 0.01: %f\n", (float)grad(grad(func3<Box<float>>))(100));// 0.01
    printf("grad(x*exp(x)) 5.436563: %f\n", (float)grad(func4<Box<float>>)(1));// 5.436563
    printf("grad(grad(x*exp(x))) 8.154845: %f\n", (float)grad(grad(func4<Box<float>>))(1));// 8.154845
    printf("grad( (1 - exp(-x)) / (1 + exp(-x)) ) 0.209987: %f\n", (float)grad(func5<Box<float>>)(-2));// 0.209987
    printf("grad(grad( (1 - exp(-x)) / (1 + exp(-x)) )) 0.159925: %f\n", (float)grad(grad(func5<Box<float>>))(-2));// 0.159925
    return 0;
}
