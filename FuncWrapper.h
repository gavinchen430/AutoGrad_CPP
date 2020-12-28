#include <functional>
#include "Box.h"

template<typename F>
struct FunctionTraits;

template<typename ClassType, typename R, typename... Args>
struct FunctionTraits<R(ClassType::*)(Args...)>
{
    FunctionTraits() {}
    using RetType = R;
    using ArgTypes = std::tuple<Args...>;
    ClassType mCT;
    FunctionTraits(ClassType& ct) : mCT(ct) {
    }

    R operator()(Args... args) {
        auto r = mCT(args.incTraceID()...);
        return backward_pass<R>(r.decTraceID());
    }
};

template<typename F>
struct FunctionTraits : FunctionTraits<decltype(&F::operator())> {
    FunctionTraits(F& f) : FunctionTraits<decltype(&F::operator())>(f){
    }
};


template<typename R, typename... Args>
struct FunctionTraits<R(*)(Args...)>
{
    FunctionTraits() {
    }
    using Pointer = R(*)(Args...);
    Pointer mP;
    FunctionTraits(Pointer p) : mP(p) {
    }

    R operator()(Args... args) {
        auto r = mP(args.incTraceID()...);
        return backward_pass<R>(r.decTraceID());
    }
};

template<typename T>
FunctionTraits<T> grad(T t) {
    return FunctionTraits<T>(t);
}

