#pragma once

#include "Log.h"
#include "Object.h"
#include "Util.h"

namespace mvm {

enum TypeCode {
    TInt,
    TFloat,
};

inline std::string type2Str(TypeCode code) {
    static std::map<TypeCode, std::string> s = {
        {TInt, "int"},
        {TFloat, "float"},
    };
    return s[code];
}

class TensorNode : public ObjectNode {
};

class Tensor : public Object {
public:
    Tensor() {
    }

    MVM_OBJECT_METHODS(Tensor, Object, TensorNode)
    typedef TensorNode NodeType;

    Tensor(int i) : mType(TInt) {
        mV.i = i;
    }
    Tensor(float f) : mType(TFloat) {
        mV.f = f;
    }
#define TENSOR_SCALAR_UPCAST(Type) \
    if (mType <= Type) {\
        if (Type == TInt) { \
            return mV.i; \
        } else if (Type == TFloat) {\
            switch(mType) { \
            case TInt : \
                return (int)mV.i; \
            case TFloat: \
                return mV.f; \
            } \
        } \
    } else {\
        MVM_CHECK(false, "cannot cast from %s to %s\n", type2Str(mType).c_str(), type2Str(Type).c_str()); \
    }

    operator int() const {
        checkScalar();
        TENSOR_SCALAR_UPCAST(TInt);
        return mV.i;
    }
    operator float() const {
        checkScalar();
        TENSOR_SCALAR_UPCAST(TFloat);
        return mV.f;
    }
    inline bool isScalar() const {
        return mShape.empty();
    }
    inline void checkScalar() const {
        MVM_CHECK(isScalar(), "This is not a scalar");
    }
    std::vector<int> shape() const {
        return mShape;
    }
    TypeCode type() const {
        return mType;
    }
private:
    union V {
        char c;
        int i;
        float f;
        double d;
    } mV;
    std::vector<int> mShape;
    TypeCode mType;
};

#define DEFINE_TENSOR_BINARY_OPERATOR(OP) \
    inline Tensor OP(int i, const Tensor& t) { \
        return OP(Tensor(i), t); \
    } \
    \
    inline Tensor OP(const Tensor& t, int i) { \
        return OP(Tensor(i), t); \
    }\
    inline Tensor OP(float i, const Tensor& t) { \
        return OP(Tensor(i), t); \
    } \
    \
    inline Tensor OP(const Tensor& t, float i) { \
        return OP(Tensor(i), t); \
    }\
    \

        //MVM_CHECK(t1.type() == t2.type(), "Upcast not support now, t1:%s, t2:%s", type2Str(t1.type()).c_str(), type2Str(t2.type()).c_str());
#define DEFINE_TENSOR_BINARY_OPERATOR_T(OP) \
    inline Tensor operator OP(const Tensor& t1, const Tensor& t2) { \
        MVM_CHECK((t1.isScalar() == t2.isScalar()), "Broadcast not support now"); \
        if (t1.isScalar()) { \
            TypeCode maxType = t1.type() < t2.type() ? t2.type() : t1.type(); /*Upcast*/\
            switch(maxType) { \
            case TInt: \
                printf("t1 "#OP" t2 : %d %d\n", (int)t1, (int)t2); \
                return Tensor((int)t1 OP (int)t2); \
            case TFloat: \
                printf("t1 "#OP" t2 : %f %f\n", (float)t1, (float)t2); \
                return Tensor((float)t1 OP (float)t2); \
            default:\
                return Tensor(-1); \
            } \
        } else { \
            MVM_CHECK(t1.type() == t2.type(), "type should be same for array"); \
        } \
    } \


DEFINE_TENSOR_BINARY_OPERATOR_T(+);
DEFINE_TENSOR_BINARY_OPERATOR_T(-);
DEFINE_TENSOR_BINARY_OPERATOR_T(*);
DEFINE_TENSOR_BINARY_OPERATOR_T(/);

DEFINE_TENSOR_BINARY_OPERATOR(operator+);
DEFINE_TENSOR_BINARY_OPERATOR(operator-);
DEFINE_TENSOR_BINARY_OPERATOR(operator*);
DEFINE_TENSOR_BINARY_OPERATOR(operator/);

inline Tensor operator-(const Tensor& t) {
    switch(t.type()) {
    case TInt:
        return Tensor(-(int)t);
    case TFloat:
        return Tensor(-(float)t);
    default:
        return Tensor(-1);
    }
}

#define DEFINE_TENSOR_UNARY_OPERATOR(OP) \
    inline Tensor OP(const Tensor& t) { \
        if (t.isScalar()) { \
            switch(t.type()) { \
            case TInt: \
                printf("t1 "#OP" : %d %f\n", (int)t, std::OP((int)t)); \
                return Tensor((float)std::OP((int)t)); \
            case TFloat: \
                return Tensor(std::OP((float)t)); \
            default:\
                return Tensor(-1); \
            } \
        } \
        return Tensor(-1); \
    }\

DEFINE_TENSOR_UNARY_OPERATOR(log);
DEFINE_TENSOR_UNARY_OPERATOR(exp);

//class Buffer {
//public:
//    Buffer() {
//    }
//    void resize(const std::vector<int>& shape) {
//        int size = 0;
//        if (!shapeSame(mShape, shape) && (size = shapeSize(shape))) {
//            if (mBuffer != nullptr) {
//                delete[] mBuffer;
//            }
//            mBuffer = new char[size];
//            mShape = shape;
//        }
//
//    }
//    const char* data() {
//        MVM_CHECK(mBuffer, "Buffer is NULL");
//        return mBuffer;
//    }
//    char* mutableData() {
//        MVM_CHECK(mBuffer, "Buffer is NULL");
//        return mBuffer;
//    }
//private:
//    std::vector<int> mShape;
//    char* mBuffer{nullptr};
//};



} // namespace mvm
