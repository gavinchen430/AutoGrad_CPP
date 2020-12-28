#pragma once

#include <vector>
#include <map>
#include "Util.h"
#include "Tensor.h"

inline bool shapeSame(const std::vector<int>& s1,
        const std::vector<int>& s2) {
    if (s1.size() == s2.size()) {
        for (int i = 0; i < s1.size(); ++i) {
            if (s1[i] != s2[i]) {
                return false;
            }
        }
        return true;
    } else {
        return false;
    }
}

inline bool shapeSize(const std::vector<int>& s) {
    if (!s.empty()) {
        int accum = 1;
        for (int i = 0; i < s.size(); ++i) {
            accum *= s[i];
        }
        return accum > 0 ? accum : 0;
    }
    return 0;
}

using namespace mvm;

enum GradType {
    GRAD_ADD,
    GRAD_SUB,
    GRAD_MUL,
    GRAD_DIV,
    GRAD_EXP,
    GRAD_LOG,
    GRAD_NEG,
};


//using namespace mvm;
#define UNPACK_BINARY_FUNC_ARGS \
    ValueType x = args[0]; \
    ValueType y = args[1]; \

#define UNPACK_UNARY_FUNC_ARGS \
    ValueType x = args[0]; \


template<typename ValueType>
class Box;

template<typename ValueType>
class BoxNode {
public:
    BoxNode(ValueType& t) : mT(t) {
    }
    BoxNode(ValueType&& t) : mT(std::move(t)) {
    }
    operator ValueType() {
        return mT;
    }
    template<class T>
    operator T() {
        return mT;
    }

    ValueType mT;
};

template<typename ValueType>
class Box;

template<typename T>
struct isBox {
    static constexpr bool value = false;
};

template<typename T>
struct isBox<Box<T>> {
    static constexpr bool value = true;
};
template<typename ValueType>
class Box {
public:
    Box() {
    }

    Box(ValueType& t) : mNode(new BoxNode<ValueType>(t)) {}

    Box(ValueType&& t) : mNode(new BoxNode<ValueType>(std::move(t))) {}

    operator ValueType() {
        return *mNode;
    }
    BoxNode<ValueType>* operator->() {
        return mNode.get();
    }

    const BoxNode<ValueType>* operator->() const {
        return mNode.get();
    }

    template<class T>
    operator T() const {
        return *mNode;
    }

    //ValueType operator()(const ValueType& t) {
         //
    //}

    std::map<int, Box> parents() const {
        return mParents;
    }

    template<int index>
    Box<ValueType>& setParents(Box<ValueType>& parent) {
        mParents[index] = parent;
        return *this;
    }

    Box<ValueType>& setParents(const std::vector<Box<ValueType>>& parents) {
        for (int i = 0; i < parents.size(); ++i) {
            mParents[i] = parents[i];
        }
        return *this;
    }

    Box<ValueType>& setInputs(const std::vector<Box<ValueType>>& inputs) {
        mInputs = inputs;
        return *this;
    }

    std::vector<Box<ValueType>> inputs() {
        return mInputs;
    }

    Box<ValueType>& setDesc(const std::string& desc) {
        mDesc = desc;
        return *this;
    }

    Box<ValueType>& setGradType(GradType gradType) {
        mGradType = gradType;
        return *this;
    }

    GradType gradType() {
        return mGradType;
    }

    std::string desc() const {
        return mDesc;
    }

    Box<ValueType>& setTraceID(int id) {
        mTraceID = id;
        return *this;
    }
    Box<ValueType>& incTraceID() {
        mTraceID++;
        return *this;
    }
    Box<ValueType>& decTraceID() {
        MVM_CHECK(mTraceID > 0, "Cannot deduce Box as which is not a box");
        mTraceID--;
        return *this;
    }
    int traceID() const {
        return mTraceID;
    }
    std::string str() {
        std::stringstream ss;
        ss << "(" << mDesc << ", traceID:" << mTraceID << ")";
        return ss.str();
    }
private:
    std::string mDesc;
    std::shared_ptr<BoxNode<ValueType>> mNode;
    std::map<int, Box<ValueType>> mParents;//int: parameter index, Box: parent
    std::vector<Box<ValueType>> mInputs;
    GradType mGradType;
    int mTraceID{0};
};

#define DEFINE_BLOCK_BINARY_OPERATOR_DATA_TYPE(OP, GRAD_TYPE, DATA_TYPE) \
    template<typename ValueType>                                                                    \
    Box<ValueType> operator OP(DATA_TYPE a, Box<ValueType> b) {                                           \
        Box<ValueType> r;                                                                           \
        if (b.traceID() == 0) {                                                                     \
            r = Box<ValueType>(a OP b->mT);                                                         \
        } else {                                                                                    \
            Box<ValueType> c = b;                                                                   \
            c.decTraceID();                                                                         \
            r = Box<ValueType>(a OP c);                                                             \
            r.setInputs({Box<ValueType>(a).setDesc(std::to_string(a)).setTraceID(b.traceID()), b})  \
             .setTraceID(b.traceID())                                                               \
             .template setParents<1>(b);                                                            \
        }                                                                                           \
        r.setGradType(GRAD_TYPE)                                                                    \
        .setDesc(std::to_string(a) + #OP + b.desc());                                               \
        return r;                                                                                   \
    }                                                                                               \
    template<typename ValueType>                                                                    \
    Box<ValueType> operator OP(Box<ValueType> a, DATA_TYPE b) {                                           \
        Box<ValueType> r;                                                                           \
        if (a.traceID() == 0) {                                                                     \
            r = Box<ValueType>(a->mT OP b);                                                         \
        } else {                                                                                    \
            Box<ValueType> c = a;                                                                   \
            c.decTraceID();                                                                         \
            r = Box<ValueType>(c OP b);                                                             \
            r.setInputs({a, Box<ValueType>(b).setTraceID(a.traceID())})                             \
             .template setParents<0>(a)                                                             \
             .setTraceID(a.traceID());                                                              \
        }                                                                                           \
        r.setGradType(GRAD_TYPE)                                                                    \
        .setDesc(a.desc() + #OP + std::to_string(b));                                               \
        return r;                                                                                   \
    }                                                                                               \

#define DEFINE_BLOCK_BINARY_OPERATOR(OP, GRAD_TYPE)                                                 \
    template<typename ValueType>                                                                    \
    Box<ValueType> operator OP(Box<ValueType> a, Box<ValueType> b) {                                \
        MVM_CHECK(a.traceID() >= 0 && b.traceID() >= 0, "traceID of a and b must be same, %d %d"    \
                , a.traceID(), b.traceID());                                                        \
        Box<ValueType> r;                                                                           \
        if (a.traceID() == 0 && b.traceID() == 0) {                                                 \
            r = Box<ValueType>(a->mT OP b->mT);                                                     \
        } else if (a.traceID() == 0) {                                                              \
            r = Box<ValueType>(a->mT OP b);                                                         \
        } else if (b.traceID() == 0) {                                                              \
            r = Box<ValueType>(a OP b->mT);                                                         \
        } else {                                                                                    \
            Box<ValueType> c = a;                                                                   \
            Box<ValueType> d = b;                                                                   \
            c.decTraceID();                                                                         \
            d.decTraceID();                                                                         \
            r = Box<ValueType>(c OP d);                                                             \
            r.setInputs({c, d});                                                                    \
            r.setTraceID(a.traceID());                                                              \
            r.setParents({a, b});                                                                   \
        }                                                                                           \
        r.setGradType(GRAD_TYPE)                                                                    \
        .setDesc(a.desc() + #OP + b.desc());                                                        \
        return r;                                                                                   \
    }                                                                                               \
    DEFINE_BLOCK_BINARY_OPERATOR_DATA_TYPE(OP, GRAD_TYPE, int) \
    DEFINE_BLOCK_BINARY_OPERATOR_DATA_TYPE(OP, GRAD_TYPE, float) \


DEFINE_BLOCK_BINARY_OPERATOR(+, GRAD_ADD)
DEFINE_BLOCK_BINARY_OPERATOR(-, GRAD_SUB)
DEFINE_BLOCK_BINARY_OPERATOR(*, GRAD_MUL)
DEFINE_BLOCK_BINARY_OPERATOR(/, GRAD_DIV)

#define DEFINE_BLOCK_UNARY_OPERATOR(OP, GRAD_TYPE)                                                  \
    template<typename ValueType>                                                                    \
    Box<ValueType> OP(Box<ValueType> a) {                                                           \
        MVM_CHECK(a.traceID() >= 0, "traceID of a must greater and equal 0, %d"                     \
                , a.traceID());                                                                     \
        Box<ValueType> r;                                                                           \
        if (a.traceID() == 0) {                                                                     \
            r = Box<ValueType>(OP(a->mT));                                                          \
        } else {                                                                                    \
            Box<ValueType> c = a;                                                                   \
            c.decTraceID();                                                                         \
            r = Box<ValueType>(OP(c));                                                              \
            r.setInputs({c})                                                                        \
             .setTraceID(a.traceID())                                                               \
             .template setParents<0>(a);                                                            \
        }                                                                                           \
        r.setGradType(GRAD_TYPE)                                                                    \
         .setDesc(#OP"(" + a.desc() + ")");                                                         \
        return r;                                                                                   \
    }                                                                                               \

DEFINE_BLOCK_UNARY_OPERATOR(log, GRAD_LOG)
DEFINE_BLOCK_UNARY_OPERATOR(exp, GRAD_EXP)
DEFINE_BLOCK_UNARY_OPERATOR(operator-, GRAD_NEG)

template<GradType gradType, typename ValueType>
class Grad;


#define GRAD_BINARY_FUNC(GRADTYPE, DX, DY)                                                          \
template<typename ValueType>                                                                        \
class Grad<GRADTYPE, ValueType> {                                                                   \
public:                                                                                             \
    ValueType operator()(ValueType g, ValueType ans, std::vector<ValueType> args, int index) {      \
        ValueType x = args[0];                                                                      \
        ValueType y = args[1];                                                                      \
        /*assert(index >=0 && index <= 1 && "Invalid index for binary grad func");*/                \
        if (index == 0) {                                                                           \
            return DX;                                                                              \
        } else {                                                                                    \
            return DY;                                                                              \
        }                                                                                           \
    }                                                                                               \
};                                                                                                  \

#define GRAD_UNARY_FUNC(GRADTYPE, DX)                                                               \
template<typename ValueType>                                                                        \
class Grad<GRADTYPE, ValueType> {                                                                   \
public:                                                                                             \
    ValueType operator()(ValueType g, ValueType ans, std::vector<ValueType> args, int index) {      \
        ValueType x = args[0];                                                                      \
        /*assert(index ==0 && "Invalid index for unary grad func");*/                               \
        return DX;                                                                                  \
    }                                                                                               \
};                                                                                                  \

template<typename ValueType>
ValueType basic_grad(GradType gradType, ValueType g, ValueType ans, std::vector<ValueType> args, int index) {
#define GRAD_FUNC_IMPLEMENT(GRADTYPE) \
        if (gradType == GRADTYPE) func = Grad<GRADTYPE, ValueType>();
    std::function<ValueType(ValueType g, ValueType ans, std::vector<ValueType> args, int index)> func;
    GRAD_FUNC_IMPLEMENT(GRAD_ADD)
    GRAD_FUNC_IMPLEMENT(GRAD_SUB)
    GRAD_FUNC_IMPLEMENT(GRAD_MUL)
    GRAD_FUNC_IMPLEMENT(GRAD_DIV)
    GRAD_FUNC_IMPLEMENT(GRAD_LOG)
    GRAD_FUNC_IMPLEMENT(GRAD_EXP)
    GRAD_FUNC_IMPLEMENT(GRAD_NEG)
    return func(g, ans, args, index);
}

GRAD_BINARY_FUNC(GRAD_ADD, g, g);
GRAD_BINARY_FUNC(GRAD_SUB, g, -g);
GRAD_BINARY_FUNC(GRAD_MUL, g * y, g * x);
GRAD_BINARY_FUNC(GRAD_DIV, g / y, -g * x / (y * y));
GRAD_UNARY_FUNC (GRAD_LOG, g / x);
GRAD_UNARY_FUNC (GRAD_EXP, g * ans);
GRAD_UNARY_FUNC (GRAD_NEG, -g);


template<typename ValueType>
std::string Box2Value(const Box<ValueType>& box) {
    std::stringstream ss;
    const BoxNode<ValueType>& node = *(box.operator->());
    const ValueType& t = (ValueType&)node;
    ss << t;
    return ss.str();
}


template<typename T>
struct UnBoxing {
    using type = T;
};

template<typename T>
struct UnBoxing<Box<T>> : UnBoxing<T> {
};


template<typename ValueType>
void iter(Box<ValueType>& b) {
    printf("iter:%s, parents:%lu\n", b.str().c_str(), b.parents().size());
    for (auto& bb : b.parents()) {
        printf("%s(%s) -> %s(%s) \n", b.desc().c_str(), Box2Value(b).c_str(),
                bb.second.desc().c_str(), Box2Value(bb.second).c_str());
        iter(bb.second);
    }
}

template<typename ValueType>
void toposort(Box<ValueType>& b, std::set<BoxNode<ValueType>*>& visited, std::stack<Box<ValueType>>& result) {
    visited.emplace(b.operator->());
    for (auto& bb : b.parents()) {
        if (!visited.count(bb.second.operator->())) {
            toposort(bb.second, visited, result);
        }
    }
    result.push(b);
}

template<typename ValueType>
std::vector<Box<ValueType>> toposort(Box<ValueType>& b) {
    std::vector<Box<ValueType>> result;
    std::set<BoxNode<ValueType>*> visited;
    std::stack<Box<ValueType>> rstack;
    toposort(b, visited, rstack);

    while(!rstack.empty()) {
        result.emplace_back(rstack.top());
        rstack.pop();
    }
    return result;

}

template<typename ValueType, typename std::enable_if<isBox<ValueType>::value>::type* = nullptr>
ValueType backward_pass(ValueType r) {
    //printf("forward----------------------\n");
    //iter(r);
    //printf("result:%f\n", (float)r);
    //printf("backward----------------------\n");
    //topo sort
    auto boxList = toposort(r);
    std::map<decltype(r.operator->()), ValueType> outputGrads;
    ValueType ans = 1;
    ans.setDesc("g");
    outputGrads[r.operator->()] = ans;//TODO:support array
    for (auto& b : boxList) {
        auto* node = b.operator->();
        auto& outputGrad = outputGrads[node];
        for (auto& parent : b.parents()) {
            std::vector<ValueType> inputs(b.inputs().size());
            for (int i = 0; i < inputs.size(); ++i) {
                inputs[i] = (b.inputs()[i]);
            }
            auto parentGrad = basic_grad(b.gradType(), outputGrad, b, inputs, parent.first);
            auto* parentBoxNode = parent.second.operator->();
            auto it = outputGrads.find(parentBoxNode);
            if (it != outputGrads.end()) {
                outputGrads[parentBoxNode] = it->second + parentGrad;
            } else {
                outputGrads[parentBoxNode] = parentGrad;
            }
        }
        ans = outputGrad;
        outputGrads.erase(node);
    }
    //MVM_CHECK(outputGrads.size() == 1, "outputGrads size should be 1 but not %d",
            //outputGrads.size());
    return ans;
    //return outputGrads.begin()->second;
}
