#pragma once

#include <atomic>

namespace mvm {

template<typename T>
class ObjectPtr {
public:
    ObjectPtr() {}
    virtual ~ObjectPtr() {
        reset();
    }

    ObjectPtr(T* node) : mPtr(node) {
        if (mPtr != NULL) {
            mPtr->incRef();
        }
    }

    void swap(ObjectPtr<T>& other) {
        std::swap(mPtr, other.mPtr);
    }

    void reset() {
        if (mPtr != NULL) {
            mPtr->decRef();
            mPtr = NULL;
        }
    }

    //Any object refer to mPtr cannot delete mPtr
    T* release() {
        T* r = mPtr;
        mPtr = NULL;
        return r;
    }

    ObjectPtr(const ObjectPtr<T>& other) : ObjectPtr(other.mPtr) {
    }

    template<typename U>
    ObjectPtr(const ObjectPtr<U>& other) : ObjectPtr(other.mPtr) {
        static_assert(std::is_base_of<T, U>::value,
                "can only assign of child class ObjectPtr to parent");
    }

    ObjectPtr(ObjectPtr<T>&& other) : mPtr(other.mPtr) {
        other.mPtr = NULL;
    }

    template<typename U>
    ObjectPtr(ObjectPtr<U>&& other) : mPtr(other.mPtr) {
        static_assert(std::is_base_of<T, U>::value,
                "can only assign of child class ObjectPtr to parent");
        other.mPtr = NULL;
    }

    ObjectPtr<T>& operator=(ObjectPtr<T>&& other) {
        ObjectPtr<T>(std::move(other)).swap(*this);
        return *this;
    }
    ObjectPtr<T>& operator=(const ObjectPtr<T>& other) {
        ObjectPtr<T>(other).swap(*this);
        return *this;
    }

    T& operator*() const {
        return *mPtr;
    }

    T* operator->() const {
        return get();
    }

    T* get() const {
        return mPtr;
    };

    bool defined() const {
        return get() != NULL;
    }

    int useCount() const {
        return mPtr != NULL ? mPtr->useCount() : 0;
    }

    inline bool unique() const {
        return mPtr != NULL && mPtr->useCount() == 1;
    };

    //inline operator bool() const {
    //    return defined();
    //}

    inline bool operator==(const ObjectPtr<T>& other) const {
        return mPtr == other.mPtr;
    }

    inline bool operator!=(const ObjectPtr<T>& other) const {
        return mPtr != other.mPtr;
    }

    inline bool operator<(const ObjectPtr<T>& other) const {
        return mPtr < other.mPtr;
    }

    inline bool operator<=(const ObjectPtr<T>& other) const {
        return mPtr <= other.mPtr;
    }

private:
    T* mPtr{NULL};
};

class ObjectNode;
class Object : public ObjectPtr<ObjectNode> {
public:
    Object() = default;

    explicit Object(ObjectNode* node) : ObjectPtr<ObjectNode>(node) {
    }
    explicit Object(const ObjectNode* node)
        : ObjectPtr<ObjectNode>(const_cast<ObjectNode*>(node)) {
    }

    template<typename NODE>
    inline const NODE* as() const {
        return dynamic_cast<const NODE*>(get());
    }
    using ContainerType = ObjectNode;

};

class ObjectNode {
public:
    virtual ~ObjectNode() = default;

    int useCount() const {
        return mCount.load(std::memory_order_relaxed);
    }

protected:
    void decRef() {
        if (mCount.fetch_sub(1, std::memory_order_release) == 1) {
            std::atomic_thread_fence(std::memory_order_acquire);
            delete this;
        }
    }

    void incRef() {
        mCount.fetch_add(1, std::memory_order_relaxed);
    }

private:
    std::atomic<int> mCount{0};
    template<typename>
    friend class ObjectPtr;
};

#define MVM_OBJECT_METHODS(TypeName, ParentType, NodeType)       \
    explicit TypeName(NodeType* node) : ParentType(node) {       \
    }                                                            \
    explicit TypeName(const NodeType* node) : ParentType(node) { \
    }                                                            \
    NodeType* operator->() const {                               \
        return get();                                            \
    }                                                            \
    NodeType& operator*() const {                                \
        return *get();                                           \
    }                                                            \
    NodeType* get() const {                                      \
        return static_cast<NodeType*>(ParentType::get());        \
    }

} // namespace mvm
