
#pragma once

#ifndef VARIANT_H
#define VARIANT_H

#include <cstddef>
#include <type_traits>

#include <common/macros.h>

CUDART_NAMESPACE_BEGIN
//-------------------------------------------------------------------------------------------------
// Get index of T in parameter pack
//

template <typename T, typename... Ts>
struct index_of_impl;

template <typename T, typename... Ts>
struct index_of_impl<T, T, Ts...> : std::integral_constant<size_t, 0> {
};

template <typename T, typename U, typename... Ts>
struct index_of_impl<T, U, Ts...>
    : std::integral_constant<size_t, 1 + index_of_impl<T, Ts...>::value> {
};

template <typename T, typename... Ts>
constexpr size_t index_of = index_of_impl<T, Ts...>::value;

//-------------------------------------------------------------------------------------------------
// Get type at index I
//

template <size_t I, typename... Ts>
struct type_at_impl {
};

template <typename T, typename... Ts>
struct type_at_impl<0, T, Ts...> {
    using type = T;
};

template <size_t I, typename T, typename... Ts>
struct type_at_impl<I, T, Ts...> : type_at_impl<I - 1, Ts...> {
};

template <size_t I, typename... Ts>
using type_at = typename type_at_impl<I, Ts...>::type;

template <size_t I>
using type_index = std::integral_constant<size_t, I>;

// struct Rectangle {
//     double b;
// };
// struct Triangle {
//     int a;
// };

struct Mat1 {
    float t1;

    float shade(float a)
    {
        return a;
    }
};
struct Mat2 {
    int t2;

    float shade(float a)
    {
        return a;
    }
};
struct Mat3 {
    double t3;

    float shade(float a)
    {
        return a;
    }
};

// Base case variant
template <typename... Ts>
union VariantStorage {
};

// Variadic variants
template <typename T, typename... Ts>
union VariantStorage<T, Ts...> {
    T element;
    /* Recursion stops when the empty
    VariantStorage<> template is
    being instantiated. */
    VariantStorage<Ts...> nextElements;

    // access

    CUDART_FN T &get(type_index<1>)
    {
        return element;
    }

    CUDART_FN T const &get(type_index<1>) const
    {
        return element;
    }

    template <unsigned I>
    CUDART_FN type_at<I - 1, Ts...> &get(type_index<I>)
    {
        return nextElements.get(type_index<I - 1>{});
    }

    template <unsigned I>
    CUDART_FN type_at<I - 1, Ts...> const &get(type_index<I>) const
    {
        return nextElements.get(type_index<I - 1>{});
    }
};

template <typename... Ts>
struct Variant {
    VariantStorage<Ts...> storage;
    int type_id;

    template <typename T>
    T *as()
    {
        // Reinterpret as type T if type id matches.
        if (type_id == index_of<T, Ts...>)
            return reinterpret_cast<T *>(&storage);
        else
            return nullptr;
    }
};

// struct ShadeVisitor {
//     auto operator()(Mat1 t1)
//     {
//         treatAsT1(t1);
//     }
//     auto operator()(Mat2 t2)
//     {
//         treatAsT2(t2);
//     }
//     auto operator()(Mat3 t3)
//     {
//         treatAsT3(t3);
//     }
// };

// template <typename... Ts>
// class GenericMaterial<Ts...> : public Variant<Ts...> {
// public:
//     // Construct variant from concrete material.
//     template <typename Mat>
//     GenericMaterial(Mat mat) : Variant<Ts...>(mat)
//     {
//     }
//     // Shade interface function.
//     float shade(float)
//     {
//         // ShadeVisitor visits all types in
//         // parameter pack Ts... and calls
//         // T::shade(Intersection)
//         applyVisitor(ShadeVisitor(float), *this);
//     }
// };

// template <typename T>
// void applyVisitor(Visitor(), T)
// {
// }

// Variant<Type1, Type2> var = makeVariant(···);
// applyVisitor(Visitor(), var);

// template <typename... Ts>
// class GenericShape<Ts...> : public Variant<Ts...> {
// public:
//     // Construct variant from concrete material.
//     template <typename Shape>
//     GenericShape(Shape mat) : Variant<Ts...>(mat)
//     {
//     }

//     // Shade interface function.
//     int shade(float test)
//     {
//         // ShadeVisitor visits all types in
//         // parameter pack Ts... and calls
//         // T::shade(Intersection)
//         applyVisitor(ShadeVisitor(float), *this);
//     }
// };

// struct Visitor {
//     auto operator()(Type1 t1)
//     {
//         treatAsT1(t1);
//     }
//     auto operator()(Type2 t2)
//     {
//         treatAsT2(t2);
//     }
//     auto operator()(Type3 t3)
//     {
//         treatAsT3(t3);
//     }
// };

CUDART_NAMESPACE_END

#endif