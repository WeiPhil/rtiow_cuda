// The MIT License (MIT)

// Copyright (c) 2015 S. Zellmann

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#ifndef VARIANT_H
#define VARIANT_H

#include <cstddef>
#include <type_traits>

#include <common/macros.h>
#include "base/vector.h"

CUDART_NAMESPACE_BEGIN

//-------------------------------------------------------------------------------------------------
// Get index of T in parameter pack
//

template <typename... Ts>
struct index_of {
    enum { value = 0 };
};

template <typename U, typename... Ts>
struct index_of<U, U, Ts...> {
    enum { value = 1 };
};

template <typename U, typename T, typename... Ts>
struct index_of<U, T, Ts...> {
    enum { v = index_of<U, Ts...>::value };
    enum { value = v == 0 ? 0 : 1 + v };
};

//-------------------------------------------------------------------------------------------------
// Get type at index I
//

template <size_t I, typename... Ts>
struct type_at_impl {
};

template <typename T, typename... Ts>
struct type_at_impl<1, T, Ts...> {
    using type = T;
};

template <size_t I, typename T, typename... Ts>
struct type_at_impl<I, T, Ts...> : type_at_impl<I - 1, Ts...> {
};

template <size_t I, typename... Ts>
using type_at = typename type_at_impl<I, Ts...>::type;

template <size_t I>
using type_index = std::integral_constant<size_t, I>;

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

    template <size_t I>
    CUDART_FN type_at<I - 1, Ts...> &get(type_index<I>)
    {
        return nextElements.get(type_index<I - 1>{});
    }

    template <size_t I>
    CUDART_FN type_at<I - 1, Ts...> const &get(type_index<I>) const
    {
        return nextElements.get(type_index<I - 1>{});
    }
};

template <typename... Ts>
struct Variant {
    Variant() = default;

    template <typename T>
    CUDART_FN Variant(T const &value) : type_index_(index_of<T, Ts...>::value)
    {
        storage_.get(type_index<index_of<T, Ts...>::value>()) = value;
    }

    template <typename T>
    CUDART_FN Variant &operator=(T const &value)
    {
        type_index_ = index_of<T, Ts...>::value;
        storage_.get(type_index<index_of<T, Ts...>::value>()) = value;
        return *this;
    }

    template <typename T>
    CUDART_FN T *as()
    {
        return type_index_ == index_of<T, Ts...>::value
                   ? &storage_.get(type_index<index_of<T, Ts...>::value>())
                   : nullptr;
    }

    template <typename T>
    CUDART_FN T const *as() const
    {
        return type_index_ == index_of<T, Ts...>::value
                   ? &storage_.get(type_index<index_of<T, Ts...>::value>())
                   : nullptr;
    }

private:
    size_t type_index_;
    VariantStorage<Ts...> storage_;
};

//-------------------------------------------------------------------------------------------------
// Template Visitor pattern
//

template <size_t I, typename... Ts>
struct apply_visitor_impl;

template <size_t I, typename T, typename... Ts>
struct apply_visitor_impl<I, T, Ts...> {
    template <typename Visitor, typename Variant>
    CUDART_FN typename Visitor::ReturnType operator()(Visitor const &visitor,
                                                      Variant const &var) const
    {
        auto ptr = var.template as<T>();
        if (ptr)
            return visitor(*ptr);
        else
            return apply_visitor_impl<I - 1, Ts...>()(visitor, var);
    }
};

template <>
struct apply_visitor_impl<0> {
    template <typename Visitor, typename Variant>
    CUDART_FN typename Visitor::ReturnType operator()(Visitor const &, Variant const &)
    {
        // error if here?
        return typename Visitor::ReturnType();
    }
};

template <typename Visitor, typename... Ts>
CUDART_FN typename Visitor::ReturnType apply_visitor(Visitor const &visitor,
                                                     Variant<Ts...> const &var)
{
    return apply_visitor_impl<sizeof...(Ts), Ts...>()(visitor, var);
}

CUDART_NAMESPACE_END

#endif