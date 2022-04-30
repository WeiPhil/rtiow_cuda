#ifndef BSDF_H
#define BSDF_H

#include "common/macros.h"
#include <glm/glm.hpp>

#include "common/variant.h"

CUDART_NAMESPACE_BEGIN

//-------------------------------------------------------------------------------------------------
// Generic Bsdf
//

template <typename... Ts>
class GenericBsdf;

// Forward declarations of Bsdfs
struct Dielectric;
struct Diffuse;

// Bsdf alias
using Bsdf = GenericBsdf<Diffuse, Dielectric>;

template <typename T, typename... Ts>
class GenericBsdf<T, Ts...> : public Variant<T, Ts...> {
public:
    using BaseType = Variant<T, Ts...>;

public:
    GenericBsdf() = default;

    template <typename Mat>
    /* implicit */ GenericBsdf(Mat const &Bsdf);

    CUDART_FN float sample(float a, int b, float c) const;

private:
    // Variant visitors
    struct SampleVisitor;
};

template <typename T, typename... Ts>
struct GenericBsdf<T, Ts...>::SampleVisitor {
    using ReturnType = float;

    CUDART_FN
    SampleVisitor(float a, int b, float c) : _a(a), _b(b), _c(c) {}

    template <typename Mat>
    CUDART_FN ReturnType operator()(Mat const &ref) const
    {
        return ref.sample(_a, _b, _c);
    }

    float _a;
    int _b;
    float _c;
};

template <typename T, typename... Ts>
template <typename Mat>
inline GenericBsdf<T, Ts...>::GenericBsdf(Mat const &Bsdf)
    : GenericBsdf<T, Ts...>::BaseType(Bsdf)
{
}

template <typename T, typename... Ts>
CUDART_FN inline float GenericBsdf<T, Ts...>::sample(float a, int b, float c) const
{
    return apply_visitor(SampleVisitor(a, b, c), *this);
}

CUDART_NAMESPACE_END

#endif