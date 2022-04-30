#ifndef DIFFUSE_H
#define DIFFUSE_H

#include "common/macros.h"

CUDART_NAMESPACE_BEGIN

struct Diffuse {
    CUDART_FN float sample(float a, int b, float c) const;
};

CUDART_NAMESPACE_END

#endif
