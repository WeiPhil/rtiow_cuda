#ifndef DIELECTRIC_H
#define DIELECTRIC_H

#include "common/macros.h"

CUDART_NAMESPACE_BEGIN

struct Dielectric {
    CUDART_FN float sample(float a, int b, float c) const;
};

CUDART_NAMESPACE_END

#endif
