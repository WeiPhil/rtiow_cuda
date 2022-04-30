#include "base/bsdf/diffuse.h"

CUDART_FN float cudart::Diffuse::sample(float a, int b, float c) const
{
    return a * b * c;
}
