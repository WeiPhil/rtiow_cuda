#include "base/bsdf/dielectric.h"

CUDART_FN float cudart::Dielectric::sample(float a, int b, float c) const
{
    return a * a * b * b * c * c;
}