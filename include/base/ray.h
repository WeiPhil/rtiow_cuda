#pragma once

#ifndef RAY_H
#define RAY_H

#include "base/vector.h"
#include "common/macros.h"

CUDART_NAMESPACE_BEGIN

class Ray {
public:
    CUDART_FN Ray() {}
    CUDART_FN Ray(const Point3f &origin, const Vector3f &direction)
        : origin(origin), dir(direction)
    {
    }

    CUDART_FN Point3f at(float t) const
    {
        return origin + t * dir;
    }

public:
    Point3f origin;
    Vector3f dir;
};

CUDART_NAMESPACE_END

#endif