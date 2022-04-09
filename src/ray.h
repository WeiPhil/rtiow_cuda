#pragma once

#ifndef RAY_H
#define RAY_H

#include "vector.h"

class Ray {
public:
    __host__ __device__ Ray() {}
    __host__ __device__ Ray(const Point3f &origin, const Vector3f &direction)
        : origin(origin), dir(direction)
    {
    }

     __host__ __device__ Point3f at(float t) const
    {
        return origin + t * dir;
    }

public:
    Point3f origin;
    Vector3f dir;
};

#endif