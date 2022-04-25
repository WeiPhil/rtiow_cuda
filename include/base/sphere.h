#ifndef SPHERE_H
#define SPHERE_H

#include "base/hittable.h"
#include "base/vector.h"
#include "common/macros.h"

CUDART_NAMESPACE_BEGIN

class Sphere : public Hittable {
public:
    CUDART_FN Sphere() {}
    CUDART_FN Sphere(Point3f center, float radius) : center(center), radius(radius) {}

    CUDART_FN virtual bool hit(const Ray &r,
                               float t_min,
                               float t_max,
                               HitRecord &rec) const override;

public:
    Point3f center;
    float radius;
};

CUDART_NAMESPACE_END

#endif