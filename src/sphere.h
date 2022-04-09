#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"
#include "vector.h"

class Sphere : public Hittable {
public:
    __device__ Sphere() {}
    __device__ Sphere(Point3f center, float radius) : center(center), radius(radius) {}

    __device__ virtual bool hit(const Ray &r,
                                float t_min,
                                float t_max,
                                HitRecord &rec) const override;

public:
    Point3f center;
    float radius;
};

#endif