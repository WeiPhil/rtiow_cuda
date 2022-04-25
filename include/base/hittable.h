#ifndef HITTABLE_H
#define HITTABLE_H

#include <glm/glm.hpp>

#include "base/ray.h"
#include "common/macros.h"

CUDART_NAMESPACE_BEGIN

struct HitRecord {
    Point3f p;
    Vector3f normal;
    float t;

    bool front_face;

    CUDART_FN inline void set_face_normal(const Ray &ray, const Vector3f &outward_normal)
    {
        front_face = dot(ray.dir, outward_normal) < 0.f;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class Hittable {
public:
    CUDART_FN virtual bool hit(const Ray &r,
                               float t_min,
                               float t_max,
                               HitRecord &rec) const = 0;
};

CUDART_NAMESPACE_END

#endif