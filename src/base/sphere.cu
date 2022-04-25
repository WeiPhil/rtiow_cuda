
#include <glm/gtx/norm.hpp>

#include "base/sphere.h"
#include "common/macros.h"

CUDART_NAMESPACE_BEGIN

CUDART_FN bool Sphere::hit(const Ray &ray, float t_min, float t_max, HitRecord &rec) const
{
    Vector3f oc = ray.origin - center;
    float a = length2(ray.dir);
    float half_b = dot(oc, ray.dir);
    float c = length2(oc) - radius * radius;
    float discriminant = half_b * half_b - a * c;

    if (discriminant < 0.0) {
        return false;
    }
    auto sqrtd = sqrt(discriminant);

    // Find the nearest root that lies in the acceptable range.
    float root = (-half_b - sqrtd) / a;
    if (root < t_min || t_max < root) {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || t_max < root)
            return false;
    }

    rec.t = root;
    rec.p = ray.at(rec.t);
    rec.normal = (rec.p - center) / radius;

    Vector3f outward_normal = (rec.p - center) / radius;
    rec.set_face_normal(ray, outward_normal);

    return true;
}

CUDART_NAMESPACE_END