#ifndef SCENE_H
#define SCENE_H

#include "hittable.h"

#include <memory>
#include <vector>

#include "common/allocation.h"
#include "common/macros.h"

CUDART_NAMESPACE_BEGIN

class Scene {
public:
    CUDART_FN Scene() {}
    CUDART_FN Scene(Hittable **objects, int num_objects)
        : objects(objects), num_objects(num_objects)
    {
    }
    CUDART_FN bool hit(const Ray &ray, float t_min, float t_max, HitRecord &rec) const;

public:
    Hittable **objects;
    int num_objects;
};

CUDART_NAMESPACE_END

#endif