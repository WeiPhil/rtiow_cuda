#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "hittable.h"

#include <memory>
#include <vector>

class Scene : public Hittable {
public:
    __device__ Scene() {}
    __device__ Scene(Hittable **objects, int num_objects)
        : objects(objects), num_objects(num_objects)
    {
    }
    __device__ virtual bool hit(const Ray &ray,
                                float t_min,
                                float t_max,
                                HitRecord &rec) const override;

public:
    Hittable **objects;
    int num_objects;
};

__device__ bool Scene::hit(const Ray &ray, float t_min, float t_max, HitRecord &rec) const
{
    HitRecord temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;

    for (int i = 0; i < num_objects; i++) {
        if (objects[i]->hit(ray, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}

#endif