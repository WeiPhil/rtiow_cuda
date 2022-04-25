
#include <memory>
#include <vector>

#include "base/hittable.h"
#include "base/scene.h"

#include "common/allocation.h"
#include "common/macros.h"

CUDART_NAMESPACE_BEGIN

CUDART_FN bool Scene::hit(const Ray &ray, float t_min, float t_max, HitRecord &rec) const
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

CUDART_NAMESPACE_END
