#pragma once

#ifndef CAMERA_H
#define CAMERA_H

#include "vector.h"

class Camera {
public:
    Camera() {}
    Camera(float aspect_ratio)
    {
        viewport_height = 2.f;
        viewport_width = aspect_ratio * viewport_height;
        focal_length = 1.f;

        origin = Point3f(0.f, 0.f, 0.f);
        horizontal = Vector3f(viewport_width, 0.f, 0.f);
        vertical = Vector3f(0.f, viewport_height, 0.f);
        lower_left_corner =
            origin - horizontal / 2.f - vertical / 2.f - Vector3f(0.f, 0.f, focal_length);
    }

    void move_origin(Vector3f translation)
    {
        origin += translation;
        lower_left_corner =
            origin - horizontal / 2.f - vertical / 2.f - Vector3f(0.f, 0.f, focal_length);
    }

    void update_aspect_ratio(float aspect_ratio)
    {
        viewport_width = aspect_ratio * viewport_height;

        horizontal = Vector3f(viewport_width, 0.f, 0.f);
        vertical = Vector3f(0.f, viewport_height, 0.f);
        lower_left_corner =
            origin - horizontal / 2.f - vertical / 2.f - Vector3f(0.f, 0.f, focal_length);
    }

public:
    Point3f origin;
    Vector3f horizontal;
    Vector3f vertical;
    Vector3f lower_left_corner;

private:
    float focal_length;
    float viewport_height;
    float viewport_width;
};

#endif