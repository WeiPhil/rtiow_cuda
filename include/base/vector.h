#pragma once

#include "common/macros.h"
#ifndef VEC3_H
#define VEC3_H

#include <glm/vec3.hpp>

CUDART_NAMESPACE_BEGIN

typedef glm::vec3 Vector3f;
typedef glm::vec3 Color3f;
typedef glm::vec3 Point3f;

CUDART_NAMESPACE_END

#endif