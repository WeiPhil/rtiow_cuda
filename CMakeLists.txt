cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(cudart LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OpenGL_GL_PREFERENCE GLVND)


if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

if(MSVC)
    # Force to always compile with W4
    if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
        string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
    endif()
    elseif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
    # Update if necessary
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic")
endif()

# ====== OpenGL ======
find_package(OpenGL REQUIRED)
include_directories( ${OPENGL_INCLUDE_DIRS} )
# ====== OpenGL ======

# ====== GLM ======
add_definitions(-DGLM_ENABLE_EXPERIMENTAL -DSDL_MAIN_HANDLED)

find_package(glm REQUIRED PATHS ${CMAKE_SOURCE_DIR}/ext/glm/cmake)
# ====== GLM ======

# ====== SDL2 ======  
find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})
# ====== SDL2 ====== 

  

# Adding cpp files per directory
set(SRC_DIR src)
aux_source_directory(${SRC_DIR} SRC_FILES)
aux_source_directory(${SRC_DIR}/opengl SRC_FILES_OPENGL)

# Include path for the project 
include_directories(${SRC_DIR})

add_executable(cudart ${GL3W_SRC} ${SRC_FILES} ${SRC_FILES_OPENGL})
 
set_property(TARGET cudart 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(cudart GLEW ${OPENGL_LIBRARIES}  ${SDL2_LIBRARIES})
 
