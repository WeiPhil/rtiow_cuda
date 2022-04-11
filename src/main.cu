#include <fstream>
#include <iostream>
#include <memory>
#include <assert.h>

#include <SDL2/SDL.h>
#include <glm/gtx/norm.hpp>

#include "camera.h"
#include "hittable.h"
#include "opengl/gl_display.h"
#include "ray.h"

#include "sphere.h"

#include "scene.h"
#include "vector.h"

#include "allocator.h"

#define DEBUG

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCuda(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result,
                char const *const func,
                const char *const file,
                int const line)
{
    if (result) {
        std::cerr << "CUDA Runtime error: " << cudaGetErrorString(result) << " at " << file
                  << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__device__ Color3f sample(const Ray &ray, Scene *scene)
{
    // If we hit the world shade
    HitRecord rec;
    if (scene->hit(ray, 0.f, FLT_MAX, rec)) {
        Vector3f normal = normalize(ray.at(rec.t) - Vector3f(0.f, 0.f, -1.f));
        return 0.5f * Color3f(normal.x + 1.f, normal.y + 1.f, normal.z + 1.f);
    } else {
        // Otherwise display background
        Vector3f unit_direction = normalize(ray.dir);
        float t = 0.5f * (unit_direction.y + 1.f);
        return (1.f - t) * Color3f(1.f, 1.f, 1.f) + t * Color3f(0.5f, 0.7f, 1.f);
    }
}

__global__ void render(
    Color3f *fb, int im_width, int im_height, const Camera &camera, Scene **scene)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i >= im_width || j >= im_height)
        return;

    float u = float(i) / (im_width - 1);
    float v = float(j) / (im_height - 1);
    Ray ray(camera.origin,
            camera.lower_left_corner + u * camera.horizontal + v * camera.vertical -
                camera.origin);

    int index = j * im_width + i;

    fb[index] = sample(ray, *scene);
}

__global__ void create_scene(Hittable **d_objects, Scene **d_scene)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_objects[0] = new Sphere(Vector3f(0.f, 0.f, -1.f), 0.5f);
        d_objects[1] = new Sphere(Vector3f(0.f, -100.5f, -1.f), 100.f);
        *d_scene = new Scene(d_objects, 2);
    }
}

__global__ void free_scene(Scene **d_scene)
{
    for (size_t i = 0; i < (*d_scene)->num_objects; i++) {
        delete (*d_scene)->objects[i];
    }
    delete (*d_scene);
}

void save_image(float *fb, int im_width, int im_height)
{
    // output image
    std::ofstream image_file;
    image_file.open("out.ppm");
    image_file << "P3\n" << im_width << ' ' << im_height << "\n255\n";

    for (int j = im_height - 1; j >= 0; j--) {
        for (int i = 0; i < im_width; i++) {
            int index = j * im_width * 3 + i * 3;

            int ir = int(255.999 * fb[index]);
            int ig = int(255.999 * fb[index + 1]);
            int ib = int(255.999 * fb[index + 2]);

            image_file << ir << ' ' << ig << ' ' << ib << '\n';
        }
    }
    image_file.close();

    // Convert ppm to png
    system("convert out.ppm out.png");
}

int main()
{
    int im_width = 1920;
    int im_height = 1080;
    float aspect_ratio = float(im_width) / im_height;
    const int thread_num_x = 8;
    const int thread_num_y = 8;

    auto camera = cudart::make_shared<Camera>(aspect_ratio);

    size_t fb_size = im_width * im_height * sizeof(Color3f);
    // framebuffer
    Color3f *fb;
    checkCuda(cudaMallocManaged(&fb, fb_size));

    // Create scene on device
    Scene **d_scene;
    Hittable **d_objects;
    checkCuda(cudaMalloc(&d_scene, sizeof(Scene *)));
    checkCuda(cudaMalloc(&d_objects, 2 * sizeof(Hittable *)));
    create_scene<<<1, 1>>>(d_objects, d_scene);

    dim3 threads(thread_num_x, thread_num_y);
    dim3 blocks((im_width + thread_num_x - 1) / thread_num_x,
                (im_height + thread_num_y - 1) / thread_num_y);

    printf("[Cuda] Using blocks = (%u,%u,%u) , threads = (%u,%u,%u)\n",
           blocks.x,
           blocks.y,
           blocks.z,
           threads.x,
           threads.y,
           threads.z);

    // Init SDL2
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        fprintf(stderr, "Could not init SDL: %s\n", SDL_GetError());
        exit(1);
    }
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

    SDL_Window *sdl_window = SDL_CreateWindow("Cuda RTioW",
                                              SDL_WINDOWPOS_UNDEFINED,
                                              SDL_WINDOWPOS_UNDEFINED,
                                              im_width,
                                              im_height,
                                              SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
    if (nullptr == sdl_window) {
        fprintf(stderr, "Could not create window! : %s\n", SDL_GetError());
        SDL_Quit();
        exit(1);
    }

    std::unique_ptr<OpenGLDisplay> display;
    display = std::make_unique<OpenGLDisplay>(sdl_window, im_width, im_height, fb);

    SDL_Event event;
    bool running = true;

    float sensibility = 0.1f;
    auto frame_time_ms = 0;
    const uint8_t *state = SDL_GetKeyboardState(nullptr);
    while (running) {
        auto start_time_ms = SDL_GetTicks();

        // Continous-response keys
        if (state[SDL_SCANCODE_W]) {
            camera->move_origin(Vector3f(0.f, 0.f, -1.f) * sensibility);
        }
        if (state[SDL_SCANCODE_S]) {
            camera->move_origin(Vector3f(0.f, 0.f, 1.f) * sensibility);
        }
        if (state[SDL_SCANCODE_A]) {
            camera->move_origin(Vector3f(-1.f, 0.f, 0.f) * sensibility);
        }
        if (state[SDL_SCANCODE_D]) {
            camera->move_origin(Vector3f(1.f, 0.f, 0.f) * sensibility);
        }
        if (state[SDL_SCANCODE_E]) {
            camera->move_origin(Vector3f(0.f, 1.f, 0.f) * sensibility);
        }
        if (state[SDL_SCANCODE_Q]) {
            camera->move_origin(Vector3f(0.f, -1.f, 0.f) * sensibility);
        }

        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = false;
            }
            if (event.type == SDL_WINDOWEVENT &&
                event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED) {
                // Update size and update
                im_width = event.window.data1;
                im_height = event.window.data2;
                aspect_ratio = float(im_width) / im_height;
                camera->update_aspect_ratio(aspect_ratio);
                fb_size = im_width * im_height * sizeof(Color3f);
                blocks = dim3((im_width + thread_num_x - 1) / thread_num_x,
                              (im_height + thread_num_y - 1) / thread_num_y);

                checkCuda(cudaFree(fb));
                checkCuda(cudaMallocManaged(&fb, fb_size));
                display->resize(im_width, im_height, fb);

                printf("im_size : (%u,%u)\n", im_width, im_height);
            }
        }

        render<<<blocks, threads>>>(fb, im_width, im_height, *camera, d_scene);
        checkCuda(cudaGetLastError());

        // Synchronise device/host and display
        checkCuda(cudaDeviceSynchronize());
        display->display();

        SDL_GL_SwapWindow(sdl_window);

        frame_time_ms = SDL_GetTicks() - start_time_ms;
        printf("fps : %f \r", 1000.0 / frame_time_ms);
    }

    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaFree(fb));
    free_scene<<<1, 1>>>(d_scene);
    checkCuda(cudaFree(d_scene));
    checkCuda(cudaFree(d_objects));

    SDL_Quit();
}