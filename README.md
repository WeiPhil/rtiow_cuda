# Ray Tracing in One Weekend in CUDA
 
A small ray tracer written in CUDA based on Peter Shirley's excellent book series "Ray Tracing in One Weekend".

## Compiling the project

To compile the project make sure you have installed on your system *SDL2*, *glm* and a recent CUDA Toolkit (They should be detected by cmake).

Then simply run the following commands at the root of the cloned repository:

```bash
mkdir build && cd build
cmake .. -GNinja
ninja
```