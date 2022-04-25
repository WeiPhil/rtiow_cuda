#include <GL/glew.h>

#include <GL/gl.h>

#include <SDL2/SDL.h>

#include "common/macros.h"
#include "opengl/gl_display.h"

CUDART_NAMESPACE_BEGIN

#define BUFFER_OFFSET(i) ((char *)NULL + (i))

OpenGLDisplay::OpenGLDisplay(SDL_Window *window, int fb_width, int fb_height, Color3f *fb)
    : window(window),
      gl_context(SDL_GL_CreateContext(window)),
      fb_width(fb_width),
      fb_height(fb_height),
      fb(fb),
      opengl_error(false)
{
    SDL_GLContext gl_context;

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::puts("Could not initialize GLEW!");
        SDL_Quit();
        exit(1);
    }

    /// Init shader programs
    Compiler compiler;

    shaderManager.addShader("VERT_DEFAULT",
                            "../src/opengl/shaders/texture_shader.vert",
                            GL_VERTEX_SHADER,
                            compiler);
    shaderManager.addShader("FRAG_DEFAULT",
                            "../src/opengl/shaders/texture_shader.frag",
                            GL_FRAGMENT_SHADER,
                            compiler);
    programManager.addProgram("DEFAULT");

    glAttachShader(programManager("DEFAULT"), shaderManager("VERT_DEFAULT"));
    glAttachShader(programManager("DEFAULT"), shaderManager("FRAG_DEFAULT"));
    glLinkProgram(programManager("DEFAULT"));

    opengl_error = compiler.check() && opengl_error;
    opengl_error = compiler.check_program(programManager("DEFAULT")) && opengl_error;

    // Init texture
    textureManager.addTexture("OUT_IMAGE");
    glGenTextures(textureManager.size(), &textureManager.textures[0]);
    glBindTexture(GL_TEXTURE_2D, textureManager("OUT_IMAGE"));
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, fb_width, fb_height, 0, GL_RGB, GL_FLOAT, fb);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Bind to programm
    glUseProgram(programManager("DEFAULT"));
    glUniform1i(glGetUniformLocation(programManager("DEFAULT"), "image"), 0);

    // Init vertex array
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    if (!opengl_error)
        printf("Initialisation Done\n");
    else
        printf("An error occured during initialisation\n");
}

void OpenGLDisplay::resize(int new_fb_width, int new_fb_height, Color3f *new_fb)
{
    if (new_fb) {
        fb = new_fb;
        fb_width = new_fb_width;
        fb_height = new_fb_height;
        glGenTextures(textureManager.size(), &textureManager.textures[0]);
        glBindTexture(GL_TEXTURE_2D, textureManager("OUT_IMAGE"));
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, fb_width, fb_height, 0, GL_RGB, GL_FLOAT, fb);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    } else {
        printf("Failed to resize display!\n");
    }
}

OpenGLDisplay::~OpenGLDisplay()
{
    glDeleteProgram(programManager("DEFAULT"));
    glDeleteTextures(textureManager.size(), &textureManager.textures[0]);
    glDeleteVertexArrays(1, &vao);
}

void OpenGLDisplay::display()
{
    glViewport(0, 0, fb_width, fb_height);

    glClearColor(0.233f, 0.233f, 0.233f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);

    glUseProgram(programManager("DEFAULT"));

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, textureManager("OUT_IMAGE"));

    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, fb_width, fb_height, GL_RGB, GL_FLOAT, fb);

    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    // glDrawElementsInstancedBaseVertex(
    //     GL_TRIANGLES, eboManager.getElementCount(), GL_UNSIGNED_INT, 0, 1, 0);
}

CUDART_NAMESPACE_END