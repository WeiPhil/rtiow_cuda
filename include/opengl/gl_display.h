#pragma once

#ifndef DISPLAY_FRAMEBUFFER_H
#define DISPLAY_FRAMEBUFFER_H

#include "common/macros.h"
#include "opengl/compiler.h"
#include "opengl/programmanager.h"
#include "opengl/shadermanager.h"
#include "opengl/texturemanager.h"

#include "base/vector.h"

#include <SDL2/SDL.h>

CUDART_NAMESPACE_BEGIN

class OpenGLDisplay {
public:
    OpenGLDisplay(SDL_Window *window, int fb_width, int fb_height, Color3f *fb);
    ~OpenGLDisplay();

    void display();

    void resize(int fb_width, int fb_height, Color3f *fb);

private:
    SDL_Window *window;
    SDL_GLContext gl_context;
    int fb_width, fb_height;
    Color3f *fb;

    GLuint vao;
    TextureManager textureManager;
    ShaderManager shaderManager;
    ProgramManager programManager;
    bool opengl_error;
};

CUDART_NAMESPACE_END

#endif