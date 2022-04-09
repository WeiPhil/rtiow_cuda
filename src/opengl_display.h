#pragma once

#ifndef DISPLAY_FRAMEBUFFER_H
#define DISPLAY_FRAMEBUFFER_H

#include <SDL2/SDL.h>

#include "opengl/buffermanager.h"
#include "opengl/compiler.h"
#include "opengl/ebomanager.h"
#include "opengl/framebuffermanager.h"
#include "opengl/programmanager.h"
#include "opengl/semantics.h"
#include "opengl/shadermanager.h"
#include "opengl/texturemanager.h"
#include "opengl/vaomanager.h"
#include "opengl/vertex.h"

#include "vector.h"

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

    /// Buffer management
    BufferManager bufferManager;
    FramebufferManager framebufferManager;
    TextureManager textureManager;
    ShaderManager shaderManager;
    ProgramManager programManager;
    VAOManager<glf::vertex_v3fv2f> vaoManager;
    EBOManager eboManager;
    bool opengl_error = false;
};

#endif