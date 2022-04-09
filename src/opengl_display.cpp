#include "opengl_display.h"

#include <GL/gl.h>
#include <GL/glew.h>
#include <SDL2/SDL.h>

#define BUFFER_OFFSET(i) ((char *)NULL + (i))

OpenGLDisplay::OpenGLDisplay(SDL_Window *window, int fb_width, int fb_height, Color3f *fb)
    : window(window),
      gl_context(SDL_GL_CreateContext(window)),
      fb_width(fb_width),
      fb_height(fb_height),
      fb(fb)
{
    SDL_GLContext gl_context;

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::puts("Could not initialize GLEW!");
        SDL_Quit();
        exit(1);
    }

    /// Init vertices and triangles
    vaoManager.addVertex(
        glf::vertex_v3fv2f(glm::vec3(1.0f, 1.0f, 0.0f), glm::vec2(0.0f, 0.0f)));  // top right
    vaoManager.addVertex(
        glf::vertex_v3fv2f(glm::vec3(1.0f, -1.0f, 0.0f), glm::vec2(0.0f, 1.0f)));  // top left
    vaoManager.addVertex(glf::vertex_v3fv2f(glm::vec3(-1.0f, -1.0f, 0.0f),
                                            glm::vec2(1.0f, 1.0f)));  // bottom left
    vaoManager.addVertex(glf::vertex_v3fv2f(glm::vec3(-1.0f, 1.0f, 0.0f),
                                            glm::vec2(1.0f, 0.0f)));  // bottom right

    eboManager.addTriangle(0, 1, 2);
    eboManager.addTriangle(2, 3, 0);

    /// Init shader programs
    Compiler compiler;

    shaderManager.addShader(
        "VERT_DEFAULT", "../data/texture_shader.vert", GL_VERTEX_SHADER, compiler);
    shaderManager.addShader(
        "FRAG_DEFAULT", "../data/texture_shader.frag", GL_FRAGMENT_SHADER, compiler);
    programManager.addProgram("DEFAULT");

    glAttachShader(programManager("DEFAULT"), shaderManager("VERT_DEFAULT"));
    glAttachShader(programManager("DEFAULT"), shaderManager("FRAG_DEFAULT"));
    glLinkProgram(programManager("DEFAULT"));

    opengl_error = compiler.check() && opengl_error;
    opengl_error = compiler.check_program(programManager("DEFAULT")) && opengl_error;

    /// Init vertex and element buffers
    bufferManager.addBuffer("ELEMENT");
    bufferManager.addBuffer("VERTEX");

    glGenBuffers(bufferManager.size(), &bufferManager.buffers[0]);

    // Create vertex array object
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bufferManager("ELEMENT"));
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 eboManager.getElementSize(),
                 &eboManager.elementData[0],
                 GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    // Create vertex array object
    glBindBuffer(GL_ARRAY_BUFFER, bufferManager("VERTEX"));
    glBufferData(GL_ARRAY_BUFFER,
                 vaoManager.getVertexDataSize(),
                 &vaoManager.vertexData[0],
                 GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

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
    glUniform1i(glGetUniformLocation(programManager("DEFAULT"), "texture"), 0);

    // Init vertex array
    glGenVertexArrays(1, &vaoManager.id);
    glBindVertexArray(vaoManager.id);
    glBindBuffer(GL_ARRAY_BUFFER, bufferManager("VERTEX"));

    glVertexAttribPointer(semantic::attr::POSITION,
                          3,
                          GL_FLOAT,
                          GL_FALSE,
                          vaoManager.getVertexSize(),
                          BUFFER_OFFSET(0));
    glEnableVertexAttribArray(semantic::attr::POSITION);

    glVertexAttribPointer(semantic::attr::TEXCOORD,
                          2,
                          GL_FLOAT,
                          GL_FALSE,
                          vaoManager.getVertexSize(),
                          BUFFER_OFFSET(sizeof(glm::vec3)));

    glEnableVertexAttribArray(semantic::attr::TEXCOORD);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Bind element buffer array to array ob
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bufferManager("ELEMENT"));
    glBindVertexArray(0);

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
    glDeleteFramebuffers(framebufferManager.size(), &framebufferManager.framebuffers[0]);
    glDeleteProgram(programManager("DEFAULT"));

    glDeleteBuffers(bufferManager.size(), &bufferManager.buffers[0]);
    glDeleteTextures(textureManager.size(), &textureManager.textures[0]);
    glDeleteVertexArrays(1, &vaoManager.id);
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

    glBindVertexArray(vaoManager.id);
    glDrawElementsInstancedBaseVertex(
        GL_TRIANGLES, eboManager.getElementCount(), GL_UNSIGNED_INT, 0, 1, 0);
}