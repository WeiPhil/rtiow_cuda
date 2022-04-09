#version 330 core

layout(location = 0) in vec3 aPosition;
layout(location = 4) in vec2 aTextCoord;

out vec2 TexCoord;

void main()
{
    gl_Position = vec4(aPosition, 1.0);
    TexCoord = vec2(aTextCoord.x, 1.0 - aTextCoord.y);
}