#version 330 core

out vec4 fragColor;

uniform sampler2D tex;

in vec2 TexCoord;

void main() { fragColor = texture(tex, TexCoord); }