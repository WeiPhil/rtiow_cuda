#version 330 core

uniform sampler2D image;

out vec4 color;

void main(void){ 
	ivec2 uv = ivec2(gl_FragCoord.x, gl_FragCoord.y);
	color = texelFetch(image, uv, 0);
}