#version 330 core

in vec2 in_position;
in vec2 in_uv;

out vec2 v_uv;

uniform vec3 u_camera;

void main() {
    v_uv = in_uv;
    vec2 pos = in_position * u_camera.z;
    gl_Position = vec4(pos, 0.0, 1.0);
}
