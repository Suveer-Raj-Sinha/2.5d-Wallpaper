#version 330 core

in vec2 in_position;
in vec2 in_uv;

out vec2 v_uv;
out float v_depth;

uniform vec3 u_camera;             // (x, y, zoom)
uniform float u_layer_depth;       // [0.0, 1.0] (0=far, 1=near)
uniform float u_parallax_strength; // global strength
uniform float u_parallax_factor;   // layer specific multiplier
uniform vec2 u_scale;              // (scale_x, scale_y)
uniform vec2 u_offset;             // (offset_x, offset_y)
uniform float u_explode;           // [0.0, 1.0] exploded 3D view factor
uniform float u_aspect;            // aspect ratio compensation

void main() {
    v_uv = in_uv;
    v_depth = u_layer_depth;

    // Base parallax displacement
    // Depth: 0.0 (far) has small displacement, 1.0 (near) has large displacement
    // Reference focal plane at 0.5
    float focal_depth = 0.5;
    float depth_offset = (u_layer_depth - focal_depth);
    vec2 parallax_shift = u_camera.xy * depth_offset * u_parallax_factor * u_parallax_strength * 4.0;

    // Apply scale and offset
    vec2 pos = in_position * u_scale + u_offset;

    // Exploded 3D view adds perspective tilt and vertical/horizontal separation
    if (u_explode > 0.001) {
        float z_sep = (u_layer_depth - 0.5) * u_explode * 0.8;
        // Subtle perspective shear and separation
        pos.x += z_sep * 0.45;
        pos.y -= z_sep * 0.35;
        pos *= (1.0 + z_sep * 0.2);
    }

    // Camera shift and zoom
    pos = (pos + parallax_shift) * u_camera.z;

    gl_Position = vec4(pos, 0.0, 1.0);
}
