#version 330 core

in vec2 v_uv;
in float v_depth;

out vec4 frag_color;

uniform sampler2D u_texture;
uniform float u_opacity;
uniform int u_debug_tint;

void main() {
    vec4 tex = texture(u_texture, v_uv);

    // Alpha cutoff
    if (tex.a < 0.005) {
        discard;
    }

    vec3 rgb = tex.rgb;

    // Optional debug tinting to highlight separate layers in 3D exploded view
    if (u_debug_tint == 1) {
        // Subtle depth tint: cooler far away, warmer close up
        vec3 tint = mix(vec3(0.5, 0.7, 1.0), vec3(1.0, 0.8, 0.4), v_depth);
        rgb = mix(rgb, rgb * tint, 0.25);
    }

    frag_color = vec4(rgb, tex.a * u_opacity);
}
