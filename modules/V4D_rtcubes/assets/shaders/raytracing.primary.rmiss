#version 460 core

#extension GL_GOOGLE_cpp_style_line_directive : enable

#define _DEBUG
#define SHADER_RMISS
#define SHADER_SUBPASS_0
#define SHADER_OFFSET_0

#line 1 "/home/olivier/projects/chill/src/v4d/modules/V4D_rtcubes/base.glsl"
#ifndef _RTCUBES_SHADER_BASE_INCLUDED_
#define _RTCUBES_SHADER_BASE_INCLUDED_

#extension GL_EXT_ray_tracing : require
#extension GL_EXT_shader_atomic_float : require
#extension GL_ARB_shader_clock : enable

#line 1 "/home/olivier/projects/chill/src/v4d/game/graphics/glsl/base.glsl"
#ifndef _SHADER_BASE_INCLUDED_
#define _SHADER_BASE_INCLUDED_

#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_nonuniform_qualifier : require

#line 1 "/home/olivier/projects/chill/src/v4d/game/graphics/glsl/../cpp_glsl.hh"
#line 2 "/home/olivier/projects/chill/src/v4d/game/graphics/glsl/../cpp_glsl.hh"
#line 1 "/home/olivier/projects/chill/src/v4d/core/v4d.h"
#ifdef __cplusplus
#line 3 "/home/olivier/projects/chill/src/v4d/core/v4d.h"

// Vulkan4D Core Header

#define V4D_VERSION_MAJOR 0
#define V4D_VERSION_MINOR 0
#define V4D_VERSION_PATCH 0

// V4D Core class (Compiled into v4d.dll)
# include "Core.h"

#endif // __cplusplus
#line 1 "/home/olivier/projects/chill/src/v4d/core/utilities/graphics/shaders/cpp_glsl_head.hh"
#ifdef __cplusplus

	// https://github.com/KhronosGroup/GLSL/blob/master/extensions/ext/GL_EXT_shader_explicit_arithmetic_types.txt

	#define aligned_int8_t alignas(1) int8_t
	#define aligned_uint8_t alignas(1) uint8_t
	#define aligned_int16_t alignas(2) int16_t
	#define aligned_uint16_t alignas(2) uint16_t
	#define aligned_int32_t alignas(4) int32_t
	#define aligned_uint32_t alignas(4) uint32_t
	#define aligned_int64_t alignas(8) int64_t
	#define aligned_uint64_t alignas(8) uint64_t

	#define aligned_float32_t alignas(4) glm::float32_t
	#define aligned_float64_t alignas(8) glm::float64_t

	#define aligned_i8vec2 alignas(2) glm::i8vec2
	#define aligned_u8vec2 alignas(2) glm::u8vec2
	#define aligned_i8vec3 alignas(4) glm::i8vec3
	#define aligned_u8vec3 alignas(4) glm::u8vec3
	#define aligned_i8vec4 alignas(4) glm::i8vec4
	#define aligned_u8vec4 alignas(4) glm::u8vec4

	#define aligned_i16vec2 alignas(4) glm::i16vec2
	#define aligned_u16vec2 alignas(4) glm::u16vec2
	#define aligned_i16vec3 alignas(8) glm::i16vec3
	#define aligned_u16vec3 alignas(8) glm::u16vec3
	#define aligned_i16vec4 alignas(8) glm::i16vec4
	#define aligned_u16vec4 alignas(8) glm::u16vec4

	#define aligned_f32vec2 alignas(8) glm::f32vec2
	#define aligned_i32vec2 alignas(8) glm::i32vec2
	#define aligned_u32vec2 alignas(8) glm::u32vec2
	#define aligned_f32vec3 alignas(16) glm::f32vec3
	#define aligned_i32vec3 alignas(16) glm::i32vec3
	#define aligned_u32vec3 alignas(16) glm::u32vec3
	#define aligned_f32vec4 alignas(16) glm::f32vec4
	#define aligned_i32vec4 alignas(16) glm::i32vec4
	#define aligned_u32vec4 alignas(16) glm::u32vec4

	#define aligned_f64vec2 alignas(16) glm::f64vec2
	#define aligned_i64vec2 alignas(16) glm::i64vec2
	#define aligned_u64vec2 alignas(16) glm::u64vec2
	#define aligned_f64vec3 alignas(32) glm::f64vec3
	#define aligned_i64vec3 alignas(32) glm::i64vec3
	#define aligned_u64vec3 alignas(32) glm::u64vec3
	#define aligned_f64vec4 alignas(32) glm::f64vec4
	#define aligned_i64vec4 alignas(32) glm::i64vec4
	#define aligned_u64vec4 alignas(32) glm::u64vec4

	#define aligned_f32mat3x4 alignas(16) glm::f32mat3x4
	#define aligned_f64mat3x4 alignas(32) glm::f64mat3x4
	
	#define aligned_f32mat4 alignas(16) glm::f32mat4
	#define aligned_f64mat4 alignas(32) glm::f64mat4
	
	#define aligned_VkDeviceAddress alignas(8) VkDeviceAddress

	#define STATIC_ASSERT_ALIGNED16_SIZE(T, X) static_assert(sizeof(T) == X && sizeof(T) % 16 == 0);
	#define STATIC_ASSERT_SIZE(T, X) static_assert(sizeof(T) == X);
	#define PUSH_CONSTANT_STRUCT struct
	#define BUFFER_REFERENCE_FORWARD_DECLARE(TypeName)
	#define BUFFER_REFERENCE_STRUCT(align) struct
	#define BUFFER_REFERENCE_STRUCT_READONLY(align) struct
	#define BUFFER_REFERENCE_STRUCT_WRITEONLY(align) struct
	#define BUFFER_REFERENCE_ADDR(type) aligned_VkDeviceAddress
	
#else // GLSL

	#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
	#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
	#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
	#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
	#extension GL_EXT_shader_explicit_arithmetic_types_float32 : enable
	#extension GL_EXT_shader_explicit_arithmetic_types_float64 : enable
	
	#define aligned_int8_t int8_t
	#define aligned_uint8_t uint8_t
	#define aligned_int16_t int16_t
	#define aligned_uint16_t uint16_t
	#define aligned_int32_t int32_t
	#define aligned_uint32_t uint32_t
	#define aligned_int64_t int64_t
	#define aligned_uint64_t uint64_t

	#define aligned_float32_t float32_t
	#define aligned_float64_t float64_t

	#define aligned_i8vec2 i8vec2
	#define aligned_u8vec2 u8vec2
	#define aligned_i8vec3 i8vec3
	#define aligned_u8vec3 u8vec3
	#define aligned_i8vec4 i8vec4
	#define aligned_u8vec4 u8vec4

	#define aligned_i16vec2 i16vec2
	#define aligned_u16vec2 u16vec2
	#define aligned_i16vec3 i16vec3
	#define aligned_u16vec3 u16vec3
	#define aligned_i16vec4 i16vec4
	#define aligned_u16vec4 u16vec4

	#define aligned_f32vec2 f32vec2
	#define aligned_i32vec2 i32vec2
	#define aligned_u32vec2 u32vec2
	#define aligned_f32vec3 f32vec3
	#define aligned_i32vec3 i32vec3
	#define aligned_u32vec3 u32vec3
	#define aligned_f32vec4 f32vec4
	#define aligned_i32vec4 i32vec4
	#define aligned_u32vec4 u32vec4

	#define aligned_f64vec2 f64vec2
	#define aligned_i64vec2 i64vec2
	#define aligned_u64vec2 u64vec2
	#define aligned_f64vec3 f64vec3
	#define aligned_i64vec3 i64vec3
	#define aligned_u64vec3 u64vec3
	#define aligned_f64vec4 f64vec4
	#define aligned_i64vec4 i64vec4
	#define aligned_u64vec4 u64vec4

	#define aligned_f32mat3x4 f32mat3x4
	#define aligned_f64mat3x4 f64mat3x4
	
	#define aligned_f32mat4 f32mat4
	#define aligned_f64mat4 f64mat4
	
	#define aligned_VkDeviceAddress uint64_t
	
	#define STATIC_ASSERT_ALIGNED16_SIZE(T,X)
	#define STATIC_ASSERT_SIZE(T,X)
	#define PUSH_CONSTANT_STRUCT layout(push_constant) uniform
	#define BUFFER_REFERENCE_FORWARD_DECLARE(TypeName) layout(buffer_reference) buffer TypeName;
	#define BUFFER_REFERENCE_STRUCT(align) layout(buffer_reference, std430, buffer_reference_align = align) buffer
	#define BUFFER_REFERENCE_STRUCT_READONLY(align) layout(buffer_reference, std430, buffer_reference_align = align) buffer readonly
	#define BUFFER_REFERENCE_STRUCT_WRITEONLY(align) layout(buffer_reference, std430, buffer_reference_align = align) buffer writeonly
	#define BUFFER_REFERENCE_ADDR(type) type
	
#endif
#line 15 "/home/olivier/projects/chill/src/v4d/core/v4d.h"
#line 3 "/home/olivier/projects/chill/src/v4d/game/graphics/glsl/../cpp_glsl.hh"
#ifdef __cplusplus
	namespace game::graphics {
#endif

#define SET0_BINDING_CAMERAS 0
#define SET0_BINDING_IMG_COMPOSITE 1
#define SET0_BINDING_SAMPLER_COMPOSITE 2
#define SET0_BINDING_IMG_HISTORY 3
#define SET0_BINDING_SAMPLER_HISTORY 4
#define SET0_BINDING_IMG_MOTION 5
#define SET0_BINDING_SAMPLER_MOTION 6
#define SET0_BINDING_IMG_DEPTH 7
#define SET0_BINDING_SAMPLER_DEPTH 8
#define SET0_BINDING_IMG_RESOLVED 9
#define SET0_BINDING_SAMPLER_RESOLVED 10
#define SET0_BINDING_IMG_POST 11
#define SET0_BINDING_SAMPLER_POST 12
#define SET0_BINDING_IMG_DEBUG 13
#define SET0_BINDING_TEXTURES 14

#define SET1_BINDING_INOUT_IMG_RESOLVED 0
#define SET1_BINDING_IN_IMG_OVERLAY 1
#define SET1_BINDING_IMG_THUMBNAIL 2
#define SET1_BINDING_LUMINANCE_BUFFER 3

// up to 32 render options
#define RENDER_OPTION_TXAA (1u<< 0)
#define RENDER_OPTION_DLSS (1u<< 1)
#define RENDER_OPTION_TONE_MAPPING (1u<< 2)

// Debug view modes
#define RENDER_DEBUG_VIEWMODE_NONE 0

// Configuration
#define TXAA_SAMPLES 16
#define MAX_CAMERAS 64
#define MAX_TEXTURE_BINDINGS 65535

#define NB_RENDERABLE_TYPES 8 // this will NOT change, it is a limitation of VK_KHR_ray_tracing
// Renderable types
#define RENDERABLE_TYPE_SOLID 0 // Solid mostly-opaque object, may be somewhat dynamic, typically vehicles, hulls, small to medium size game objects
#define RENDERABLE_TYPE_SELF 1 // Typically refers to the current player's avatar when in first person view, invisible for primary rays but visible for reflections and shadow rays
#define RENDERABLE_TYPE_STATIC 2 // Geometry that usually doesn't move and can extend to very large scales, typically Terrain
#define RENDERABLE_TYPE_DYNAMIC 3 // Highly dynamic, small objects, typically mobs and temporary-animated objects, they do not occlude indirect lighting
#define RENDERABLE_TYPE_VOLUME 4 // Geometry that is transparent and that can include other geometries within it, typically Water and Fog, may be somewhat dynamic
// #define RENDERABLE_TYPE_CLUTTER 5 // Geometry that is usually present in large repeating quantities, typically Plants and Rocks, they do not occlude indirect lighting
// #define RENDERABLE_TYPE_LIGHT_RADIUS 6 // Special geometry that denotes a light's radius, to be point-traced for finding relevant light sources that may be shining on surfaces for direct lighting
// #define RENDERABLE_TYPE_OVERLAY 7 // Geometry that is ONLY visible from primary rays and that do not cast shadows, typically transparent bounding boxes and debug stuff

#define RENDERABLE_MASK_SOLID (1u<< RENDERABLE_TYPE_SOLID)
#define RENDERABLE_MASK_SELF (1u<< RENDERABLE_TYPE_SELF)
#define RENDERABLE_MASK_STATIC (1u<< RENDERABLE_TYPE_STATIC)
#define RENDERABLE_MASK_DYNAMIC (1u<< RENDERABLE_TYPE_DYNAMIC)
#define RENDERABLE_MASK_VOLUME (1u<< RENDERABLE_TYPE_VOLUME)
// #define RENDERABLE_MASK_CLUTTER (1u<< RENDERABLE_TYPE_CLUTTER)
// #define RENDERABLE_MASK_LIGHT_RADIUS (1u<< RENDERABLE_TYPE_LIGHT_RADIUS)
// #define RENDERABLE_MASK_OVERLAY (1u<< RENDERABLE_TYPE_OVERLAY)
#define RENDERABLE_ALL 0xff
#define RENDERABLE_PRIMARY (~RENDERABLE_MASK_SELF)
#define RENDERABLE_ALL_EXCEPT_VOLUME (RENDERABLE_ALL & ~RENDERABLE_MASK_VOLUME)
#define RENDERABLE_PRIMARY_EXCEPT_VOLUME (RENDERABLE_PRIMARY & ~RENDERABLE_MASK_VOLUME)

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Structs and Buffer References -- Must use aligned_* explicit arithmetic types (or VkDeviceAddress as an uint64_t, or BUFFER_REFERENCE_ADDR(StructType))

BUFFER_REFERENCE_STRUCT_WRITEONLY(16) MVPBufferCurrent {aligned_f32mat4 mvp;};
BUFFER_REFERENCE_STRUCT_READONLY(16) MVPBufferHistory {aligned_f32mat4 mvp;};
BUFFER_REFERENCE_STRUCT_WRITEONLY(8) RealtimeBufferCurrent {aligned_uint64_t mvpFrameIndex;};
BUFFER_REFERENCE_STRUCT_READONLY(8) RealtimeBufferHistory {aligned_uint64_t mvpFrameIndex;};

struct CameraData {
	aligned_f32mat4 viewMatrix;
	aligned_f32mat4 historyViewMatrix;
	aligned_f32mat4 projectionMatrix;
	aligned_f32mat4 projectionMatrixWithTXAA;
	aligned_f32mat4 reprojectionMatrix;
	aligned_f32vec4 luminance;
	aligned_uint32_t width;
	aligned_uint32_t height;
	aligned_float32_t fov;
	aligned_float32_t zNear;
	aligned_float32_t zFar;
	aligned_float32_t brightness;
	aligned_float32_t contrast;
	aligned_float32_t gamma;
	aligned_float32_t renderScale;
	aligned_uint32_t options;
	aligned_uint32_t _unused;
	aligned_uint32_t debugViewMode;
	aligned_uint64_t frameIndex;
	aligned_float64_t deltaTime;
	BUFFER_REFERENCE_ADDR(MVPBufferCurrent) mvpBuffer;
	BUFFER_REFERENCE_ADDR(MVPBufferHistory) mvpBufferHistory;
	BUFFER_REFERENCE_ADDR(RealtimeBufferCurrent) realtimeBuffer;
	BUFFER_REFERENCE_ADDR(RealtimeBufferHistory) realtimeBufferHistory;
};
STATIC_ASSERT_ALIGNED16_SIZE(CameraData, 432)

struct FSRPushConstant {
	aligned_u32vec4 Const0;
	aligned_u32vec4 Const1;
	aligned_u32vec4 Const2;
	aligned_u32vec4 Const3;
	aligned_u32vec4 Sample;
};
STATIC_ASSERT_SIZE(FSRPushConstant, 80)

BUFFER_REFERENCE_STRUCT_READONLY(16) AabbData {
	aligned_float32_t aabb[6];
	aligned_uint64_t extra; // Arbitrary data defined per-shader
};
STATIC_ASSERT_ALIGNED16_SIZE(AabbData, 32)

BUFFER_REFERENCE_STRUCT_READONLY(16) GeometryData {
	aligned_uint8_t sbtHandle[32];
	BUFFER_REFERENCE_ADDR(AabbData) aabbs;
	aligned_VkDeviceAddress vertices;
	aligned_VkDeviceAddress indices32;
	aligned_VkDeviceAddress indices16;
};
STATIC_ASSERT_ALIGNED16_SIZE(GeometryData, 64)

BUFFER_REFERENCE_STRUCT_READONLY(16) RenderableInstanceData {
	BUFFER_REFERENCE_ADDR(GeometryData) geometries;
	aligned_uint64_t _unused;
};
STATIC_ASSERT_ALIGNED16_SIZE(RenderableInstanceData, 16)

BUFFER_REFERENCE_STRUCT(16) AimBuffer {
	aligned_f32vec3 localPosition;
	#ifdef __cplusplus
		uint32_t geometryIndex : 12;
		uint32_t aimID : 20 = 0;
	#else
		uint32_t aimGeometryID;
	#endif
	aligned_f32vec3 worldSpaceHitNormal;
	aligned_uint32_t aabbIndex;
	aligned_f32vec3 worldSpacePosition; // MUST COMPENSATE FOR ORIGIN RESET
	aligned_float32_t hitDistance;
	aligned_f32vec4 color;
	aligned_f32vec3 viewSpaceHitNormal;
	aligned_uint32_t tlasInstanceIndex;
};
STATIC_ASSERT_ALIGNED16_SIZE(AimBuffer, 80)

#ifdef __cplusplus
	}
#endif
#line 8 "/home/olivier/projects/chill/src/v4d/game/graphics/glsl/base.glsl"

// Set 0
layout(set = 0, binding = SET0_BINDING_CAMERAS) uniform CameraUniformBuffer {CameraData cameras[MAX_CAMERAS];};
layout(set = 0, binding = SET0_BINDING_IMG_COMPOSITE, rgba32f) uniform image2D img_composite;
layout(set = 0, binding = SET0_BINDING_SAMPLER_COMPOSITE) uniform sampler2D sampler_composite;
layout(set = 0, binding = SET0_BINDING_IMG_HISTORY, rgba32f) uniform image2D img_history;
layout(set = 0, binding = SET0_BINDING_SAMPLER_HISTORY) uniform sampler2D sampler_history;
layout(set = 0, binding = SET0_BINDING_IMG_MOTION, rgba32f) uniform image2D img_motion;
layout(set = 0, binding = SET0_BINDING_SAMPLER_MOTION) uniform sampler2D sampler_motion;
layout(set = 0, binding = SET0_BINDING_IMG_DEPTH, r32f) uniform image2D img_depth;
layout(set = 0, binding = SET0_BINDING_SAMPLER_DEPTH) uniform sampler2D sampler_depth;
layout(set = 0, binding = SET0_BINDING_IMG_RESOLVED, rgba32f) uniform image2D img_resolved;
layout(set = 0, binding = SET0_BINDING_SAMPLER_RESOLVED) uniform sampler2D sampler_resolved;
layout(set = 0, binding = SET0_BINDING_IMG_POST, rgba8) uniform image2D img_post;
layout(set = 0, binding = SET0_BINDING_SAMPLER_POST) uniform sampler2D sampler_post;
layout(set = 0, binding = SET0_BINDING_IMG_DEBUG, rgba32f) uniform image2D img_debug;
layout(set = 0, binding = SET0_BINDING_TEXTURES) uniform sampler2D textures[];

CameraData camera = cameras[0];

#ifdef SHADER_VERT
	#define IMPORT_DEFAULT_FULLSCREEN_VERTEX_SHADER \
		void main() {\
			gl_Position = vec4(vec2((gl_VertexIndex << 1) & 2, 1-(gl_VertexIndex & 2)) * 2.0f -1.0f, 0.0f, 1.0f);\
		}
#endif

#ifdef SHADER_FRAG
	vec2 GetFragUV() {
		return gl_FragCoord.st / vec2(camera.width, camera.height);
	}
#endif

float GetDepth(vec2 uv) {
	return texture(sampler_depth, uv).r;
}

vec3 GetMotionVector(in vec2 uv, in float depth) {
	vec4 ndc = vec4(uv * 2 - 1, depth, 1);
	vec4 ndcHistory = camera.reprojectionMatrix * ndc;
	return ndcHistory.xyz / ndcHistory.w - ndc.xyz;
}

bool ReprojectHistoryUV(inout vec2 uv) {
	uv += texture(sampler_motion, uv).rg * 0.5;
	return uv.x > 0 && uv.x < 1 && uv.y > 0 && uv.y < 1;
}

float GetFragDepthFromWorldSpacePosition(vec3 worldSpacePos) {
	vec4 clipSpace = mat4(camera.projectionMatrixWithTXAA) * mat4(camera.viewMatrix) * vec4(worldSpacePos, 1);
	return clipSpace.z / clipSpace.w;
}

vec3 VarianceClamp5(in vec3 color, in sampler2D tex, in vec2 uv) {
	vec3 nearColor0 = texture(tex, uv).rgb;
	vec3 nearColor1 = textureLodOffset(tex, uv, 0.0, ivec2( 1,  0)).rgb;
	vec3 nearColor2 = textureLodOffset(tex, uv, 0.0, ivec2( 0,  1)).rgb;
	vec3 nearColor3 = textureLodOffset(tex, uv, 0.0, ivec2(-1,  0)).rgb;
	vec3 nearColor4 = textureLodOffset(tex, uv, 0.0, ivec2( 0, -1)).rgb;
	vec3 m1 = nearColor0
			+ nearColor1
			+ nearColor2
			+ nearColor3
			+ nearColor4
	; m1 /= 5;
	vec3 m2 = nearColor0*nearColor0
			+ nearColor1*nearColor1
			+ nearColor2*nearColor2
			+ nearColor3*nearColor3
			+ nearColor4*nearColor4
	; m2 /= 5;
	vec3 sigma = sqrt(m2 - m1*m1);
	const float sigmaNoVarianceThreshold = 0.0001;
	if (abs(sigma.r) < sigmaNoVarianceThreshold || abs(sigma.g) < sigmaNoVarianceThreshold || abs(sigma.b) < sigmaNoVarianceThreshold) {
		return nearColor0;
	}
	vec3 boxMin = m1 - sigma;
	vec3 boxMax = m1 + sigma;
	return clamp(color, boxMin, boxMax);
}

vec3 VarianceClamp9(in vec3 color, in sampler2D tex, in vec2 uv) {
	vec3 nearColor0 = texture(tex, uv).rgb;
	vec3 nearColor1 = textureLodOffset(tex, uv, 0.0, ivec2( 1,  0)).rgb;
	vec3 nearColor2 = textureLodOffset(tex, uv, 0.0, ivec2( 0,  1)).rgb;
	vec3 nearColor3 = textureLodOffset(tex, uv, 0.0, ivec2(-1,  0)).rgb;
	vec3 nearColor4 = textureLodOffset(tex, uv, 0.0, ivec2( 0, -1)).rgb;
	vec3 nearColor5 = textureLodOffset(tex, uv, 0.0, ivec2( 1,  1)).rgb;
	vec3 nearColor6 = textureLodOffset(tex, uv, 0.0, ivec2(-1,  1)).rgb;
	vec3 nearColor7 = textureLodOffset(tex, uv, 0.0, ivec2(-1, -1)).rgb;
	vec3 nearColor8 = textureLodOffset(tex, uv, 0.0, ivec2( 1, -1)).rgb;
	vec3 m1 = nearColor0
			+ nearColor1
			+ nearColor2
			+ nearColor3
			+ nearColor4
			+ nearColor5
			+ nearColor6
			+ nearColor7
			+ nearColor8
	; m1 /= 9;
	vec3 m2 = nearColor0*nearColor0
			+ nearColor1*nearColor1
			+ nearColor2*nearColor2
			+ nearColor3*nearColor3
			+ nearColor4*nearColor4
			+ nearColor5*nearColor5
			+ nearColor6*nearColor6
			+ nearColor7*nearColor7
			+ nearColor8*nearColor8
	; m2 /= 9;
	vec3 sigma = sqrt(m2 - m1*m1);
	const float sigmaNoVarianceThreshold = 0.0001;
	if (abs(sigma.r) < sigmaNoVarianceThreshold || abs(sigma.g) < sigmaNoVarianceThreshold || abs(sigma.b) < sigmaNoVarianceThreshold) {
		return nearColor0;
	}
	vec3 boxMin = m1 - sigma;
	vec3 boxMax = m1 + sigma;
	return clamp(color, boxMin, boxMax);
}

float Fresnel(const vec3 position, const vec3 normal, const float indexOfRefraction) {
	vec3 incident = normalize(position);
	float cosi = clamp(dot(incident, normal), -1, 1);
	float etai;
	float etat;
	if (cosi > 0) {
		etat = 1;
		etai = indexOfRefraction;
	} else {
		etai = 1;
		etat = indexOfRefraction;
	}
	// Compute sini using Snell's law
	float sint = etai / etat * sqrt(max(0.0, 1.0 - cosi * cosi));
	if (sint >= 1) {
		// Total internal reflection
		return 1.0;
	} else {
		float cost = sqrt(max(0.0, 1.0 - sint * sint));
		cosi = abs(cosi);
		float Rs = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
		float Rp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));
		return (Rs * Rs + Rp * Rp) / 2;
	}
}

bool Refract(inout vec3 rayDirection, in vec3 surfaceNormal, in float iOR) {
	const float vDotN = dot(rayDirection, surfaceNormal);
	const float niOverNt = vDotN > 0 ? iOR : 1.0 / iOR;
	vec3 dir = rayDirection;
	rayDirection = refract(rayDirection, -sign(vDotN) * surfaceNormal, niOverNt);
	if (dot(rayDirection,rayDirection) > 0) {
		rayDirection = normalize(rayDirection);
		return true;
	} else {
		rayDirection = normalize(reflect(dir, -sign(vDotN) * surfaceNormal));
	}
	return false;
}

vec3 ApplyToneMapping(in vec3 in_color) {
	vec3 color = in_color;
	
	// HDR ToneMapping (Reinhard)
	if ((camera.options & RENDER_OPTION_TONE_MAPPING) != 0) {
		float lumRgbTotal = camera.luminance.r + camera.luminance.g + camera.luminance.b;
		float exposure = lumRgbTotal > 0 ? camera.luminance.a / lumRgbTotal : 1;
		color.rgb = vec3(1.0) - exp(-color.rgb * clamp(exposure, 0.001, 10.0));
	}
	
	// Contrast / Brightness
	if (camera.contrast != 1.0 || camera.brightness != 1.0) {
		color.rgb = mix(vec3(0.5), color.rgb, camera.contrast) * camera.brightness;
	}
	
	// Gamma correction
	color.rgb = pow(color.rgb, vec3(1.0 / camera.gamma));
	
	return clamp(color, vec3(0), vec3(1));
}

vec3 Heatmap(float t) {
	if (t <= 0) return vec3(0);
	if (t >= 1) return vec3(1);
	const vec3 c[10] = {
		vec3(0.0f / 255.0f,   2.0f / 255.0f,  91.0f / 255.0f),
		vec3(0.0f / 255.0f, 108.0f / 255.0f, 251.0f / 255.0f),
		vec3(0.0f / 255.0f, 221.0f / 255.0f, 221.0f / 255.0f),
		vec3(51.0f / 255.0f, 221.0f / 255.0f,   0.0f / 255.0f),
		vec3(255.0f / 255.0f, 252.0f / 255.0f,   0.0f / 255.0f),
		vec3(255.0f / 255.0f, 180.0f / 255.0f,   0.0f / 255.0f),
		vec3(255.0f / 255.0f, 104.0f / 255.0f,   0.0f / 255.0f),
		vec3(226.0f / 255.0f,  22.0f / 255.0f,   0.0f / 255.0f),
		vec3(191.0f / 255.0f,   0.0f / 255.0f,  83.0f / 255.0f),
		vec3(145.0f / 255.0f,   0.0f / 255.0f,  65.0f / 255.0f)
	};

	const float s = t * 10.0f;

	const int cur = int(s) <= 9 ? int(s) : 9;
	const int prv = cur >= 1 ? cur - 1 : 0;
	const int nxt = cur < 9 ? cur + 1 : 9;

	const float blur = 0.8f;

	const float wc = smoothstep(float(cur) - blur, float(cur) + blur, s) * (1.0f - smoothstep(float(cur + 1) - blur, float(cur + 1) + blur, s));
	const float wp = 1.0f - smoothstep(float(cur) - blur, float(cur) + blur, s);
	const float wn = smoothstep(float(cur + 1) - blur, float(cur + 1) + blur, s);

	const vec3 r = wc * c[cur] + wp * c[prv] + wn * c[nxt];
	return vec3(clamp(r.x, 0.0f, 1.0f), clamp(r.y, 0.0f, 1.0f), clamp(r.z, 0.0f, 1.0f));
}


// Random
#extension GL_EXT_control_flow_attributes : require
uint InitRandomSeed(uint val0, uint val1) {
	uint v0 = val0, v1 = val1, s0 = 0;
	[[unroll]]
	for (uint n = 0; n < 16; n++) {
		s0 += 0x9e3779b9;
		v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
		v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
	}
	return v0;
}
uint RandomInt(inout uint seed) {
	return (seed = 1664525 * seed + 1013904223);
}
float RandomFloat(inout uint seed) {
	return (float(RandomInt(seed) & 0x00FFFFFF) / float(0x01000000));
}
vec2 RandomInUnitSquare(inout uint seed) {
	return 2 * vec2(RandomFloat(seed), RandomFloat(seed)) - 1;
}
vec2 RandomInUnitDisk(inout uint seed) {
	for (;;) {
		const vec2 p = 2 * vec2(RandomFloat(seed), RandomFloat(seed)) - 1;
		if (dot(p, p) < 1) {
			return p;
		}
	}
}
vec3 RandomInUnitSphere(inout uint seed) {
	for (;;) {
		const vec3 p = 2 * vec3(RandomFloat(seed), RandomFloat(seed), RandomFloat(seed)) - 1;
		if (dot(p, p) < 1) {
			return p;
		}
	}
}



#endif // _SHADER_BASE_INCLUDED_
#line 9 "/home/olivier/projects/chill/src/v4d/modules/V4D_rtcubes/base.glsl"
#line 1 "/home/olivier/projects/chill/src/v4d/modules/V4D_rtcubes/cpp_glsl.hh"
#ifdef __cplusplus
#line 3 "/home/olivier/projects/chill/src/v4d/modules/V4D_rtcubes/cpp_glsl.hh"
#endif
#line 1 "/home/olivier/projects/chill/src/v4d/core/v4d.h"
#ifdef __cplusplus
#line 3 "/home/olivier/projects/chill/src/v4d/core/v4d.h"

// Vulkan4D Core Header

#define V4D_VERSION_MAJOR 0
#define V4D_VERSION_MINOR 0
#define V4D_VERSION_PATCH 0

// V4D Core class (Compiled into v4d.dll)
# include "Core.h"

#endif // __cplusplus
#line 1 "/home/olivier/projects/chill/src/v4d/core/utilities/graphics/shaders/cpp_glsl_head.hh"
#ifdef __cplusplus

	// https://github.com/KhronosGroup/GLSL/blob/master/extensions/ext/GL_EXT_shader_explicit_arithmetic_types.txt

	#define aligned_int8_t alignas(1) int8_t
	#define aligned_uint8_t alignas(1) uint8_t
	#define aligned_int16_t alignas(2) int16_t
	#define aligned_uint16_t alignas(2) uint16_t
	#define aligned_int32_t alignas(4) int32_t
	#define aligned_uint32_t alignas(4) uint32_t
	#define aligned_int64_t alignas(8) int64_t
	#define aligned_uint64_t alignas(8) uint64_t

	#define aligned_float32_t alignas(4) glm::float32_t
	#define aligned_float64_t alignas(8) glm::float64_t

	#define aligned_i8vec2 alignas(2) glm::i8vec2
	#define aligned_u8vec2 alignas(2) glm::u8vec2
	#define aligned_i8vec3 alignas(4) glm::i8vec3
	#define aligned_u8vec3 alignas(4) glm::u8vec3
	#define aligned_i8vec4 alignas(4) glm::i8vec4
	#define aligned_u8vec4 alignas(4) glm::u8vec4

	#define aligned_i16vec2 alignas(4) glm::i16vec2
	#define aligned_u16vec2 alignas(4) glm::u16vec2
	#define aligned_i16vec3 alignas(8) glm::i16vec3
	#define aligned_u16vec3 alignas(8) glm::u16vec3
	#define aligned_i16vec4 alignas(8) glm::i16vec4
	#define aligned_u16vec4 alignas(8) glm::u16vec4

	#define aligned_f32vec2 alignas(8) glm::f32vec2
	#define aligned_i32vec2 alignas(8) glm::i32vec2
	#define aligned_u32vec2 alignas(8) glm::u32vec2
	#define aligned_f32vec3 alignas(16) glm::f32vec3
	#define aligned_i32vec3 alignas(16) glm::i32vec3
	#define aligned_u32vec3 alignas(16) glm::u32vec3
	#define aligned_f32vec4 alignas(16) glm::f32vec4
	#define aligned_i32vec4 alignas(16) glm::i32vec4
	#define aligned_u32vec4 alignas(16) glm::u32vec4

	#define aligned_f64vec2 alignas(16) glm::f64vec2
	#define aligned_i64vec2 alignas(16) glm::i64vec2
	#define aligned_u64vec2 alignas(16) glm::u64vec2
	#define aligned_f64vec3 alignas(32) glm::f64vec3
	#define aligned_i64vec3 alignas(32) glm::i64vec3
	#define aligned_u64vec3 alignas(32) glm::u64vec3
	#define aligned_f64vec4 alignas(32) glm::f64vec4
	#define aligned_i64vec4 alignas(32) glm::i64vec4
	#define aligned_u64vec4 alignas(32) glm::u64vec4

	#define aligned_f32mat3x4 alignas(16) glm::f32mat3x4
	#define aligned_f64mat3x4 alignas(32) glm::f64mat3x4
	
	#define aligned_f32mat4 alignas(16) glm::f32mat4
	#define aligned_f64mat4 alignas(32) glm::f64mat4
	
	#define aligned_VkDeviceAddress alignas(8) VkDeviceAddress

	#define STATIC_ASSERT_ALIGNED16_SIZE(T, X) static_assert(sizeof(T) == X && sizeof(T) % 16 == 0);
	#define STATIC_ASSERT_SIZE(T, X) static_assert(sizeof(T) == X);
	#define PUSH_CONSTANT_STRUCT struct
	#define BUFFER_REFERENCE_FORWARD_DECLARE(TypeName)
	#define BUFFER_REFERENCE_STRUCT(align) struct
	#define BUFFER_REFERENCE_STRUCT_READONLY(align) struct
	#define BUFFER_REFERENCE_STRUCT_WRITEONLY(align) struct
	#define BUFFER_REFERENCE_ADDR(type) aligned_VkDeviceAddress
	
#else // GLSL

	#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
	#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
	#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
	#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
	#extension GL_EXT_shader_explicit_arithmetic_types_float32 : enable
	#extension GL_EXT_shader_explicit_arithmetic_types_float64 : enable
	
	#define aligned_int8_t int8_t
	#define aligned_uint8_t uint8_t
	#define aligned_int16_t int16_t
	#define aligned_uint16_t uint16_t
	#define aligned_int32_t int32_t
	#define aligned_uint32_t uint32_t
	#define aligned_int64_t int64_t
	#define aligned_uint64_t uint64_t

	#define aligned_float32_t float32_t
	#define aligned_float64_t float64_t

	#define aligned_i8vec2 i8vec2
	#define aligned_u8vec2 u8vec2
	#define aligned_i8vec3 i8vec3
	#define aligned_u8vec3 u8vec3
	#define aligned_i8vec4 i8vec4
	#define aligned_u8vec4 u8vec4

	#define aligned_i16vec2 i16vec2
	#define aligned_u16vec2 u16vec2
	#define aligned_i16vec3 i16vec3
	#define aligned_u16vec3 u16vec3
	#define aligned_i16vec4 i16vec4
	#define aligned_u16vec4 u16vec4

	#define aligned_f32vec2 f32vec2
	#define aligned_i32vec2 i32vec2
	#define aligned_u32vec2 u32vec2
	#define aligned_f32vec3 f32vec3
	#define aligned_i32vec3 i32vec3
	#define aligned_u32vec3 u32vec3
	#define aligned_f32vec4 f32vec4
	#define aligned_i32vec4 i32vec4
	#define aligned_u32vec4 u32vec4

	#define aligned_f64vec2 f64vec2
	#define aligned_i64vec2 i64vec2
	#define aligned_u64vec2 u64vec2
	#define aligned_f64vec3 f64vec3
	#define aligned_i64vec3 i64vec3
	#define aligned_u64vec3 u64vec3
	#define aligned_f64vec4 f64vec4
	#define aligned_i64vec4 i64vec4
	#define aligned_u64vec4 u64vec4

	#define aligned_f32mat3x4 f32mat3x4
	#define aligned_f64mat3x4 f64mat3x4
	
	#define aligned_f32mat4 f32mat4
	#define aligned_f64mat4 f64mat4
	
	#define aligned_VkDeviceAddress uint64_t
	
	#define STATIC_ASSERT_ALIGNED16_SIZE(T,X)
	#define STATIC_ASSERT_SIZE(T,X)
	#define PUSH_CONSTANT_STRUCT layout(push_constant) uniform
	#define BUFFER_REFERENCE_FORWARD_DECLARE(TypeName) layout(buffer_reference) buffer TypeName;
	#define BUFFER_REFERENCE_STRUCT(align) layout(buffer_reference, std430, buffer_reference_align = align) buffer
	#define BUFFER_REFERENCE_STRUCT_READONLY(align) layout(buffer_reference, std430, buffer_reference_align = align) buffer readonly
	#define BUFFER_REFERENCE_STRUCT_WRITEONLY(align) layout(buffer_reference, std430, buffer_reference_align = align) buffer writeonly
	#define BUFFER_REFERENCE_ADDR(type) type
	
#endif
#line 15 "/home/olivier/projects/chill/src/v4d/core/v4d.h"
#line 5 "/home/olivier/projects/chill/src/v4d/modules/V4D_rtcubes/cpp_glsl.hh"

/////////////////////

#define SET1_BINDING_TLAS 0
#define SET1_BINDING_RENDERER_DATA 1

#define SBT_HITGROUPS_PER_GEOMETRY 1

#define RENDERER_OPTION_TEXTURES (1u<< 0)
#define RENDERER_OPTION_REFLECTIONS (1u<< 1)
#define RENDERER_OPTION_TRANSPARENCY (1u<< 2)
#define RENDERER_OPTION_INDIRECT_LIGHTING (1u<< 3)
#define RENDERER_OPTION_DIRECT_LIGHTING (1u<< 4)
#define RENDERER_OPTION_SOFT_SHADOWS (1u<< 5)

#define RENDERER_DEBUG_MODE_NONE 0
#define RENDERER_DEBUG_MODE_RAYGEN_TIME 1
#define RENDERER_DEBUG_MODE_RAYHIT_TIME 2
#define RENDERER_DEBUG_MODE_RAYINT_TIME 3
#define RENDERER_DEBUG_MODE_TRACE_RAY_COUNT 4
#define RENDERER_DEBUG_MODE_NORMALS 5
#define RENDERER_DEBUG_MODE_MOTION 6
#define RENDERER_DEBUG_MODE_DISTANCE 7
#define RENDERER_DEBUG_MODE_UVS 8
#define RENDERER_DEBUG_MODE_FEFLECTIVITY 9
#define RENDERER_DEBUG_MODE_TRANSPARENCY 10
#define RENDERER_DEBUG_MODE_AIM_RENDERABLE 11
#define RENDERER_DEBUG_MODE_AIM_GEOMETRY 12
#define RENDERER_DEBUG_MODE_AIM_PRIMITIVE 13
#define RENDERER_DEBUG_MODE_GLOBAL_ILLUMINATION 14

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Structs and Buffer References -- Must use aligned_* explicit arithmetic types (or VkDeviceAddress as an uint64_t, or BUFFER_REFERENCE_ADDR(StructType))

struct RayTracingPushConstant {
	aligned_uint32_t cameraIndex;
};
STATIC_ASSERT_SIZE(RayTracingPushConstant, 4)

BUFFER_REFERENCE_STRUCT_READONLY(16) TLASInstance {
	aligned_f32mat3x4 transform;
	aligned_uint32_t instanceCustomIndex_and_mask; // mask>>24, customIndex&0xffffff
	aligned_uint32_t instanceShaderBindingTableRecordOffset_and_flags; // flags>>24
	aligned_VkDeviceAddress accelerationStructureReference;
};
STATIC_ASSERT_ALIGNED16_SIZE(TLASInstance, 64)

BUFFER_REFERENCE_STRUCT(16) GlobalIllumination {
	aligned_f32vec4 radiance;
	aligned_int64_t frameIndex;
	aligned_uint32_t iteration;
	aligned_int32_t lock;
};
STATIC_ASSERT_ALIGNED16_SIZE(GlobalIllumination, 32);

struct RendererData {
	BUFFER_REFERENCE_ADDR(RenderableInstanceData) renderableInstances;
	BUFFER_REFERENCE_ADDR(TLASInstance) tlasInstances;
	BUFFER_REFERENCE_ADDR(AimBuffer) aim;
	BUFFER_REFERENCE_ADDR(GlobalIllumination) globalIllumination;
	aligned_float64_t timestamp;
	aligned_f32vec3 skyLightColor;
	aligned_uint32_t options;
	aligned_f32vec3 sunDir;
	aligned_float32_t wireframeThickness;
	aligned_f32vec4 wireframeColor;
	aligned_i32vec3 worldOrigin;
	aligned_uint32_t globalIlluminationTableCount;
	
	aligned_uint32_t giIteration;
	aligned_int32_t giMaxSamples;
	aligned_int32_t fogSteps;
};
#line 10 "/home/olivier/projects/chill/src/v4d/modules/V4D_rtcubes/base.glsl"
const float EPSILON = 0.00001;

layout(push_constant) uniform PushConstant {
	RayTracingPushConstant pushConstant;
};

layout(set = 1, binding = SET1_BINDING_RENDERER_DATA) buffer RendererDataBuffer { RendererData renderer; };

bool OPTION_TEXTURES = (renderer.options & RENDERER_OPTION_TEXTURES) != 0;
bool OPTION_REFLECTIONS = (renderer.options & RENDERER_OPTION_REFLECTIONS) != 0;
bool OPTION_TRANSPARENCY = (renderer.options & RENDERER_OPTION_TRANSPARENCY) != 0;
bool OPTION_INDIRECT_LIGHTING = (renderer.options & RENDERER_OPTION_INDIRECT_LIGHTING) != 0;
bool OPTION_DIRECT_LIGHTING = (renderer.options & RENDERER_OPTION_DIRECT_LIGHTING) != 0;
bool OPTION_SOFT_SHADOWS = (renderer.options & RENDERER_OPTION_SOFT_SHADOWS) != 0;

#define RAY_PAYLOAD_PRIMARY 0
struct RayPayload {
	vec4 color;
	vec3 localPosition;
	float hitDistance;
	vec3 normal; // world-space
	float reflection;
	uvec2 index;
	int tlasInstanceIndex;
	float totalDistanceFromEye;
	vec3 nextPosition; // For translucency with Refraction
	uint bounces;
};
#ifdef SHADER_RGEN
	layout(location = RAY_PAYLOAD_PRIMARY) rayPayloadEXT RayPayload ray;
#endif
#if defined(SHADER_RCHIT) || defined(SHADER_RAHIT) || defined(SHADER_RMISS)
	#ifdef SHADER_OFFSET_0
		layout(location = RAY_PAYLOAD_PRIMARY) rayPayloadInEXT RayPayload ray;
	#endif
#endif
#if defined(SHADER_RINT) || defined(SHADER_RCHIT) || defined(SHADER_RAHIT)
	#define INSTANCE renderer.renderableInstances[gl_InstanceID]
	#define GEOMETRY INSTANCE.geometries[gl_GeometryIndexEXT*SBT_HITGROUPS_PER_GEOMETRY]
	#define AABB GEOMETRY.aabbs[gl_PrimitiveID]
	#define AABB_MIN vec3(AABB.aabb[0], AABB.aabb[1], AABB.aabb[2])
	#define AABB_MAX vec3(AABB.aabb[3], AABB.aabb[4], AABB.aabb[5])
	#define AABB_CENTER ((AABB_MIN + AABB_MAX) * 0.5)
	#define AABB_CENTER_INT ivec3(round(AABB_CENTER))
	#define MODELVIEW (camera.viewMatrix * mat4(gl_ObjectToWorldEXT))
	#define MVP (camera.projectionMatrix * MODELVIEW)
	#define MVP_AA (camera.projectionMatrixWithTXAA * MODELVIEW)
	#define MVP_HISTORY (camera.projectionMatrix * MODELVIEW_HISTORY)
	#define AIM_GEOMETRY_ID ((uint32_t(gl_InstanceCustomIndexEXT) << 12) | uint32_t(gl_GeometryIndexEXT))
	#define BOX_INTERSECTION_KIND_OUTSIDE_FACE 0
	#define BOX_INTERSECTION_KIND_INSIDE_FACE 1
#endif

#define WORLD2VIEWNORMAL transpose(inverse(mat3(camera.viewMatrix)))
#define VIEW2WORLDNORMAL transpose(mat3(camera.viewMatrix))

#if defined(SHADER_RINT) || defined(SHADER_RCHIT)
	// Intersects ray with a BOX, generating T1 and T2 values
	#define COMPUTE_BOX_INTERSECTION \
		const vec3  _invDir = 1.0 / gl_ObjectRayDirectionEXT;\
		const vec3  _tbot   = _invDir * (AABB_MIN - gl_ObjectRayOriginEXT);\
		const vec3  _ttop   = _invDir * (AABB_MAX - gl_ObjectRayOriginEXT);\
		const vec3  _tmin   = min(_ttop, _tbot);\
		const vec3  _tmax   = max(_ttop, _tbot);\
		const float T1     = max(_tmin.x, max(_tmin.y, _tmin.z));\
		const float T2     = min(_tmax.x, min(_tmax.y, _tmax.z));
	#define RAY_STARTS_OUTSIDE_T1_T2 (gl_RayTminEXT <= T1 && T1 < gl_RayTmaxEXT && T2 > T1)
	#define RAY_STARTS_BETWEEN_T1_T2 (T1 <= gl_RayTminEXT && T2 >= gl_RayTminEXT)
	const float BLOCK_GRID_OFFSET = -0.5; // -0.5 to center blocks on integer grid
	const vec3[6] BOX_NORMAL_DIRS = {
		vec3(-1,0,0),
		vec3(0,-1,0),
		vec3(0,0,-1),
		vec3(+1,0,0),
		vec3(0,+1,0),
		vec3(0,0,+1)
	};
#endif

#ifdef SHADER_RCHIT
	vec3 DoubleSidedNormals(in vec3 worldSpaceNormal) {
		return -sign(dot(worldSpaceNormal, gl_WorldRayDirectionEXT)) * worldSpaceNormal;
	}
	vec3 DoubleSidedNormals(in vec3 worldSpaceNormal, in float bias) {
		return -sign(dot(worldSpaceNormal, gl_WorldRayDirectionEXT)-bias) * worldSpaceNormal;
	}
	void ApplyFresnelReflection(float indexOfRefraction) {
		ray.reflection = Fresnel((camera.viewMatrix * vec4(gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * ray.hitDistance, 1)).xyz, normalize(WORLD2VIEWNORMAL * ray.normal), indexOfRefraction);
	}
	// Compute normal for a box (this method works for boxes with arbitrary width/height/depth)
	#define CLOSEST_HIT_BOX_INTERSECTION_COMPUTE_NORMAL \
		vec3 BOX_SIZE = AABB_MAX - AABB_MIN; uint BOX_FACE; vec3 BOX_NORMAL; vec2 BOX_UV; vec2 BOX_COORD; {\
			const float threshold = EPSILON * ray.totalDistanceFromEye * 0.05;\
			const vec3 absMin = abs(ray.localPosition.xyz - (gl_HitKindEXT == BOX_INTERSECTION_KIND_OUTSIDE_FACE? AABB_MIN.xyz : AABB_MAX.xyz));\
			const vec3 absMax = abs(ray.localPosition.xyz - (gl_HitKindEXT == BOX_INTERSECTION_KIND_OUTSIDE_FACE? AABB_MAX.xyz : AABB_MIN.xyz));\
				if (absMin.x < threshold) {BOX_FACE = 0; BOX_COORD = vec2(ray.localPosition.zy) * vec2(+1,-1) + vec2(BLOCK_GRID_OFFSET); BOX_UV = BOX_COORD / BOX_SIZE.zy;}\
			else if (absMin.y < threshold) {BOX_FACE = 1; BOX_COORD = vec2(ray.localPosition.xz) * vec2(-1,-1) + vec2(BLOCK_GRID_OFFSET); BOX_UV = BOX_COORD / BOX_SIZE.xz;}\
			else if (absMin.z < threshold) {BOX_FACE = 2; BOX_COORD = vec2(ray.localPosition.xy) * vec2(-1,-1) + vec2(BLOCK_GRID_OFFSET); BOX_UV = BOX_COORD / BOX_SIZE.xy;}\
			else if (absMax.x < threshold) {BOX_FACE = 3; BOX_COORD = vec2(ray.localPosition.zy) * vec2(-1,-1) + vec2(BLOCK_GRID_OFFSET); BOX_UV = BOX_COORD / BOX_SIZE.zy;}\
			else if (absMax.y < threshold) {BOX_FACE = 4; BOX_COORD = vec2(ray.localPosition.xz) * vec2(+1,+1) + vec2(BLOCK_GRID_OFFSET); BOX_UV = BOX_COORD / BOX_SIZE.xz;}\
			else if (absMax.z < threshold) {BOX_FACE = 5; BOX_COORD = vec2(ray.localPosition.xy) * vec2(+1,-1) + vec2(BLOCK_GRID_OFFSET); BOX_UV = BOX_COORD / BOX_SIZE.xy;}\
			BOX_NORMAL = BOX_NORMAL_DIRS[BOX_FACE];\
			ray.normal = normalize(mat3(gl_ObjectToWorldEXT) * BOX_NORMAL);\
			if (gl_HitKindEXT == BOX_INTERSECTION_KIND_INSIDE_FACE) {\
				BOX_FACE = (BOX_FACE + 3) % 6;\
			}\
		}
#endif

#define CLOSEST_HIT_BEGIN_T(_t) {\
	ray.hitDistance = _t;\
	ray.totalDistanceFromEye += _t;\
	ray.localPosition = gl_ObjectRayOriginEXT + gl_ObjectRayDirectionEXT * ray.hitDistance;\
	ray.nextPosition = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * ray.hitDistance;\
	ray.index = uvec2(AIM_GEOMETRY_ID, gl_PrimitiveID);\
	ray.tlasInstanceIndex = gl_InstanceID;\
	ray.color = vec4(vec3(0.5), 1.0);\
	ray.reflection = 0;\
}
#define CLOSEST_HIT_BEGIN CLOSEST_HIT_BEGIN_T(gl_HitTEXT)
#define CLOSEST_HIT_END {\
	const float bias = 0.01;\
	float rDotN = dot(gl_WorldRayDirectionEXT, ray.normal);\
	if (rDotN < 0.5 && rDotN > -bias) {\
		vec3 tmp = normalize(cross(gl_WorldRayDirectionEXT, ray.normal));\
		ray.normal = normalize(mix(-gl_WorldRayDirectionEXT, normalize(cross(-gl_WorldRayDirectionEXT, tmp)), 1.0-bias));\
	}\
	ray.normal = DoubleSidedNormals(ray.normal);\
	if (ray.bounces == 0 && camera.debugViewMode == RENDERER_DEBUG_MODE_RAYHIT_TIME) WRITE_DEBUG_TIME\
}

vec4 GetTexture(in sampler2D tex, in vec2 coords, in float t) {
	const uvec2 texSize = textureSize(tex, 0).st;
	const float resolutionRatio = min(texSize.s, texSize.t) / (max(camera.width, camera.height) / camera.renderScale);
	return textureLod(nonuniformEXT(tex), coords, pow(t,0.5) * resolutionRatio - 0.5);
}

#if (defined(SHADER_RCHIT) || defined(SHADER_RAHIT) || defined(SHADER_RMISS)) && defined(SHADER_OFFSET_0)
	vec4 ApplyTexture(in uint index, in vec2 uv) {
		if (index != 0) {
			#if (defined(SHADER_RCHIT) || defined(SHADER_RAHIT) || defined(SHADER_RMISS))
				return GetTexture(textures[nonuniformEXT(index)], uv, ray.totalDistanceFromEye);
			#else
				return texture(textures[nonuniformEXT(index)], uv);
			#endif
		} else return vec4(0);
	}
	vec4 ApplyTexture(in sampler2D tex, in vec2 uv) {
		#if (defined(SHADER_RCHIT) || defined(SHADER_RAHIT) || defined(SHADER_RMISS))
			return GetTexture(tex, uv, ray.totalDistanceFromEye);
		#else
			return texture(tex, uv);
		#endif
	}
#endif

float sdfCube(vec3 p, float r) {
	return max(max(abs(p.x), abs(p.y)), abs(p.z)) - r;
}

float sdfSphere(vec3 p, float r) {
	return length(p) - r;
}

// Global Illumination
uint HashGlobalPosition(uvec3 data) {
	uint hash = 8u, tmp;

	hash += data.x & 0xffffu;
	tmp = (((data.x >> 16) & 0xffffu) << 11) ^ hash;
	hash = (hash << 16) ^ tmp;
	hash += hash >> 11;

	hash += data.y & 0xffffu;
	tmp = (((data.y >> 16) & 0xffffu) << 11) ^ hash;
	hash = (hash << 16) ^ tmp;
	hash += hash >> 11;

	hash += data.z & 0xffffu;
	tmp = (((data.z >> 16) & 0xffffu) << 11) ^ hash;
	hash = (hash << 16) ^ tmp;
	hash += hash >> 11;

	hash ^= hash << 3;
	hash += hash >> 5;
	hash ^= hash << 4;
	hash += hash >> 17;
	hash ^= hash << 25;
	hash += hash >> 6;

	return hash;
}

uvec3 HashGlobalPosition3(uvec3 v) {

	v = v * 1664525u + 1013904223u;

	v.x += v.y*v.z;
	v.y += v.z*v.x;
	v.z += v.x*v.y;

	v ^= v >> 16u;

	v.x += v.y*v.z;
	v.y += v.z*v.x;
	v.z += v.x*v.y;

	return v;
}

uint GetGiIndex(in ivec3 worldPosition) {
	uvec3 p = uvec3(worldPosition + renderer.worldOrigin + ivec3(1<<30));
	return HashGlobalPosition(p) % renderer.globalIlluminationTableCount;
}
GlobalIllumination GetGi(uint index) {
	return renderer.globalIllumination[index];
}

vec3 GetSkyColor(in vec3 dir) {
	return renderer.skyLightColor * mix(vec3(0.7,0.7,1.0), abs(dir), 0.5) * 2;
}

// Debug Stuff
uint64_t startTime = clockARB();
#define WRITE_DEBUG_TIME {imageStore(img_debug, ivec2(gl_LaunchIDEXT.xy), vec4(Heatmap(float(double(clockARB() - startTime) / double(500000.0))), 1));}
#define traceRayEXT {if (camera.debugViewMode == RENDERER_DEBUG_MODE_TRACE_RAY_COUNT) imageStore(img_debug, ivec2(gl_LaunchIDEXT.xy), vec4(0,0,0, imageLoad(img_debug, ivec2(gl_LaunchIDEXT.xy)).a + 1));} traceRayEXT

#ifdef SHADER_RCHIT
	#define SUN_LIGHT_SOLID_ANGLE 0.01
	#define FOG_DISTANCE_FACTOR 300
	
	layout(set = 1, binding = SET1_BINDING_TLAS) uniform accelerationStructureEXT tlas;
	uint seed = InitRandomSeed(InitRandomSeed(gl_LaunchIDEXT.x, gl_LaunchIDEXT.y), uint(camera.frameIndex));
	
	vec3 GetDirectLighting(in vec3 albedo, in float diffuseMultiplier, in float specularMultiplier, in float specularPower) {
		const vec3 ambient = vec3(0);// albedo * 0.03;
		if (OPTION_DIRECT_LIGHTING) {
			if (ray.bounces < 2) {// Direct Lighting (must apply this for diffuse materials only)
				vec3 shadowRayDir = normalize(renderer.sunDir + vec3(0.0012765f));
				if (OPTION_SOFT_SHADOWS) {
					float pointRadius = SUN_LIGHT_SOLID_ANGLE * RandomFloat(seed);
					float pointAngle = RandomFloat(seed) * 2.0 * 3.1415926535;
					vec2 diskPoint = vec2(pointRadius * cos(pointAngle), pointRadius * sin(pointAngle));
					vec3 lightTangent = normalize(cross(shadowRayDir, ray.normal));
					vec3 lightBitangent = normalize(cross(lightTangent, shadowRayDir));
					shadowRayDir = normalize(shadowRayDir + diskPoint.x * lightTangent + diskPoint.y * lightBitangent);
				}
				if (dot(shadowRayDir, ray.normal) > 0.001) {
					vec3 surfacePosition = ray.nextPosition + ray.normal * 0.001;
					vec3 directLighting = vec3(1);
					RayPayload originalRay = ray;
					for (;;) {
						if (dot(directLighting,directLighting) < 0.00001) {
							directLighting = ambient;
							break;
						}
						ray.hitDistance = 0;
						++ray.bounces;
						traceRayEXT(tlas, 0, RENDERABLE_ALL, 0/*rayType*/, SBT_HITGROUPS_PER_GEOMETRY/*nbRayTypes*/, 0/*missIndex*/, surfacePosition, camera.zNear, shadowRayDir, camera.zFar, RAY_PAYLOAD_PRIMARY);
						if (ray.hitDistance == -1) {
							vec3 diffuse = albedo * renderer.skyLightColor * dot(renderer.sunDir, originalRay.normal) * diffuseMultiplier;
							vec3 specular = renderer.skyLightColor * pow(clamp(dot(originalRay.normal, normalize(renderer.sunDir - gl_WorldRayDirectionEXT)), 0, 1), specularPower) * specularMultiplier;
							directLighting *= diffuse + specular;
							break;
						} else if (ray.color.a < 1.0) {
							// Diffuse (within transparent non-air medium, like water or glass)
							directLighting *= 1-ray.color.a;
							surfacePosition = ray.nextPosition + shadowRayDir * 0.001;
						} else {
							directLighting = ambient;
							break;
						}
					}
					ray = originalRay;
					return directLighting;
				}
			}
		}
		return ambient;
	}
	
	void ApplyFog() {
		if (ray.bounces == 0) {
			if (renderer.fogSteps > 0) {
				// Volumetric Fog (with God-Rays)
				const float fogAmount = pow(min(1, ray.hitDistance / FOG_DISTANCE_FACTOR), 1.4) / renderer.fogSteps;
				RayPayload originalRay = ray;
				++ray.bounces;
				for (int i = 0; i < renderer.fogSteps; ++i) {
					float dist = RandomFloat(seed) * originalRay.hitDistance;
					vec3 worldPos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * dist;
					ray.hitDistance = 0;
					traceRayEXT(tlas, 0, RENDERABLE_ALL, 0/*rayType*/, SBT_HITGROUPS_PER_GEOMETRY/*nbRayTypes*/, 0/*missIndex*/, worldPos, 0, normalize(renderer.sunDir + vec3(0.0012765f)), camera.zFar, RAY_PAYLOAD_PRIMARY);
					if (ray.hitDistance == -1) {
						originalRay.color.rgb = mix(originalRay.color.rgb, renderer.skyLightColor, fogAmount);
					} else {
						originalRay.color.rgb = mix(originalRay.color.rgb, vec3(0), fogAmount);
					}
				}
				ray = originalRay;
			} else {
				// Basic Fog
				ray.color.rgb = mix(ray.color.rgb, renderer.skyLightColor, pow(min(1, ray.hitDistance / FOG_DISTANCE_FACTOR), 1.4));
			}
		}
	}
	
#endif

#endif // _RTCUBES_SHADER_BASE_INCLUDED_
#line 2 "/home/olivier/projects/chill/src/v4d/modules/V4D_rtcubes/assets/shaders/raytracing.glsl"


#line 168 "/home/olivier/projects/chill/src/v4d/modules/V4D_rtcubes/assets/shaders/raytracing.glsl"
void main() {
	ray.hitDistance = -1;
	ray.reflection = 0;
	ray.normal = vec3(0,0,0);
	ray.color = vec4(GetSkyColor(gl_WorldRayDirectionEXT), 1.0);
	ray.index = uvec2(0,0);
	ray.tlasInstanceIndex = -1;
	if (ray.bounces == 0) {
		const float sunGlowAngle = 0.01;
		const float sunSolidAngle = 0.0003;
		ray.color.rgb += ray.color.rgb * pow(smoothstep(0.5, 1, dot(gl_WorldRayDirectionEXT, renderer.sunDir)), 2);
		ray.color.rgb += ray.color.rgb * pow(smoothstep(1-sunGlowAngle, 1.002, dot(gl_WorldRayDirectionEXT, renderer.sunDir)), 2) * 0.5;
		ray.color.rgb += ray.color.rgb * smoothstep(1-sunSolidAngle, 1, dot(gl_WorldRayDirectionEXT, renderer.sunDir)) * 100;
	}
	// if (dot(gl_WorldRayDirectionEXT, vec3(0,1,0)) < 0) {
	// 	ray.color.rgb *= vec3(0.3);
	// }
}

///////////////////////////////////////////////////////////////////////////////////
// Default hit groups

