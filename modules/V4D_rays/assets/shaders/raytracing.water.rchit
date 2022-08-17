#version 460 core

#extension GL_GOOGLE_cpp_style_line_directive : enable

#define _DEBUG
#define GLSL 1
#define SHADER_RCHIT
#define SHADER_SUBPASS_2
#define SHADER_OFFSET_2

#line 1 "/home/olivier/projects/chill/src/v4d/modules/V4D_rays/assets/shaders/../../common.glsl.h"
#line 2 "/home/olivier/projects/chill/src/v4d/modules/V4D_rays/assets/shaders/../../common.glsl.h"
#if GLSL
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
#define RENDER_OPTION_TAA (1u<< 0)
#define RENDER_OPTION_SSAO (1u<< 1)
#define RENDER_OPTION_TEMPORAL_UPSCALING (1u<< 2)
#define RENDER_OPTION_TONE_MAPPING (1u<< 3)
#define RENDER_OPTION_ACCUMULATE (1u<< 4)

// Debug view modes
#define RENDER_DEBUG_VIEWMODE_NONE 0

// Configuration
#define TAA_SAMPLES 16
#define MAX_CAMERAS 64
#define MAX_TEXTURE_BINDINGS 65535

#define NB_RENDERABLE_TYPES 8 // this will NOT change, it is a limitation of VK_KHR_ray_tracing
// Renderable types
#define RENDERABLE_TYPE_VOXEL 0 // Standard voxel geometries, typically terrain, structures and vehicles
#define RENDERABLE_TYPE_MOB 1 // Mobile entities that are very dynamic, does not affect GI, typically players avatars and NPCs
#define RENDERABLE_TYPE_WATER 2 // Transparent geometries that can include other geometries within it, affecting the visual environment, typically Water
#define RENDERABLE_TYPE_CLOUD 3 // Volumetric non-solid geometries that is often ray-marched into, typically Clouds, Smoke and Atmosphere
#define RENDERABLE_TYPE_CLUTTER 4 // Small geometries usually present in very large quantities, typically Rocks and Grass
#define RENDERABLE_TYPE_LIGHT_RADIUS 5 // Special geometry that denotes a light's radius, to be point-traced for finding relevant light sources that may be shining on surfaces for direct lighting
#define RENDERABLE_TYPE_OVERLAY 6 // Geometry that is ONLY visible from primary rays and that do not cast shadows, typically holograms and debug stuff
// #define RENDERABLE_TYPE_OTHER 7 // 

#define RENDERABLE_MASK_VOXEL (1u<< RENDERABLE_TYPE_VOXEL)
#define RENDERABLE_MASK_MOB (1u<< RENDERABLE_TYPE_MOB)
#define RENDERABLE_MASK_WATER (1u<< RENDERABLE_TYPE_WATER)
#define RENDERABLE_MASK_CLUTTER (1u<< RENDERABLE_TYPE_CLUTTER)
#define RENDERABLE_MASK_CLOUD (1u<< RENDERABLE_TYPE_CLOUD)
#define RENDERABLE_MASK_LIGHT_RADIUS (1u<< RENDERABLE_TYPE_LIGHT_RADIUS)
#define RENDERABLE_MASK_OVERLAY (1u<< RENDERABLE_TYPE_OVERLAY)

#define RENDERABLE_STANDARD (0xff & ~RENDERABLE_MASK_LIGHT_RADIUS)
#define RENDERABLE_STANDARD_EXCEPT_WATER (RENDERABLE_STANDARD & ~RENDERABLE_MASK_WATER)

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
	aligned_f32mat4 projectionMatrixWithTAA;
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
	aligned_float32_t debugViewScale;
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

BUFFER_REFERENCE_STRUCT(16) AimBuffer {
	aligned_f32vec3 localPosition;
	aligned_uint32_t aimID;
	aligned_f32vec3 worldSpaceHitNormal;
	aligned_uint32_t aabbIndex;
	aligned_f32vec3 worldSpacePosition; // MUST COMPENSATE FOR ORIGIN RESET
	aligned_float32_t hitDistance;
	aligned_f32vec4 color;
	aligned_f32vec3 viewSpaceHitNormal;
	aligned_uint32_t tlasInstanceIndex;
	aligned_f32vec3 _unused;
	aligned_uint32_t geometryIndex;
};
STATIC_ASSERT_ALIGNED16_SIZE(AimBuffer, 96)

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

#define WORLD2VIEWNORMAL transpose(inverse(mat3(camera.viewMatrix)))
#define VIEW2WORLDNORMAL transpose(mat3(camera.viewMatrix))

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
	vec4 clipSpace = mat4(camera.projectionMatrixWithTAA) * mat4(camera.viewMatrix) * vec4(worldSpacePos, 1);
	return clipSpace.z / clipSpace.w;
}

vec3 GetViewSpacePositionFromDepthAndUV(float depth, vec2 uv) {
	vec4 viewSpacePos = inverse(camera.projectionMatrixWithTAA) * vec4((uv * 2 - 1), depth, 1);
	viewSpacePos.xyz /= viewSpacePos.w;
	if (depth == 0) viewSpacePos.z = camera.zFar;
	return viewSpacePos.xyz;
}

float GetTrueDistanceFromDepthBuffer(float depth) {
	if (depth == 0 || depth == 1) return camera.zFar;
	return 2.0 * (camera.zFar * camera.zNear) / (camera.zNear + camera.zFar - (depth * 2.0 - 1.0) * (camera.zNear - camera.zFar));
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
#line 4 "/home/olivier/projects/chill/src/v4d/modules/V4D_rays/assets/shaders/../../common.glsl.h"
	#extension GL_EXT_ray_tracing : require
	#extension GL_EXT_shader_atomic_float : require
	#extension GL_ARB_shader_clock : enable
#endif

#define RAY_MAX_RECURSION 8

#define SET1_BINDING_TLAS 0
#define SET1_BINDING_RENDERER_DATA 1
#define SET1_BINDING_RT_PAYLOAD_IMAGE 2

#define RENDERER_DEBUG_MODE_NONE 0
#define RENDERER_DEBUG_MODE_RAYGEN_TIME 1
#define RENDERER_DEBUG_MODE_RAYHIT_TIME 2
#define RENDERER_DEBUG_MODE_RAYINT_TIME 3
#define RENDERER_DEBUG_MODE_TRACE_RAY_COUNT 4
#define RENDERER_DEBUG_MODE_NORMALS 5
#define RENDERER_DEBUG_MODE_MOTION 6
#define RENDERER_DEBUG_MODE_DISTANCE 7
#define RENDERER_DEBUG_MODE_UVS 8
#define RENDERER_DEBUG_MODE_REFLECTIVITY 9
#define RENDERER_DEBUG_MODE_TRANSPARENCY 10
#define RENDERER_DEBUG_MODE_AIM_RENDERABLE 11
#define RENDERER_DEBUG_MODE_AIM_GEOMETRY 12
#define RENDERER_DEBUG_MODE_AIM_PRIMITIVE 13
#define RENDERER_DEBUG_MODE_GLOBAL_ILLUMINATION 14
#define RENDERER_DEBUG_MODE_TEST 15

struct RayTracingPushConstant {
	aligned_uint32_t cameraIndex;
	aligned_uint32_t stage;
};
STATIC_ASSERT_SIZE(RayTracingPushConstant, 8)

BUFFER_REFERENCE_STRUCT(16) GlobalIllumination {
	aligned_f32vec4 radiance;
	aligned_int64_t frameIndex;
	aligned_uint32_t iteration;
	aligned_int32_t lock;
};
STATIC_ASSERT_ALIGNED16_SIZE(GlobalIllumination, 32);

BUFFER_REFERENCE_STRUCT_READONLY(16) TLASInstance {
	aligned_f32mat3x4 transform;
	aligned_uint32_t instanceCustomIndex_and_mask; // mask>>24, customIndex&0xffffff
	aligned_uint32_t instanceShaderBindingTableRecordOffset_and_flags; // flags>>24
	aligned_VkDeviceAddress accelerationStructureReference;
};
STATIC_ASSERT_ALIGNED16_SIZE(TLASInstance, 64)

BUFFER_REFERENCE_STRUCT_READONLY(16) GeometryData {
	BUFFER_REFERENCE_ADDR(AabbData) aabbs;
	aligned_VkDeviceAddress vertices;
	aligned_VkDeviceAddress indices32;
	aligned_VkDeviceAddress indices16;
};
STATIC_ASSERT_ALIGNED16_SIZE(GeometryData, 32)

BUFFER_REFERENCE_STRUCT_READONLY(16) RenderableInstanceData {
	BUFFER_REFERENCE_ADDR(GeometryData) geometries;
	aligned_uint64_t data; // custom data defined per-shader
};
STATIC_ASSERT_ALIGNED16_SIZE(RenderableInstanceData, 16)

struct RendererData {
	BUFFER_REFERENCE_ADDR(RenderableInstanceData) renderableInstances;
	BUFFER_REFERENCE_ADDR(TLASInstance) tlasInstances;
	BUFFER_REFERENCE_ADDR(AimBuffer) aim;
	BUFFER_REFERENCE_ADDR(GlobalIllumination) globalIllumination;
	aligned_f32vec3 sunDir;
	aligned_uint32_t giIteration;
	aligned_f32vec3 skyLightColor;
	aligned_float32_t waterWaves;
	aligned_f32vec3 wireframeColor;
	aligned_float32_t wireframeThickness;
	aligned_i32vec3 worldOrigin;
	aligned_uint32_t globalIlluminationTableCount;
	aligned_float64_t timestamp;
	aligned_float64_t _unused_3;
};
STATIC_ASSERT_ALIGNED16_SIZE(RendererData, 112);
#line 2 "/home/olivier/projects/chill/src/v4d/modules/V4D_rays/assets/shaders/raytracing.glsl"
#line 1 "/home/olivier/projects/chill/src/v4d/modules/CHILL_terrain/assets/shaders/voxel.hh"
#line 2 "/home/olivier/projects/chill/src/v4d/modules/CHILL_terrain/assets/shaders/voxel.hh"
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
#line 3 "/home/olivier/projects/chill/src/v4d/modules/CHILL_terrain/assets/shaders/voxel.hh"

#if 0 // 8-bit indexing
	/* 
	TerrainStack (Renderable)
		a Stack consists of 256 chunks vertically
		a Chunk contains 8x4x8 = 256 voxels of 1m
		a Voxel contains 4x4x4 = 64 smaller voxels of 25cm
	*/
	#define VOXEL_INDEX_BITS_XZ 3
	#define VOXEL_INDEX_BITS_Y 2
#else // 16-bit indexing (with padding)
	/* 
	TerrainStack (Renderable)
		a Stack consists of 64 chunks vertically
		a Chunk contains 16x16x16 = 4096 voxels of 1m
		a Voxel contains 4x4x4 = 64 smaller voxels of 25cm
	*/
	#define VOXEL_INDEX_BITS_XZ 4
	#define VOXEL_INDEX_BITS_Y 4
#endif

#define VOXELS_TOTAL_HEIGHT 1024
#define VOXEL_GRID_OFFSET -0.5 // -0.5 to put the center of voxels on integer grid
#define VOXEL_CHUNK_LOD_START_DISTANCE 300


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#if (VOXEL_INDEX_BITS_XZ+VOXEL_INDEX_BITS_XZ+VOXEL_INDEX_BITS_Y == 8) || (VOXEL_INDEX_BITS_XZ+VOXEL_INDEX_BITS_XZ+VOXEL_INDEX_BITS_Y == 16)
	#define VOXEL_INDEX_PADDING 0
#else
	#if (VOXEL_INDEX_BITS_XZ+VOXEL_INDEX_BITS_XZ+VOXEL_INDEX_BITS_Y) < 8
		#define VOXEL_INDEX_PADDING (8 - (VOXEL_INDEX_BITS_XZ+VOXEL_INDEX_BITS_XZ+VOXEL_INDEX_BITS_Y))
	#else
		#define VOXEL_INDEX_PADDING (16 - (VOXEL_INDEX_BITS_XZ+VOXEL_INDEX_BITS_XZ+VOXEL_INDEX_BITS_Y))
	#endif
#endif
#define VOXEL_INDEX_TOTAL_BITS (VOXEL_INDEX_BITS_XZ + VOXEL_INDEX_BITS_XZ + VOXEL_INDEX_BITS_Y + VOXEL_INDEX_PADDING)
#if VOXEL_INDEX_TOTAL_BITS == 8
	#define VOXEL_INDEX_TYPE uint8_t
#else
	#if VOXEL_INDEX_TOTAL_BITS == 16
		#define VOXEL_INDEX_TYPE uint16_t
	#endif
#endif
#define VOXEL_CHUNK_LOD_LIMIT 7 // 0-7 (must fit in 3 bits)
#define VOXEL_CHUNK_LOD_DISTANCE(lod) (uint(VOXEL_CHUNK_LOD_START_DISTANCE) << uint(lod))
#define VOXEL_GRID_SIZE_HD 4
#define VOXELS_XZ (1 << VOXEL_INDEX_BITS_XZ)
#define VOXELS_Y (1 << VOXEL_INDEX_BITS_Y)
#define VOXELS_X VOXELS_XZ
#define VOXELS_Z VOXELS_XZ
#define CHUNKS_Y (VOXELS_TOTAL_HEIGHT / VOXELS_Y)
#define VOXELS_PER_CHUNK (VOXELS_X*VOXELS_Y*VOXELS_Z)
#define VOXEL_STACK_HEIGHT (CHUNKS_Y*VOXELS_Y)
#if CHUNKS_Y <= 256
	#define ChunkIndex uint8_t
#else
	#define ChunkIndex uint16_t
#endif
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define VOXEL_EMPTY 0
#ifdef _WIN32
	#define VOXEL_FULL 0xffffffffffffffffull
#else
	#define VOXEL_FULL 0xfffffffffffffffful
#endif

#ifdef __cplusplus // C++
	static_assert(sizeof(VOXEL_INDEX_TYPE) == 2);
	union VoxelIndex {
		VOXEL_INDEX_TYPE index;
		struct {
			VOXEL_INDEX_TYPE x : VOXEL_INDEX_BITS_XZ;
			VOXEL_INDEX_TYPE z : VOXEL_INDEX_BITS_XZ;
			VOXEL_INDEX_TYPE y : VOXEL_INDEX_BITS_Y;
			#if VOXEL_INDEX_PADDING > 0
				VOXEL_INDEX_TYPE _ : VOXEL_INDEX_PADDING;
			#endif
		};
		VoxelIndex(VOXEL_INDEX_TYPE index = 0) noexcept : index(index) {
			assert(index < VOXELS_PER_CHUNK);
		}
		VoxelIndex(const glm::ivec3& p) noexcept : x(p.x), z(p.z), y(p.y)
			#if VOXEL_INDEX_PADDING > 0
				,_(0)
			#endif
		{
			assert(p.x >= 0);
			assert(p.z >= 0);
			assert(p.y >= 0);
			assert(p.x < VOXELS_X);
			assert(p.z < VOXELS_Z);
			assert(p.y < VOXELS_Y);
			assert(index < VOXELS_PER_CHUNK);
			assert(p.x == x);
			assert(p.y == y);
			assert(p.z == z);
			#if VOXEL_INDEX_PADDING > 0
				assert(_ == 0);
			#endif
		}
		operator glm::ivec3() const noexcept {
			return {x,y,z};
		}
		glm::ivec3 Position() const noexcept {
			return {x,y,z};
		}
		operator VOXEL_INDEX_TYPE() const noexcept {
			assert(index < VOXELS_PER_CHUNK);
			return index;
		}
		VoxelIndex operator + (const glm::ivec3& offset) const noexcept {
			return glm::ivec3{x,y,z} + offset;
		}
		bool Continue() {
			if (index == VOXELS_PER_CHUNK-1) return false;
			++index;
			return true;
		}
	};
	STATIC_ASSERT_SIZE(VoxelIndex, sizeof(VOXEL_INDEX_TYPE));
#else // GLSL
	#define VoxelIndex(x,y,z) (VOXEL_INDEX_TYPE(x) | (VOXEL_INDEX_TYPE(z) << VOXEL_INDEX_BITS_XZ) | (VOXEL_INDEX_TYPE(y) << (VOXEL_INDEX_BITS_XZ+VOXEL_INDEX_BITS_XZ)))
	#define VoxelIndexHD(x,y,z) (uint8_t(x) | (uint8_t(z) << 2) | (uint8_t(y) << 4))
	#define VoxelFillBitHD(iPos) (1ul << VoxelIndexHD(iPos.x, iPos.y, iPos.z))
	#define VoxelIndex_iPos(index) ivec3(\
		int32_t(uint32_t(index) & ((1u << VOXEL_INDEX_BITS_XZ) - 1)),\
		int32_t((uint32_t(index) & (((1u << VOXEL_INDEX_BITS_Y) - 1) << (VOXEL_INDEX_BITS_XZ+VOXEL_INDEX_BITS_XZ))) >> (VOXEL_INDEX_BITS_XZ + VOXEL_INDEX_BITS_XZ)),\
		int32_t((uint32_t(index) & (((1u << VOXEL_INDEX_BITS_XZ) - 1) << VOXEL_INDEX_BITS_XZ)) >> VOXEL_INDEX_BITS_XZ)\
	)
#endif

struct Voxel {
	uint64_t fill;
	uint16_t type;
	uint8_t data;
	#ifdef __cplusplus
		Voxel(uint64_t fill = 0, uint16_t type = 0, uint8_t data = 0)
		: fill(fill), type(type), data(data) {}
	#endif
};
BUFFER_REFERENCE_STRUCT_READONLY(16) ChunkVoxelData {
	aligned_uint64_t fill[VOXELS_PER_CHUNK]; // bitfield for 4x4x4 hd voxels
	aligned_uint16_t type[VOXELS_PER_CHUNK]; // points to a callable shader
	aligned_uint8_t data[VOXELS_PER_CHUNK]; // arbitrary data for use by callable shader
	aligned_i32vec3 aabbOffset;
	aligned_uint32_t voxelSize;
	aligned_uint16_t bounds[6];
	aligned_uint32_t voxelCount;
};
STATIC_ASSERT_ALIGNED16_SIZE(ChunkVoxelData, (8+2+1)*VOXELS_PER_CHUNK + 16 + 16);

BUFFER_REFERENCE_FORWARD_DECLARE(ChunkData)
BUFFER_REFERENCE_STRUCT_READONLY(16) ChunkData {
	BUFFER_REFERENCE_ADDR(ChunkVoxelData) voxels;
	aligned_uint64_t _unused;
};
STATIC_ASSERT_ALIGNED16_SIZE(ChunkData, 16);

#ifndef __cplusplus // GLSL
	bool IsVoxel(in uint64_t chunkID, in ivec3 iPos) {return chunkID != 0 && uint64_t(ChunkData(chunkID).voxels) != 0 && iPos.x >= 0 && iPos.y >= 0 && iPos.z >= 0 && iPos.x < VOXELS_X && iPos.y < VOXELS_Y && iPos.z < VOXELS_Z;}
	
	// Voxel Surface Shaders
	struct VoxelSurface {
		vec4 color;
		vec3 normal;
		float diffuse;
		vec3 emission;
		float specular;
		vec3 posInVoxel;
		float metallic; // set negative value to automatically apply a shiny clearcoat Fresnel-based reflection, the absolute value will be used as the reflection logarithm
		vec2 uv;
		float ior;
		uint16_t voxelIndex;
		uint8_t data;
		uint8_t face;
		uint64_t fill;
		uint64_t chunkAddr;
	};
	#define VOXEL_SURFACE_CALLABLE 0
	#if defined(SHADER_RCHIT)
		layout(location = VOXEL_SURFACE_CALLABLE) callableDataEXT VoxelSurface surface;
	#endif
	#if defined(SHADER_RCALL)
		layout(location = VOXEL_SURFACE_CALLABLE) callableDataInEXT VoxelSurface surface;
		layout (constant_id = 0) const uint32_t textureID = 0;
		vec4 SampleTexture(uint index) {
			return textureLod(textures[nonuniformEXT(textureID + index)], surface.uv, 0);
		}
	#endif
	
#endif
#line 3 "/home/olivier/projects/chill/src/v4d/modules/V4D_rays/assets/shaders/raytracing.glsl"
#line 1 "/home/olivier/projects/chill/src/v4d/modules/V4D_rays/assets/shaders/../../noise.glsl"
vec4 _permute(vec4 x){return mod(((x*34.0)+1.0)*x, 289.0);} // used for Simplex
dvec4 _permute(dvec4 x){return mod(((x*34.0)+1.0)*x, 289.0);} // used for Simplex

// simple-precision Simplex noise, suitable for pos range (-1M, +1M) with a step of 0.001 and gradient of 1.0
// Returns a float value between -1.000 and +1.000 with a distribution that strongly tends towards the center (0.5)
float Simplex(vec3 pos){
	const vec2 C = vec2(1.0/6.0, 1.0/3.0);
	const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);

	vec3 i = floor(pos + dot(pos, C.yyy));
	vec3 x0 = pos - i + dot(i, C.xxx);

	vec3 g = step(x0.yzx, x0.xyz);
	vec3 l = 1.0 - g;
	vec3 i1 = min( g.xyz, l.zxy);
	vec3 i2 = max( g.xyz, l.zxy);

	vec3 x1 = x0 - i1 + 1.0 * C.xxx;
	vec3 x2 = x0 - i2 + 2.0 * C.xxx;
	vec3 x3 = x0 - 1. + 3.0 * C.xxx;

	i = mod(i, 289.0); 
	vec4 p = _permute(_permute(_permute(i.z + vec4(0.0, i1.z, i2.z, 1.0)) + i.y + vec4(0.0, i1.y, i2.y, 1.0)) + i.x + vec4(0.0, i1.x, i2.x, 1.0));

	float n_ = 1.0/7.0;
	vec3  ns = n_ * D.wyz - D.xzx;

	vec4 j = p - 49.0 * floor(p * ns.z *ns.z);

	vec4 x_ = floor(j * ns.z);
	vec4 y_ = floor(j - 7.0 * x_);

	vec4 x = x_ *ns.x + ns.yyyy;
	vec4 y = y_ *ns.x + ns.yyyy;
	vec4 h = 1.0 - abs(x) - abs(y);

	vec4 b0 = vec4(x.xy, y.xy);
	vec4 b1 = vec4(x.zw, y.zw);

	vec4 s0 = floor(b0)*2.0 + 1.0;
	vec4 s1 = floor(b1)*2.0 + 1.0;
	vec4 sh = -step(h, vec4(0.0));

	vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy;
	vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww;

	vec3 p0 = vec3(a0.xy,h.x);
	vec3 p1 = vec3(a0.zw,h.y);
	vec3 p2 = vec3(a1.xy,h.z);
	vec3 p3 = vec3(a1.zw,h.w);

	vec4 norm = 1.79284291400159 - 0.85373472095314 * vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3));
	p0 *= norm.x;
	p1 *= norm.y;
	p2 *= norm.z;
	p3 *= norm.w;

	vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
	return 42.0 * dot(m*m*m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
}

float SimplexFractal(vec3 pos, int octaves) {
	float amplitude = 0.533333333333333;
	float frequency = 1.0;
	float f = Simplex(pos * frequency);
	for (int i = 1; i < octaves; ++i) {
		amplitude /= 2.0;
		frequency *= 2.0;
		f += amplitude * Simplex(pos * frequency);
	}
	return f;
}

#define APPLY_NORMAL_BUMP_NOISE(_noiseFunc, _position, _normal, _waveHeight) {\
	vec3 _tangentX = normalize(cross(normalize(vec3(0.356,1.2145,0.24537))/* fixed arbitrary vector in object space */, _normal));\
	vec3 _tangentY = normalize(cross(_normal, _tangentX));\
	mat3 _TBN = mat3(_tangentX, _tangentY, _normal);\
	float _altitudeTop = _noiseFunc(_position + _tangentY*_waveHeight);\
	float _altitudeBottom = _noiseFunc(_position - _tangentY*_waveHeight);\
	float _altitudeRight = _noiseFunc(_position + _tangentX*_waveHeight);\
	float _altitudeLeft = _noiseFunc(_position - _tangentX*_waveHeight);\
	vec3 _bump = normalize(vec3((_altitudeRight-_altitudeLeft), (_altitudeBottom-_altitudeTop), 2));\
	_normal = normalize(_TBN * _bump);\
}
#line 4 "/home/olivier/projects/chill/src/v4d/modules/V4D_rays/assets/shaders/raytracing.glsl"
#line 1 "/home/olivier/projects/chill/src/v4d/game/water.hh"
#line 2 "/home/olivier/projects/chill/src/v4d/game/water.hh"

#define WATER_LEVEL 199.25
#define WATER_MAX_LIGHT_DEPTH 32
#define WATER_IOR 1.33
#define WATER_OPACITY 0.5
#define WATER_COLOR (vec3(0.3,0.5,0.8) * renderer.skyLightColor)
#line 5 "/home/olivier/projects/chill/src/v4d/modules/V4D_rays/assets/shaders/raytracing.glsl"

layout(set = 1, binding = SET1_BINDING_TLAS) uniform accelerationStructureEXT tlas;
layout(set = 1, binding = SET1_BINDING_RENDERER_DATA) buffer RendererDataBuffer { RendererData renderer; };
layout(set = 1, binding = SET1_BINDING_RT_PAYLOAD_IMAGE, rgba8ui) uniform uimage2D rtPayloadImage; // Recursions, Shadow, Gi, Underwater

layout(push_constant) uniform PushConstant {
	RayTracingPushConstant pushConstant;
};

struct RayPayload {
	vec4 color;
	vec3 normal;
	float _unused1;
	vec3 localPosition;
	float _unused2;
	vec3 worldPosition;
	float hitDistance;
	int id;
	int renderableIndex;
	int geometryIndex;
	int primitiveIndex;
};

#ifdef SHADER_RGEN
	layout(location = 0) rayPayloadEXT RayPayload ray;
#endif
#if defined(SHADER_RCHIT) || defined(SHADER_RAHIT) || defined(SHADER_RMISS)
	layout(location = 0) rayPayloadInEXT RayPayload ray;
#endif

ivec2 COORDS = ivec2(gl_LaunchIDEXT.xy);

uint64_t startTime = clockARB();
#define WRITE_DEBUG_TIME {float elapsedTime = imageLoad(img_debug, COORDS).a + float(clockARB() - startTime); imageStore(img_debug, COORDS, vec4(0,0,0, elapsedTime));}
const float EPSILON = 0.00001;
#define traceRayEXT {if (camera.debugViewMode == RENDERER_DEBUG_MODE_TRACE_RAY_COUNT) imageStore(img_debug, COORDS, imageLoad(img_debug, COORDS) + uvec4(0,0,0,1));} traceRayEXT
#define DEBUG_TEST(color) {if (camera.debugViewMode == RENDERER_DEBUG_MODE_TEST) imageStore(img_debug, COORDS, color);}
#define RAY_RECURSIONS imageLoad(rtPayloadImage, COORDS).r
#define RAY_RECURSION_PUSH imageStore(rtPayloadImage, COORDS, imageLoad(rtPayloadImage, COORDS) + u8vec4(1,0,0,0));
#define RAY_RECURSION_POP imageStore(rtPayloadImage, COORDS, imageLoad(rtPayloadImage, COORDS) - u8vec4(1,0,0,0));
#define RAY_IS_SHADOW (imageLoad(rtPayloadImage, COORDS).g > 0)
#define RAY_SHADOW_PUSH imageStore(rtPayloadImage, COORDS, imageLoad(rtPayloadImage, COORDS) + u8vec4(0,1,0,0));
#define RAY_SHADOW_POP imageStore(rtPayloadImage, COORDS, imageLoad(rtPayloadImage, COORDS) - u8vec4(0,1,0,0));
#define RAY_IS_GI (imageLoad(rtPayloadImage, COORDS).b > 0)
#define RAY_GI_PUSH imageStore(rtPayloadImage, COORDS, imageLoad(rtPayloadImage, COORDS) + u8vec4(0,0,1,0));
#define RAY_GI_POP imageStore(rtPayloadImage, COORDS, imageLoad(rtPayloadImage, COORDS) - u8vec4(0,0,1,0));
#define RAY_IS_UNDERWATER (imageLoad(rtPayloadImage, COORDS).a > 0)
#define RAY_UNDERWATER_PUSH imageStore(rtPayloadImage, COORDS, imageLoad(rtPayloadImage, COORDS) + u8vec4(0,0,0,1));
#define RAY_UNDERWATER_POP imageStore(rtPayloadImage, COORDS, imageLoad(rtPayloadImage, COORDS) - u8vec4(0,0,0,1));
#define INSTANCE renderer.renderableInstances[gl_InstanceID]
#define GEOMETRY INSTANCE.geometries[gl_GeometryIndexEXT]
#define AABB GEOMETRY.aabbs[gl_PrimitiveID]
#define AABB_MIN vec3(AABB.aabb[0], AABB.aabb[1], AABB.aabb[2])
#define AABB_MAX vec3(AABB.aabb[3], AABB.aabb[4], AABB.aabb[5])
#define AABB_CENTER ((AABB_MIN + AABB_MAX) * 0.5)
#define AABB_CENTER_INT ivec3(round(AABB_CENTER))
#define MODELVIEW (camera.viewMatrix * mat4(gl_ObjectToWorldEXT))
#define MVP (camera.projectionMatrix * MODELVIEW)
#define MVP_AA (camera.projectionMatrixWithTAA * MODELVIEW)
#define MVP_HISTORY (camera.projectionMatrix * MODELVIEW_HISTORY)
#define COMPUTE_BOX_INTERSECTION \
	const vec3 _tbot = (AABB_MIN - gl_ObjectRayOriginEXT) / gl_ObjectRayDirectionEXT;\
	const vec3 _ttop = (AABB_MAX - gl_ObjectRayOriginEXT) / gl_ObjectRayDirectionEXT;\
	const vec3 _tmin = min(_ttop, _tbot);\
	const vec3 _tmax = max(_ttop, _tbot);\
	const float T1 = max(_tmin.x, max(_tmin.y, _tmin.z));\
	const float T2 = min(_tmax.x, min(_tmax.y, _tmax.z));
#define RAY_STARTS_OUTSIDE_T1_T2 (gl_RayTminEXT <= T1 && T1 < gl_RayTmaxEXT && T2 > T1)
#define RAY_STARTS_BETWEEN_T1_T2 (T1 <= gl_RayTminEXT && T2 >= gl_RayTminEXT)
#define WATER_INTERSECTION_UNDER 0
#define WATER_INTERSECTION_ABOVE 1

uint seed = InitRandomSeed(InitRandomSeed(gl_LaunchIDEXT.x, gl_LaunchIDEXT.y), uint(camera.frameIndex));
uint coherentSeed = InitRandomSeed(uint(camera.frameIndex),0);
CameraData cam = cameras[pushConstant.cameraIndex];

#define MAX_GI_ACCUMULATION 256
#define ACCUMULATOR_MAX_FRAME_INDEX_DIFF 500

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

uint GetGiIndex(in vec3 worldPosition, uint level) {
	uint halfCount = renderer.globalIlluminationTableCount / 2;
	uvec3 p = uvec3(ivec3(round(worldPosition)) - renderer.worldOrigin + ivec3(1<<30));
	return (HashGlobalPosition(p) % halfCount) + level * halfCount;
}
#define GetGi(i) renderer.globalIllumination[i]

float sdfSphere(vec3 p, float r) {
	return length(p) - r;
}


#line 762 "/home/olivier/projects/chill/src/v4d/modules/V4D_rays/assets/shaders/raytracing.glsl"

void ApplyUnderwaterFog() {
	const vec3 origin = gl_WorldRayOriginEXT;
	const vec3 dir = gl_WorldRayDirectionEXT;
	const float distFactor = clamp(ray.hitDistance / WATER_MAX_LIGHT_DEPTH, 0 ,1);
	const float fogStrength = max(WATER_OPACITY, pow(distFactor, 0.25));
	const vec3 fogColor = WATER_COLOR;

	ray.color.rgb = mix(ray.color.rgb, vec3(0), pow(clamp(ray.hitDistance / WATER_MAX_LIGHT_DEPTH, 0, 1), 0.5));
	ray.color.rgb = min(mix(normalize(fogColor), ray.color.rgb, 0.999), mix(ray.color.rgb, fogColor, fogStrength));
}

float WaterWaves(vec3 pos) {
	return 0
		+ Simplex(vec3(pos.xz*0.05, float(renderer.timestamp - pos.z*0.5)*0.5))*2
		+ Simplex(vec3(pos.xz*0.3, float(renderer.timestamp - pos.z)))
		+ Simplex(vec3(pos.xz*vec2(2, 4), float(renderer.timestamp - pos.z*2)))*0.5
	;
}

void SetHitWater() {
	ray.id = gl_InstanceCustomIndexEXT;
	ray.renderableIndex = gl_InstanceID;
	ray.geometryIndex = gl_GeometryIndexEXT;
	ray.primitiveIndex = gl_PrimitiveID;
	ray.localPosition = gl_ObjectRayOriginEXT + gl_ObjectRayDirectionEXT * gl_HitTEXT;
	ray.worldPosition = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
}

void main() {
	
	ray.hitDistance = gl_HitTEXT;
	ray.normal = vec3(0,1,0);
	ray.color = vec4(vec3(0), 1);
	
	if (RAY_RECURSIONS >= RAY_MAX_RECURSION) {
		ray.id = -1;
		ray.renderableIndex = -1;
		return;
	}
	
	bool rayIsGi = RAY_IS_GI;
	bool rayIsShadow = RAY_IS_SHADOW;
	vec3 worldPosition = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
	vec3 localPosition = gl_ObjectRayOriginEXT + gl_ObjectRayDirectionEXT * gl_HitTEXT;
	
	const float waterWavesStrength = pow(renderer.waterWaves, 2);
	
	if (gl_HitKindEXT == WATER_INTERSECTION_ABOVE) {
		// Above water
		
		vec3 reflection = vec3(0);
		vec3 refraction = vec3(0);
		
		vec3 surfaceNormal = vec3(0,1,0);
		if (waterWavesStrength > 0) APPLY_NORMAL_BUMP_NOISE(WaterWaves, worldPosition, surfaceNormal, waterWavesStrength * 0.05 * smoothstep(1000, 100, gl_HitTEXT))
		float fresnel = Fresnel((camera.viewMatrix * vec4(worldPosition, 1)).xyz, normalize(WORLD2VIEWNORMAL * surfaceNormal), WATER_IOR);
		
		// Reflection on top of water surface
		vec3 reflectDir = normalize(reflect(gl_WorldRayDirectionEXT, surfaceNormal));
		RAY_RECURSION_PUSH
			ray.color.rgb = vec3(0);
			traceRayEXT(tlas, 0, RENDERABLE_STANDARD_EXCEPT_WATER, 0/*rayType*/, 0/*nbRayTypes*/, 0/*missIndex*/, worldPosition, cam.zNear, reflectDir, cam.zFar, 0);
		RAY_RECURSION_POP
		reflection = ray.color.rgb;
		
		// See through water (refraction)
		vec3 rayDirection = gl_WorldRayDirectionEXT;
		if (Refract(rayDirection, surfaceNormal, WATER_IOR)) {
			RAY_RECURSION_PUSH
				RAY_UNDERWATER_PUSH
					ray.color.rgb = vec3(0);
					traceRayEXT(tlas, 0, RENDERABLE_STANDARD_EXCEPT_WATER, 0/*rayType*/, 0/*nbRayTypes*/, 0/*missIndex*/, worldPosition, camera.zNear, rayDirection, WATER_MAX_LIGHT_DEPTH, 0);
				RAY_UNDERWATER_POP
			RAY_RECURSION_POP
			if (ray.hitDistance == -1) {
				ray.hitDistance = WATER_MAX_LIGHT_DEPTH;
				ray.color.rgb = vec3(0);
			}
			refraction = ray.color.rgb * (1-clamp(ray.hitDistance / WATER_MAX_LIGHT_DEPTH, 0, 1));
		}
		
		ray.hitDistance = gl_HitTEXT;
		ray.color.rgb = reflection * fresnel + refraction * (1-fresnel);
		ray.normal = surfaceNormal;
		
		SetHitWater();
		
	} else {
		// Underwater
		
		if (dot(gl_WorldRayDirectionEXT, vec3(0,1,0)) > 0) {
			// Looking up towards surface

			float underwaterDepth = AABB_MAX.y - localPosition.y;
			float distanceToSurface = clamp(min(underwaterDepth/max(0.001, dot(gl_WorldRayDirectionEXT, vec3(0,1,0))), camera.zFar), camera.zNear, WATER_MAX_LIGHT_DEPTH);
			vec3 surfaceNormal = vec3(0,-1,0);
			vec3 wavePosition = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * distanceToSurface;
			if (waterWavesStrength > 0) APPLY_NORMAL_BUMP_NOISE(WaterWaves, wavePosition, surfaceNormal, waterWavesStrength * 0.05)
			
			// See through water (underwater looking up, possibly at surface)
			vec3 rayPosition = gl_WorldRayOriginEXT;
			vec3 rayDirection = gl_WorldRayDirectionEXT;
			RAY_RECURSION_PUSH
				RAY_UNDERWATER_PUSH
					ray.color.rgb = vec3(0);
					traceRayEXT(tlas, 0, RENDERABLE_STANDARD_EXCEPT_WATER, 0/*rayType*/, 0/*nbRayTypes*/, 0/*missIndex*/, rayPosition, camera.zNear, rayDirection, distanceToSurface, 0);
				RAY_UNDERWATER_POP
			RAY_RECURSION_POP
			
			if (ray.hitDistance == -1) {
				// Surface refraction seen from underwater
				rayPosition += rayDirection * distanceToSurface;
				float maxRayDistance = camera.zFar;
				if (!Refract(rayDirection, surfaceNormal, 1.0 / WATER_IOR)) {
					maxRayDistance = WATER_MAX_LIGHT_DEPTH;
				}
				RAY_RECURSION_PUSH
					ray.color.rgb = vec3(0);
					traceRayEXT(tlas, 0, RENDERABLE_STANDARD_EXCEPT_WATER, 0/*rayType*/, 0/*nbRayTypes*/, 0/*missIndex*/, rayPosition, camera.zNear, rayDirection, maxRayDistance, 0);
				RAY_RECURSION_POP
				if (maxRayDistance == WATER_MAX_LIGHT_DEPTH) {
					if (ray.hitDistance == -1) {
						ray.hitDistance = WATER_MAX_LIGHT_DEPTH;
					}
					ray.color.rgb *= pow(1.0 - clamp(ray.hitDistance / WATER_MAX_LIGHT_DEPTH, 0, 1), 2);
				}
				ray.hitDistance = distanceToSurface;
				ray.normal = vec3(0,-1,0);
			}
			ray.color.rgb *= WATER_COLOR * pow(1.0 - clamp(ray.hitDistance / WATER_MAX_LIGHT_DEPTH, 0, 1), 2);
		
		} else {
			// See through water (underwater looking down)
			
			vec3 rayPosition = gl_WorldRayOriginEXT;
			vec3 rayDirection = gl_WorldRayDirectionEXT;
			RAY_RECURSION_PUSH
				RAY_UNDERWATER_PUSH
					ray.color.rgb = vec3(0);
					traceRayEXT(tlas, 0, RENDERABLE_STANDARD_EXCEPT_WATER, 0/*rayType*/, 0/*nbRayTypes*/, 0/*missIndex*/, rayPosition, camera.zNear, rayDirection, WATER_MAX_LIGHT_DEPTH, 0);
				RAY_UNDERWATER_POP
			RAY_RECURSION_POP
			if (ray.hitDistance == -1) {
				ray.hitDistance = WATER_MAX_LIGHT_DEPTH;
				ray.color = vec4(0,0,0,1);
				ray.normal = vec3(0,1,0);
				SetHitWater();
			} else {
				ray.color.rgb *= WATER_COLOR * pow(1-clamp(ray.hitDistance / WATER_MAX_LIGHT_DEPTH, 0, 1), 2);
			}
			
		}
		
		ApplyUnderwaterFog();
	}
}
