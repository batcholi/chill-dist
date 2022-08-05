#version 460 core

#extension GL_GOOGLE_cpp_style_line_directive : enable

#define _DEBUG
#define SHADER_RCHIT
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
#define RENDER_OPTION_TAA (1u<< 0)
#define RENDER_OPTION_TEMPORAL_UPSCALING (1u<< 1)
#define RENDER_OPTION_TONE_MAPPING (1u<< 2)

// Debug view modes
#define RENDER_DEBUG_VIEWMODE_NONE 0

// Configuration
#define TAA_SAMPLES 16
#define MAX_CAMERAS 64
#define MAX_TEXTURE_BINDINGS 65535

#define NB_RENDERABLE_TYPES 8 // this will NOT change, it is a limitation of VK_KHR_ray_tracing
// Renderable types
#define RENDERABLE_TYPE_GENERIC 0 // Standard geometries
#define RENDERABLE_TYPE_SELF 1 // Typically refers to the current player's avatar when in first person view, invisible for primary rays but visible for reflections and shadow rays
#define RENDERABLE_TYPE_TERRAIN 2 // Large scale static geometries, typically Terrain
#define RENDERABLE_TYPE_WATER 3 // Transparent geometries that can include other geometries within it, affecting the visual environment, typically Water
#define RENDERABLE_TYPE_CLUTTER 4 // Small geometries usually present in very large quantities, typically Rocks and Grass
#define RENDERABLE_TYPE_CLOUD 5 // Volumetric non-solid geometries that is often ray-marched into, typically Clouds, Smoke and Atmosphere
#define RENDERABLE_TYPE_LIGHT_RADIUS 6 // Special geometry that denotes a light's radius, to be point-traced for finding relevant light sources that may be shining on surfaces for direct lighting
#define RENDERABLE_TYPE_OVERLAY 7 // Geometry that is ONLY visible from primary rays and that do not cast shadows, typically holograms and debug stuff

#define RENDERABLE_MASK_GENERIC (1u<< RENDERABLE_TYPE_GENERIC)
#define RENDERABLE_MASK_SELF (1u<< RENDERABLE_TYPE_SELF)
#define RENDERABLE_MASK_TERRAIN (1u<< RENDERABLE_TYPE_TERRAIN)
#define RENDERABLE_MASK_WATER (1u<< RENDERABLE_TYPE_WATER)
#define RENDERABLE_MASK_CLUTTER (1u<< RENDERABLE_TYPE_CLUTTER)
#define RENDERABLE_MASK_CLOUD (1u<< RENDERABLE_TYPE_CLOUD)
#define RENDERABLE_MASK_LIGHT_RADIUS (1u<< RENDERABLE_TYPE_LIGHT_RADIUS)
#define RENDERABLE_MASK_OVERLAY (1u<< RENDERABLE_TYPE_OVERLAY)

#define RENDERABLE_STANDARD (0xff & ~RENDERABLE_MASK_LIGHT_RADIUS)
#define RENDERABLE_PRIMARY (RENDERABLE_STANDARD & ~RENDERABLE_MASK_SELF)
#define RENDERABLE_STANDARD_EXCEPT_WATER (RENDERABLE_STANDARD & ~RENDERABLE_MASK_WATER)
#define RENDERABLE_PRIMARY_EXCEPT_WATER (RENDERABLE_PRIMARY & ~RENDERABLE_MASK_WATER)

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
	vec4 clipSpace = mat4(camera.projectionMatrixWithTAA) * mat4(camera.viewMatrix) * vec4(worldSpacePos, 1);
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
#define SET1_BINDING_RT_PAYLOAD_IMAGE 2

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
#define RENDERER_DEBUG_MODE_REFLECTIVITY 9
#define RENDERER_DEBUG_MODE_TRANSPARENCY 10
#define RENDERER_DEBUG_MODE_AIM_RENDERABLE 11
#define RENDERER_DEBUG_MODE_AIM_GEOMETRY 12
#define RENDERER_DEBUG_MODE_AIM_PRIMITIVE 13
#define RENDERER_DEBUG_MODE_GLOBAL_ILLUMINATION 14
#define RENDERER_DEBUG_MODE_TEST 15

#define TRACE_TYPE_PRIMARY (1u<< 0)
#define TRACE_TYPE_TRANSPARENT (1u<< 1)
#define TRACE_TYPE_REFLECTION (1u<< 2)
#define TRACE_TYPE_FOG (1u<< 3)
#define TRACE_TYPE_DIRECT_LIGHTING (1u<< 4)
#define TRACE_TYPE_GI (1u<< 5)

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Structs and Buffer References -- Must use aligned_* explicit arithmetic types (or VkDeviceAddress as an uint64_t, or BUFFER_REFERENCE_ADDR(StructType))

struct RayTracingPushConstant {
	aligned_uint32_t cameraIndex;
	aligned_uint32_t outputIndex;
	aligned_uint32_t traceTypes;
	aligned_uint32_t scale; // from this rgen, we multiply by this value to get back to the full render resolution
};
STATIC_ASSERT_SIZE(RayTracingPushConstant, 16)

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
	aligned_float32_t fogStartDistance;
	aligned_float32_t fogEndDistance;
	aligned_float32_t fogPow;
	aligned_float32_t fogIntensity;
	aligned_float32_t waterLevel;
	aligned_float64_t timestamp;
	aligned_float32_t waterMaxLightDepth;
	aligned_float32_t testValue;
};
STATIC_ASSERT_ALIGNED16_SIZE(RendererData, 9*16);
#line 10 "/home/olivier/projects/chill/src/v4d/modules/V4D_rtcubes/base.glsl"
const float EPSILON = 0.00001;

layout(push_constant) uniform PushConstant {
	RayTracingPushConstant pushConstant;
};

layout(set = 1, binding = SET1_BINDING_RENDERER_DATA) buffer RendererDataBuffer { RendererData renderer; };
layout(set = 1, binding = SET1_BINDING_RT_PAYLOAD_IMAGE, r32ui) uniform uimage2D rtPayloadImage;

bool OPTION_TEXTURES = ((renderer.options & RENDERER_OPTION_TEXTURES) != 0);
bool OPTION_REFLECTIONS = ((renderer.options & RENDERER_OPTION_REFLECTIONS) != 0 && (pushConstant.traceTypes & TRACE_TYPE_REFLECTION) != 0);
bool OPTION_TRANSPARENCY = ((renderer.options & RENDERER_OPTION_TRANSPARENCY) != 0 && (pushConstant.traceTypes & TRACE_TYPE_TRANSPARENT) != 0);
bool OPTION_INDIRECT_LIGHTING = ((renderer.options & RENDERER_OPTION_INDIRECT_LIGHTING) != 0 && (pushConstant.traceTypes & TRACE_TYPE_GI) != 0);
bool OPTION_DIRECT_LIGHTING = ((renderer.options & RENDERER_OPTION_DIRECT_LIGHTING) != 0 && (pushConstant.traceTypes & TRACE_TYPE_DIRECT_LIGHTING) != 0);
bool OPTION_SOFT_SHADOWS = ((renderer.options & RENDERER_OPTION_SOFT_SHADOWS) != 0);

#define RAY_MAX_RECURSION 8

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
	#define MVP_AA (camera.projectionMatrixWithTAA * MODELVIEW)
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
	const float bias = 0.002;\
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
#define GetGi(i) renderer.globalIllumination[i]

vec3 GetSkyColor(in vec3 dir) {
	return renderer.skyLightColor * vec3(0.5,0.6,1.5);
}

// Debug Stuff
uint64_t startTime = clockARB();
#define WRITE_DEBUG_TIME {imageStore(img_debug, ivec2(gl_LaunchIDEXT.xy), vec4(Heatmap(float(double(clockARB() - startTime) / double(500000.0))), 1));}
#define traceRayEXT {if (camera.debugViewMode == RENDERER_DEBUG_MODE_TRACE_RAY_COUNT) imageStore(img_debug, ivec2(gl_LaunchIDEXT.xy), vec4(0,0,0, imageLoad(img_debug, ivec2(gl_LaunchIDEXT.xy)).a + 1));} traceRayEXT
#define DEBUG_TEST(color) {if (camera.debugViewMode == RENDERER_DEBUG_MODE_TEST) imageStore(img_debug, ivec2(gl_LaunchIDEXT.xy), color);}

#define RT_PAYLOAD_FLAGS imageLoad(rtPayloadImage, ivec2(gl_LaunchIDEXT.xy)).r
#define SET_RT_PAYLOAD_FLAGS(flags) imageStore(rtPayloadImage, ivec2(gl_LaunchIDEXT.xy), uvec4(flags));
#define SET_RT_PAYLOAD_FLAG(flag) SET_RT_PAYLOAD_FLAGS(RT_PAYLOAD_FLAGS | flag)
#define UNSET_RT_PAYLOAD_FLAG(flag) SET_RT_PAYLOAD_FLAGS(RT_PAYLOAD_FLAGS & ~flag)

// Up to 32 flags
#define RT_PAYLOAD_FLAG_UNDERWATER (1u<< 0)
#define RT_PAYLOAD_FLAG_FOG_RAY (1u<< 1)
#define RT_PAYLOAD_FLAG_GI_RAY (1u<< 2)
#define RT_PAYLOAD_FLAG_SHADOW_RAY (1u<< 3)
#define RT_PAYLOAD_FLAG_REFLECTION_RAY (1u<< 4)
#define RT_PAYLOAD_FLAG_TRANSPARENT_RAY (1u<< 5)

#define WATER_LEVEL renderer.waterLevel
#define MAX_WATER_DEPTH renderer.waterMaxLightDepth

uint seed = InitRandomSeed(InitRandomSeed(gl_LaunchIDEXT.x, gl_LaunchIDEXT.y), uint(camera.frameIndex));

#if defined(SHADER_RCHIT) || defined(SHADER_RGEN)
	layout(set = 1, binding = SET1_BINDING_TLAS) uniform accelerationStructureEXT tlas;

	void ApplyFog(in vec3 origin, in vec3 dir) {
		if ((pushConstant.traceTypes & TRACE_TYPE_FOG) != 0 && (RT_PAYLOAD_FLAGS & RT_PAYLOAD_FLAG_UNDERWATER) == 0) {
			const float dist = ray.hitDistance > 0 ? ray.hitDistance : renderer.fogEndDistance;
			const float sunIncidentAngle = dot(-renderer.sunDir, dir);
			const float distFactor = min(1, smoothstep(renderer.fogStartDistance, renderer.fogEndDistance, dist));
			const float fogStrength = pow(distFactor, renderer.fogPow) * renderer.fogIntensity;
			vec3 fogColor = mix(renderer.skyLightColor, GetSkyColor(dir), distFactor);
			
			// Rainbow
			const vec3 rainbowStart = cos(radians(vec3(41,40,39)));
			const vec3 rainbowEnd =   cos(radians(vec3(46,44,41)));
			const vec3 rainbowPeak = (rainbowStart + rainbowEnd) * 0.5;
			fogColor += renderer.skyLightColor * renderer.skyLightColor * 0.5 * smoothstep(rainbowStart, rainbowPeak, vec3(sunIncidentAngle)) * smoothstep(rainbowEnd, rainbowPeak, vec3(sunIncidentAngle));
			
			if (ray.bounces == 0 && renderer.fogSteps > 0) {
				// Volumetric Fog (with God-Rays)
				const float fogAmount = fogStrength / renderer.fogSteps;
				RayPayload originalRay = ray;
				++ray.bounces;
				SET_RT_PAYLOAD_FLAG(RT_PAYLOAD_FLAG_FOG_RAY)
				for (int i = 0; i < renderer.fogSteps; ++i) {
					float dist = RandomFloat(seed) * dist;
					vec3 worldPos = origin + dir * dist;
					ray.hitDistance = 0;
					traceRayEXT(tlas, 0, RENDERABLE_PRIMARY, 0/*rayType*/, SBT_HITGROUPS_PER_GEOMETRY/*nbRayTypes*/, 0/*missIndex*/, worldPos, 0, normalize(renderer.sunDir), camera.zFar, RAY_PAYLOAD_PRIMARY);
					if (ray.hitDistance == -1) {
						originalRay.color.rgb = mix(originalRay.color.rgb, fogColor, fogAmount);
					} else {
						originalRay.color.rgb = mix(originalRay.color.rgb, vec3(0), fogAmount);
					}
				}
				ray = originalRay;
				UNSET_RT_PAYLOAD_FLAG(RT_PAYLOAD_FLAG_FOG_RAY)
			} else {
				// Basic Fog
				ray.color.rgb = mix(ray.color.rgb, fogColor, fogStrength);
			}
		}
	}
#endif

#ifdef SHADER_RCHIT
	#define SUN_LIGHT_SOLID_ANGLE 0.01
	
	vec3 GetDirectLighting(in vec3 albedo, in float diffuseMultiplier, in float specularMultiplier, in float specularPower) {
		uint flags = RT_PAYLOAD_FLAGS;
		if ((flags & RT_PAYLOAD_FLAG_SHADOW_RAY) != 0) return vec3(0);
		if ((flags & RT_PAYLOAD_FLAG_FOG_RAY) != 0) return vec3(0);
		bool isAlreadyShadowRay = (flags & RT_PAYLOAD_FLAG_SHADOW_RAY) != 0;
		vec3 ambient = vec3(0);
		vec3 directLighting = vec3(0);
		if (!OPTION_INDIRECT_LIGHTING) {
			ambient = albedo * 0.01;
		}
		bool isUnderWater = (flags & RT_PAYLOAD_FLAG_UNDERWATER) != 0;
		const float t = dot(renderer.sunDir, vec3(0,1,0));
		float sunset = pow(1-abs(t), 4);
		// vec3 sunColor = mix(renderer.skyLightColor, vec3(1.0f,0.8f,0.5f), sunset);
		vec3 sunColor = mix(renderer.skyLightColor, vec3(0.15f,0.08f,0.03f), sunset);
		if (OPTION_DIRECT_LIGHTING) {
			if (ray.bounces < RAY_MAX_RECURSION) {// Direct Lighting (must apply this for diffuse materials only)
				float sunSolidAngle = SUN_LIGHT_SOLID_ANGLE;
				vec3 shadowRayDir = normalize(renderer.sunDir + vec3(0.0012765f));
				if (OPTION_SOFT_SHADOWS) {
					float pointRadius = sunSolidAngle * RandomFloat(seed);
					float pointAngle = RandomFloat(seed) * 2.0 * 3.1415926535;
					vec2 diskPoint = vec2(pointRadius * cos(pointAngle), pointRadius * sin(pointAngle));
					vec3 lightTangent = normalize(cross(shadowRayDir, ray.normal));
					vec3 lightBitangent = normalize(cross(lightTangent, shadowRayDir));
					shadowRayDir = normalize(shadowRayDir + diskPoint.x * lightTangent + diskPoint.y * lightBitangent);
				}
				if (dot(shadowRayDir, ray.normal) > 0.001 && dot(shadowRayDir, vec3(0,1,0)) > 0) {
					vec3 surfacePosition = ray.nextPosition + ray.normal * 0.001;
					directLighting = vec3(1);
					RayPayload originalRay = ray;
					if (!isAlreadyShadowRay) SET_RT_PAYLOAD_FLAG(RT_PAYLOAD_FLAG_SHADOW_RAY)
					++ray.bounces;
					for (;;) {
						if (dot(directLighting,directLighting) < 0.00001) {
							directLighting = vec3(0);
							break;
						}
						ray.hitDistance = 0;
						traceRayEXT(tlas, 0, RENDERABLE_STANDARD_EXCEPT_WATER, 0/*rayType*/, SBT_HITGROUPS_PER_GEOMETRY/*nbRayTypes*/, 0/*missIndex*/, surfacePosition, camera.zNear, shadowRayDir, camera.zFar, RAY_PAYLOAD_PRIMARY);
						if (ray.hitDistance == -1) {
							vec3 diffuse = albedo * sunColor * dot(renderer.sunDir, originalRay.normal) * diffuseMultiplier;
							vec3 specular = sunColor * pow(clamp(dot(originalRay.normal, normalize(shadowRayDir - gl_WorldRayDirectionEXT)), 0, 1), specularPower) * specularMultiplier;
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
					if (!isAlreadyShadowRay) UNSET_RT_PAYLOAD_FLAG(RT_PAYLOAD_FLAG_SHADOW_RAY)
					ray = originalRay;
				}
			}
		} else if (t > 0) {
			vec3 diffuse = albedo * sunColor * max(0, dot(renderer.sunDir, ray.normal)) * diffuseMultiplier;
			vec3 specular = sunColor * pow(clamp(dot(ray.normal, normalize(renderer.sunDir - gl_WorldRayDirectionEXT)), 0, 1), specularPower) * specularMultiplier;
			directLighting = diffuse + specular;
		}
		
		if (isUnderWater) {
			float falloff = pow(1-clamp((WATER_LEVEL - ray.nextPosition.y) / MAX_WATER_DEPTH, 0, 1), 4);
			return ambient + directLighting * falloff;
		} else {
			return ambient + directLighting;
		}
	}
	
	void ApplyFog() {
		ApplyFog(gl_WorldRayOriginEXT, gl_WorldRayDirectionEXT);
	}
	
#endif

#endif // _RTCUBES_SHADER_BASE_INCLUDED_
#line 2 "/home/olivier/projects/chill/src/v4d/modules/CHILL_terrain/assets/shaders/voxel.glsl"

#if defined(SHADER_RCHIT) || defined(SHADER_RINT) || defined(SHADER_RAHIT)
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
		uint64_t fill;
		uint8_t data;
		uint8_t face;
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
#line 5 "/home/olivier/projects/chill/src/v4d/modules/CHILL_terrain/assets/shaders/voxel.glsl"
#endif

#ifdef SHADER_RCHIT
	// Standard Lighting stuff
	#define GI_2_BOUNCES_MAX_DISTANCE 100
	#define GI_SAMPLES_MIN 1 // min samples to take in different direction, per gi bounce, when far
	#define GI_SAMPLES_MAX renderer.giMaxSamples // max samples to take in different direction, per gi bounce, when near (THIS COULD BE A GRAPHICS SETTING THAT GOES UP TO 100)
	#define GI_SAMPLES_FALLOFF_START_DISTANCE 5
	#define GI_SAMPLES_FALLOFF_END_DISTANCE 25
	#define GI_MIN_DISTANCE 0.1 // camera.zNear
	#define GI_MAX_DISTANCE camera.zFar
	#define MAX_ACCUMULATION 128
	#define ACCUMULATOR_MAX_FRAME_INDEX_DIFF 500

	vec3 GetAmbientVoxelLighting(in vec3 albedo, in ivec3 voxelPosInChunk, in vec3 posInVoxel) {
		uint flags = RT_PAYLOAD_FLAGS;
		bool isUnderWater = (flags & RT_PAYLOAD_FLAG_UNDERWATER) != 0;
		bool isGiRay = (flags & RT_PAYLOAD_FLAG_GI_RAY) != 0;
		bool isFogRay = (flags & RT_PAYLOAD_FLAG_FOG_RAY) != 0;
		if (isFogRay) return vec3(0);
		
		const int nbAdjacentSides = 18;
		const ivec3 adjacentSides[nbAdjacentSides] = {
			ivec3( 0, 0, 1),
			ivec3( 0, 1, 0),
			ivec3( 0, 1, 1),
			ivec3( 1, 0, 0),
			ivec3( 1, 0, 1),
			ivec3( 1, 1, 0),
			
			ivec3( 0, 0,-1),
			ivec3( 0,-1, 0),
			ivec3( 0,-1,-1),
			ivec3(-1, 0, 0),
			ivec3(-1, 0,-1),
			ivec3(-1,-1, 0),
			
			ivec3( 0,-1, 1),
			ivec3(-1, 0, 1),
			ivec3(-1, 1, 0),
			
			ivec3( 0, 1,-1),
			ivec3( 1, 0,-1),
			ivec3( 1,-1, 0),
		};
		
		uint64_t chunkID = AABB.extra;
		ivec3 facingVoxelPos = voxelPosInChunk + ivec3(round(ray.normal));
		ivec3 facingWorldPos = ivec3(round(ray.nextPosition + ray.normal * 0.5));
		uint giIndex = GetGiIndex(facingWorldPos);
		
		// Accumulate indirect lighting
		if (OPTION_INDIRECT_LIGHTING) {
			
			float dist = length((camera.viewMatrix * vec4(facingWorldPos, 1)).xyz);
			
			int maxBounces = min(RAY_MAX_RECURSION, (dist > GI_2_BOUNCES_MAX_DISTANCE? 1:2) + (isUnderWater?1:0));
			
			if (ray.bounces < maxBounces) {
				
				if (!isGiRay && !isUnderWater && renderer.testValue > 0) {// Set Current Voxel GI Darker if opaque
					float ratio = renderer.testValue; // 0.5
					if (IsVoxel(chunkID, voxelPosInChunk) && ChunkData(chunkID).voxels.fill[VoxelIndex(voxelPosInChunk.x, voxelPosInChunk.y, voxelPosInChunk.z)] == VOXEL_FULL) {
						uint giIndexCurrent = GetGiIndex(ivec3(round(ray.nextPosition - ray.normal * 0.5)));
						int lock = atomicExchange(GetGi(giIndexCurrent).lock, 1);
						if (lock == 0) {
							float accumulation = min(GetGi(giIndexCurrent).radiance.a + 1, MAX_ACCUMULATION);
							if (GetGi(giIndexCurrent).iteration != renderer.giIteration || abs(GetGi(giIndexCurrent).frameIndex - int64_t(camera.frameIndex)) > ACCUMULATOR_MAX_FRAME_INDEX_DIFF) {
								accumulation = 1;
							}
							GetGi(giIndexCurrent).frameIndex = int64_t(camera.frameIndex);
							GetGi(giIndexCurrent).iteration = renderer.giIteration;
							vec3 l = GetGi(giIndexCurrent).radiance.rgb;
							GetGi(giIndexCurrent).radiance = vec4(mix(l, l*(1-ratio), min(ratio, ratio / accumulation)), accumulation);
							GetGi(giIndexCurrent).lock = 0;
						}
					}
				}
				
				uint lock = atomicExchange(GetGi(giIndex).lock, 1);
				if (lock == 0) {
					float adjacentMixRatio = 0.2;
					float accumulation = clamp(GetGi(giIndex).radiance.a + 1, 1, MAX_ACCUMULATION);
					int sampleCount = max(1, int(round(mix(GI_SAMPLES_MAX, GI_SAMPLES_MIN, pow(smoothstep(GI_SAMPLES_FALLOFF_START_DISTANCE, GI_SAMPLES_FALLOFF_END_DISTANCE, dist), 0.5)))));
					
					if (GetGi(giIndex).iteration != renderer.giIteration || abs(GetGi(giIndex).frameIndex - int64_t(camera.frameIndex)) > ACCUMULATOR_MAX_FRAME_INDEX_DIFF) {
						accumulation = 1;
						adjacentMixRatio = 1;
						sampleCount *= 4;
					}
					
					RayPayload originalRay = ray;
					
					++ray.bounces;
					if (!isGiRay) SET_RT_PAYLOAD_FLAG(RT_PAYLOAD_FLAG_GI_RAY)
					vec3 color = vec3(0);
					for (int i = 0; i < sampleCount; ++i) {
						vec3 randomBounceDirection = normalize(originalRay.normal * 0.5 + RandomInUnitSphere(seed));
						if (i == 0 && accumulation == 1) randomBounceDirection = originalRay.normal;
						float nDotR = dot(originalRay.normal, randomBounceDirection);
						if (nDotR < 0) {
							randomBounceDirection *= -1;
							nDotR *= -1;
						}
						ray.hitDistance = 0;
						traceRayEXT(tlas, 0, RENDERABLE_PRIMARY_EXCEPT_WATER, 0/*rayType*/, SBT_HITGROUPS_PER_GEOMETRY/*nbRayTypes*/, 0/*missIndex*/, originalRay.nextPosition, GI_MIN_DISTANCE, randomBounceDirection, GI_MAX_DISTANCE, RAY_PAYLOAD_PRIMARY);
						if (ray.hitDistance == -1 && isUnderWater) {
							float falloff = pow(1-clamp((WATER_LEVEL - ray.nextPosition.y) / MAX_WATER_DEPTH, 0, 1), 4);
							ray.color.rgb *= falloff;
						}
						if (ray.hitDistance > 0) {
							ray.color.rgb *= smoothstep(GI_SAMPLES_FALLOFF_END_DISTANCE, GI_SAMPLES_FALLOFF_START_DISTANCE, ray.hitDistance);
						}
						color += ray.color.rgb * nDotR / sampleCount;
					}
					if (!isGiRay) UNSET_RT_PAYLOAD_FLAG(RT_PAYLOAD_FLAG_GI_RAY)
					
					// Protect the GI buffer from NaN values
					if (isnan(color.r) || isnan(color.g) || isnan(color.b)) {
						color = vec3(0);
					}
					
					ray = originalRay;
					
					if (!isGiRay) {
						GetGi(giIndex).frameIndex = int64_t(camera.frameIndex);
						GetGi(giIndex).iteration = renderer.giIteration;
						vec3 l = GetGi(giIndex).radiance.rgb;
						l = mix(l, color, clamp(1/accumulation, 0, 1));
						GetGi(giIndex).radiance = vec4(l, accumulation);
						GetGi(giIndex).lock = 0;
					
						for (int i = 0; i < nbAdjacentSides; ++i) {
							if (abs(dot(vec3(adjacentSides[i]), ray.normal)) < 0.2) {
								uint adjacentGiIndex = GetGiIndex(facingWorldPos + adjacentSides[i]);
								int lock = atomicExchange(GetGi(adjacentGiIndex).lock, 1);
								if (lock == 0) {
									float accumulation = min(GetGi(adjacentGiIndex).radiance.a + 1, MAX_ACCUMULATION);
									if (GetGi(adjacentGiIndex).iteration != renderer.giIteration || abs(GetGi(adjacentGiIndex).frameIndex - int64_t(camera.frameIndex)) > ACCUMULATOR_MAX_FRAME_INDEX_DIFF) {
										accumulation = 1;
										adjacentMixRatio = 1;
									}
									GetGi(adjacentGiIndex).frameIndex = int64_t(camera.frameIndex);
									GetGi(adjacentGiIndex).iteration = renderer.giIteration;
									vec3 l = GetGi(adjacentGiIndex).radiance.rgb;
									l = mix(l, color, clamp(adjacentMixRatio / accumulation, 0, 1));
									GetGi(adjacentGiIndex).radiance = vec4(l, accumulation);
									GetGi(adjacentGiIndex).lock = 0;
								}
							}
						}
						
					} else {
						ray.color.rgb += color;
						GetGi(giIndex).lock = 0;
					}
				}
			}
		}
		
		// Apply indirect lighting
		if (OPTION_INDIRECT_LIGHTING && !isGiRay) {
			ivec3 facingVoxelPos = voxelPosInChunk + ivec3(round(ray.normal));
			vec4 lighting;
			if (GetGi(giIndex).iteration == renderer.giIteration && abs(GetGi(giIndex).frameIndex - int64_t(camera.frameIndex)) < ACCUMULATOR_MAX_FRAME_INDEX_DIFF) {
				lighting = vec4(GetGi(giIndex).radiance.rgb, 1);
			} else {
				lighting = vec4(0);
			}
			for (int i = 0; i < nbAdjacentSides; ++i) {
				if (dot(vec3(adjacentSides[i]), ray.normal) == 0) {
					
					if (!isUnderWater) {
						if (abs(adjacentSides[i].x) + abs(adjacentSides[i].y) + abs(adjacentSides[i].z) == 2) {
							uint diagonalsOccluded = 0;
							if (adjacentSides[i].x != 0) {
								ivec3 pos = facingVoxelPos + adjacentSides[i] - ivec3(adjacentSides[i].x, 0, 0);
								uint64_t adjacentVoxelChunkID = chunkID;
								if (IsVoxel(adjacentVoxelChunkID, pos) && ChunkData(adjacentVoxelChunkID).voxels.fill[VoxelIndex(pos.x, pos.y, pos.z)] == VOXEL_FULL) {
									diagonalsOccluded++;
								}
							}
							if (adjacentSides[i].y != 0) {
								ivec3 pos = facingVoxelPos + adjacentSides[i] - ivec3(0, adjacentSides[i].y, 0);
								uint64_t adjacentVoxelChunkID = chunkID;
								if (IsVoxel(adjacentVoxelChunkID, pos) && ChunkData(adjacentVoxelChunkID).voxels.fill[VoxelIndex(pos.x, pos.y, pos.z)] == VOXEL_FULL) {
									diagonalsOccluded++;
								}
							}
							if (adjacentSides[i].z != 0) {
								ivec3 pos = facingVoxelPos + adjacentSides[i] - ivec3(0, 0, adjacentSides[i].z);
								uint64_t adjacentVoxelChunkID = chunkID;
								if (IsVoxel(adjacentVoxelChunkID, pos) && ChunkData(adjacentVoxelChunkID).voxels.fill[VoxelIndex(pos.x, pos.y, pos.z)] == VOXEL_FULL) {
									diagonalsOccluded++;
								}
							}
							if (diagonalsOccluded == 2) {
								lighting.a++;
								continue;
							}
						}
					}
					
					uint adjacentGiIndex = GetGiIndex(facingWorldPos + adjacentSides[i]);
					vec3 p = posInVoxel - vec3(adjacentSides[i]);
					if (GetGi(adjacentGiIndex).iteration == renderer.giIteration && abs(GetGi(adjacentGiIndex).frameIndex - int64_t(camera.frameIndex)) < ACCUMULATOR_MAX_FRAME_INDEX_DIFF) {
						lighting += vec4(GetGi(adjacentGiIndex).radiance.rgb * (1 - clamp(sdfSphere(p, 0.667), 0, 1)), 1);
					}
				}
			}
			return albedo * lighting.rgb / max(1, lighting.a);
		}
		return vec3(0);
	}

#endif

/////////////////////////////////////////////////////////////

hitAttributeEXT hit {
	VOXEL_INDEX_TYPE voxelIndex;
	uint8_t normalIndex;
};


#line 382 "/home/olivier/projects/chill/src/v4d/modules/CHILL_terrain/assets/shaders/voxel.glsl"

void main() {
	CLOSEST_HIT_BEGIN
	
		uint flags = RT_PAYLOAD_FLAGS;
		bool isUnderwater = (flags & RT_PAYLOAD_FLAG_UNDERWATER) != 0;
	
		if ((flags & RT_PAYLOAD_FLAG_FOG_RAY) != 0) {
			return;
		}
		
		if ((flags & RT_PAYLOAD_FLAG_GI_RAY) != 0) {
			surface.normal = -gl_WorldRayDirectionEXT;
		}
		
		if (!OPTION_SOFT_SHADOWS && (flags & RT_PAYLOAD_FLAG_SHADOW_RAY) != 0) {
			return;
		}
		
		surface.normal = BOX_NORMAL_DIRS[normalIndex];
		
		if (AABB.extra == 0) return;
		ChunkVoxelData voxelData = ChunkData(AABB.extra).voxels;
		if (uint64_t(voxelData) == 0) return;
		
		const ivec3 iPos = VoxelIndex_iPos(voxelIndex);
		surface.posInVoxel = ray.localPosition - vec3(voxelData.aabbOffset + iPos) * voxelData.voxelSize;
		// voxelData.data[voxelIndex]
		// voxelData.fill[voxelIndex]
		
		switch (int(normalIndex)) {
			case 0 : surface.uv = vec2(surface.posInVoxel.zy) * vec2(+1,-1) - vec2(VOXEL_GRID_OFFSET); break;
			case 1 : surface.uv = vec2(surface.posInVoxel.xz) * vec2(-1,-1) - vec2(VOXEL_GRID_OFFSET); break;
			case 2 : surface.uv = vec2(surface.posInVoxel.xy) * vec2(-1,-1) - vec2(VOXEL_GRID_OFFSET); break;
			case 3 : surface.uv = vec2(surface.posInVoxel.zy) * vec2(-1,-1) - vec2(VOXEL_GRID_OFFSET); break;
			case 4 : surface.uv = vec2(surface.posInVoxel.xz) * vec2(+1,+1) - vec2(VOXEL_GRID_OFFSET); break;
			case 5 : surface.uv = vec2(surface.posInVoxel.xy) * vec2(+1,-1) - vec2(VOXEL_GRID_OFFSET); break;
		}
		
		// Prapare Surface
		surface.color = ray.color;
		surface.emission = vec3(0);
		surface.metallic = 0;
		surface.ior = 1.45;
		surface.diffuse = 1.0;
		surface.specular = 0.2;
		surface.face = normalIndex;
		surface.fill = voxelData.fill[voxelIndex];
		surface.data = voxelData.data[voxelIndex];
		
		// Execute Surface Callable
		if (OPTION_TEXTURES) {
			executeCallableEXT(voxelData.type[voxelIndex], VOXEL_SURFACE_CALLABLE);
		}
		
		// Normal
		ray.normal = normalize(surface.normal);
		
		// Opacity
		ray.color.a = clamp(surface.color.a, 0.05, 1);
		
		if ((flags & RT_PAYLOAD_FLAG_SHADOW_RAY) != 0) {
			return;
		}
		
		// Fresnel Reflection
		if (surface.metallic < 0) {
			ApplyFresnelReflection(surface.ior);
			ray.reflection = pow(ray.reflection, -1/surface.metallic);
		}
		
		// Metallic
		if (surface.metallic > 0) {
			ray.reflection = surface.metallic;
		}
		
		// Lighting and Albedo
		vec3 lighting = GetDirectLighting(surface.color.rgb, surface.diffuse, surface.specular, 10/*specularPower*/);
		lighting += GetAmbientVoxelLighting(surface.color.rgb, iPos, surface.posInVoxel);
		ray.color.rgb = mix(lighting, surface.color.rgb, clamp(surface.metallic, 0, 1));
	
		// Emission
		ray.color.rgb += surface.emission;
		
		if ((flags & RT_PAYLOAD_FLAG_GI_RAY) != 0) {
			return;
		}
	
		// Overlays
		if (ray.bounces == 0) {
			
			// Aim Wireframe
			if (renderer.aim.aimGeometryID == ray.index.x && renderer.aim.aabbIndex == ray.index.y) {
				if (ivec3(round(renderer.aim.localPosition - renderer.aim.worldSpaceHitNormal*0.01)) == ivec3(round(ray.localPosition - ray.normal * 0.01))) {
					vec2 coord = abs(fract(surface.uv) - 0.5);
					float thickness = renderer.wireframeThickness * camera.renderScale * max(1, ray.hitDistance);
					float border = step(0.5-thickness, max(coord.x, coord.y));
					ray.color = vec4(mix(ray.color.rgb, renderer.wireframeColor.rgb, border * renderer.wireframeColor.a), ray.color.a);
				}
			}
			
			const ivec2 imgCoords = ivec2(gl_LaunchIDEXT.xy);
			if (camera.debugViewMode == RENDERER_DEBUG_MODE_UVS) {
				imageStore(img_debug, imgCoords, vec4(surface.uv, 0, 1));
			}
		
		}
		
	CLOSEST_HIT_END
}
