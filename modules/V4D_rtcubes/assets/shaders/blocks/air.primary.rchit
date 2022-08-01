#version 460 core

#define _DEBUG
#define SHADER_RCHIT
#define SHADER_SUBPASS_0
#define SHADER_OFFSET_0

#ifndef _RTCUBES_SHADER_BASE_INCLUDED_
#define _RTCUBES_SHADER_BASE_INCLUDED_

#extension GL_EXT_ray_tracing : enable

#ifndef _SHADER_BASE_INCLUDED_
#define _SHADER_BASE_INCLUDED_

#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_nonuniform_qualifier : require

#ifdef __cplusplus

// Vulkan4D Core Header

#define V4D_VERSION_MAJOR 0
#define V4D_VERSION_MINOR 0
#define V4D_VERSION_PATCH 0

// V4D Core class (Compiled into v4d.dll)
# include "Core.h"

#endif // __cplusplus
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
	#define BUFFER_REFERENCE_STRUCT(align) layout(buffer_reference, std430, buffer_reference_align = align) buffer
	#define BUFFER_REFERENCE_STRUCT_READONLY(align) layout(buffer_reference, std430, buffer_reference_align = align) buffer readonly
	#define BUFFER_REFERENCE_STRUCT_WRITEONLY(align) layout(buffer_reference, std430, buffer_reference_align = align) buffer writeonly
	#define BUFFER_REFERENCE_ADDR(type) type
	
#endif
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
#define SET0_BINDING_TEXTURES 13

#define SET1_BINDING_INOUT_IMG_RESOLVED 0
#define SET1_BINDING_IN_IMG_OVERLAY 1
#define SET1_BINDING_IMG_THUMBNAIL 2
#define SET1_BINDING_LUMINANCE_BUFFER 3

// up to 32 render options
#define RENDER_OPTION_TXAA (1u<< 0)
#define RENDER_OPTION_DLSS (1u<< 1)

// up to 32 debug options
// #define RENDER_DEBUG_xxx (1u<< 0)

// Debug view modes
#define RENDER_DEBUG_VIEWMODE_NONE 0

// Configuration
#define TXAA_SAMPLES 16
#define MAX_CAMERAS 64
#define MAX_TEXTURE_BINDINGS 65535

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
	aligned_uint32_t debug;
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

BUFFER_REFERENCE_STRUCT_READONLY(16) VoxelData {
	aligned_float32_t aabb[6];
	aligned_uint64_t extra; // Arbitrary data defined per-shader
};
STATIC_ASSERT_ALIGNED16_SIZE(VoxelData, 32)

BUFFER_REFERENCE_STRUCT_READONLY(16) VoxelGeometryData {
	aligned_uint8_t sbtHandle[32];
	BUFFER_REFERENCE_ADDR(VoxelData) voxels;
	aligned_uint64_t extra[3]; // Arbitrary data defined per-shader
};
STATIC_ASSERT_ALIGNED16_SIZE(VoxelGeometryData, 64)

BUFFER_REFERENCE_STRUCT_READONLY(16) RenderableInstanceData {
	BUFFER_REFERENCE_ADDR(VoxelGeometryData) geometries;
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
	aligned_uint32_t voxelIndex;
	aligned_f32vec3 worldSpacePosition; // MUST COMPENSATE FOR ORIGIN RESET
	aligned_float32_t hitDistance;
	aligned_f32vec4 color;
};
STATIC_ASSERT_ALIGNED16_SIZE(AimBuffer, 64)

#ifdef __cplusplus
	}
#endif

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
	float lumRgbTotal = camera.luminance.r + camera.luminance.g + camera.luminance.b;
	float exposure = lumRgbTotal > 0 ? camera.luminance.a / lumRgbTotal : 1;
	color.rgb = vec3(1.0) - exp(-color.rgb * clamp(exposure, 0.0001, 10.0));
	
	// Contrast / Brightness
	const float contrast = 1.05;
	const float brightness = 1.2;
	if (contrast != 1.0 || brightness != 1.0) {
		color.rgb = mix(vec3(0.5), color.rgb, contrast) * brightness;
	}
	
	// Gamma correction
	float gammaCorrection = 2.0;
	color.rgb = pow(color.rgb, vec3(1.0 / gammaCorrection));
	
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
#ifdef __cplusplus
#endif
#ifdef __cplusplus

// Vulkan4D Core Header

#define V4D_VERSION_MAJOR 0
#define V4D_VERSION_MINOR 0
#define V4D_VERSION_PATCH 0

// V4D Core class (Compiled into v4d.dll)
# include "Core.h"

#endif // __cplusplus
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
	#define BUFFER_REFERENCE_STRUCT(align) layout(buffer_reference, std430, buffer_reference_align = align) buffer
	#define BUFFER_REFERENCE_STRUCT_READONLY(align) layout(buffer_reference, std430, buffer_reference_align = align) buffer readonly
	#define BUFFER_REFERENCE_STRUCT_WRITEONLY(align) layout(buffer_reference, std430, buffer_reference_align = align) buffer writeonly
	#define BUFFER_REFERENCE_ADDR(type) type
	
#endif

/////////////////////

#define RENDERABLE_SELF (1u<< 0)
#define RENDERABLE_SOLID (1u<< 1)
#define RENDERABLE_WATER (1u<< 2)
#define RENDERABLE_MOB (1u<< 3)
#define RENDERABLE_CLUTTER (1u<< 4)
// #define RENDERABLE_ (1u<< 5)
// #define RENDERABLE_ (1u<< 6)
// #define RENDERABLE_ (1u<< 7)
#define RENDERABLE_ALL 0xff

#define SET1_BINDING_TLAS 0
#define SET1_BINDING_RENDERER_DATA 1

#define SBT_HITGROUPS_PER_GEOMETRY 1


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

struct RendererData {
	BUFFER_REFERENCE_ADDR(RenderableInstanceData) renderableInstances;
	BUFFER_REFERENCE_ADDR(TLASInstance) tlasInstances;
	BUFFER_REFERENCE_ADDR(AimBuffer) aim;
	aligned_float64_t timestamp;
	aligned_uint8_t debugChunks;
	aligned_f32vec3 skyLightColor;
	aligned_f32vec3 torchLightColor;
};

const float EPSILON = 0.00001;

layout(push_constant) uniform PushConstant {
	RayTracingPushConstant pushConstant;
};

layout(set = 1, binding = SET1_BINDING_RENDERER_DATA) buffer RendererDataBuffer { RendererData renderer; };

#define RAY_PAYLOAD_PRIMARY 0
struct RayPayload {
	vec4 color;
	vec3 localPosition;
	float hitDistance;
	vec3 normal; // world-space
	float reflection;
	uvec2 index;
	int tlasInstanceIndex;
	float ior; // Index Of Refraction
	float totalDistanceFromEye;
	float t2;
	// float _unused_1;
	// float _unused_2;
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
	#define VOXEL GEOMETRY.voxels[gl_PrimitiveID]
	#define AABB_MIN vec3(VOXEL.aabb[0], VOXEL.aabb[1], VOXEL.aabb[2])
	#define AABB_MAX vec3(VOXEL.aabb[3], VOXEL.aabb[4], VOXEL.aabb[5])
	#define AABB_CENTER ((AABB_MIN + AABB_MAX) * 0.5)
	#define AABB_CENTER_INT ivec3(round(AABB_CENTER))
	#define MODELVIEW (camera.viewMatrix * mat4(gl_ObjectToWorldEXT))
	#define MVP (camera.projectionMatrix * MODELVIEW)
	#define MVP_AA (camera.projectionMatrixWithTXAA * MODELVIEW)
	#define MVP_HISTORY (camera.projectionMatrix * MODELVIEW_HISTORY)
	#define AIM_GEOMETRY_ID (uint32_t(gl_InstanceCustomIndexEXT) << 12) | uint32_t(gl_GeometryIndexEXT)
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
#endif

#ifdef SHADER_RCHIT
	vec3 DoubleSidedNormals(in vec3 worldSpaceNormal) {
		return -sign(dot(worldSpaceNormal, gl_WorldRayDirectionEXT)) * worldSpaceNormal;
	}
	vec3 DoubleSidedNormals(in vec3 worldSpaceNormal, in float bias) {
		return -sign(dot(worldSpaceNormal, gl_WorldRayDirectionEXT)-bias) * worldSpaceNormal;
	}
	// Compute normal for a box (this method works for boxes with arbitrary width/height/depth)
	#define CLOSEST_HIT_BOX_INTERSECTION_COMPUTE_NORMAL \
		vec3 BOX_SIZE = AABB_MAX - AABB_MIN; uint BOX_FACE; vec3 BOX_NORMAL; vec2 BOX_UV; vec2 BOX_COORD; {\
			const float threshold = EPSILON * ray.totalDistanceFromEye * 0.05;\
			const vec3 absMin = abs(ray.localPosition.xyz - (gl_HitKindEXT == BOX_INTERSECTION_KIND_OUTSIDE_FACE? AABB_MIN.xyz : AABB_MAX.xyz));\
			const vec3 absMax = abs(ray.localPosition.xyz - (gl_HitKindEXT == BOX_INTERSECTION_KIND_OUTSIDE_FACE? AABB_MAX.xyz : AABB_MIN.xyz));\
				 if (absMin.x < threshold) {BOX_FACE = 0; BOX_NORMAL.xyz = vec3(-1, 0, 0); BOX_COORD = vec2(ray.localPosition.zy) * vec2(+1,-1) - vec2(0.5); BOX_UV = BOX_COORD / BOX_SIZE.zy;}\
			else if (absMin.y < threshold) {BOX_FACE = 1; BOX_NORMAL.xyz = vec3( 0,-1, 0); BOX_COORD = vec2(ray.localPosition.xz) * vec2(-1,-1) - vec2(0.5); BOX_UV = BOX_COORD / BOX_SIZE.xz;}\
			else if (absMin.z < threshold) {BOX_FACE = 2; BOX_NORMAL.xyz = vec3( 0, 0,-1); BOX_COORD = vec2(ray.localPosition.xy) * vec2(-1,-1) - vec2(0.5); BOX_UV = BOX_COORD / BOX_SIZE.xy;}\
			else if (absMax.x < threshold) {BOX_FACE = 3; BOX_NORMAL.xyz = vec3( 1, 0, 0); BOX_COORD = vec2(ray.localPosition.zy) * vec2(-1,-1) - vec2(0.5); BOX_UV = BOX_COORD / BOX_SIZE.zy;}\
			else if (absMax.y < threshold) {BOX_FACE = 4; BOX_NORMAL.xyz = vec3( 0, 1, 0); BOX_COORD = vec2(ray.localPosition.xz) * vec2(+1,+1) - vec2(0.5); BOX_UV = BOX_COORD / BOX_SIZE.xz;}\
			else if (absMax.z < threshold) {BOX_FACE = 5; BOX_NORMAL.xyz = vec3( 0, 0, 1); BOX_COORD = vec2(ray.localPosition.xy) * vec2(+1,-1) - vec2(0.5); BOX_UV = BOX_COORD / BOX_SIZE.xy;}\
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
	ray.index = uvec2(AIM_GEOMETRY_ID, gl_PrimitiveID);\
	ray.tlasInstanceIndex = gl_InstanceID;\
	ray.color = vec4(vec3(0.5), 1.0);\
	ray.reflection = 0;\
	ray.ior = 1.45;\
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
	ray.t2 = max(ray.t2, ray.hitDistance);\
}

#define CLOSEST_HIT_BOX_AIM_WIREFRAME {\
	if (renderer.debugChunks != 0 && (renderer.aim.aimGeometryID >> 12) == (ray.index.x >> 12)) {\
		ray.color.rgb += 0.1;\
	}\
	if (renderer.aim.aimGeometryID == ray.index.x && renderer.aim.voxelIndex == ray.index.y) {\
		vec2 uv = abs(fract(BOX_UV) - 0.5);\
		float thickness = 0.0018 * max(1, ray.totalDistanceFromEye) / min(min(BOX_SIZE.x,BOX_SIZE.y),BOX_SIZE.z);\
		float border = smoothstep(0.5-thickness, 0.5, max(uv.x, uv.y));\
		ray.color.rgb = mix(ray.color.rgb, vec3(0), border);\
	}\
}

#define ANY_HIT_OCCLUSION_RAY(_color) {\
	occlusionRay.color.rgb *= _color.rgb;\
	occlusionRay.color.a = min(occlusionRay.color.a + _color.a, 1.0);\
	if (occlusionRay.color.a > 0.95) terminateRayEXT;\
	ignoreIntersectionEXT;\
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

#define APPLY_FRESNEL_REFLECTION {\
	ray.reflection = Fresnel((camera.viewMatrix * vec4(gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * ray.hitDistance, 1)).xyz, normalize(WORLD2VIEWNORMAL * ray.normal), ray.ior);\
}

#define APPLY_STANDARD_BLOCK_LIGHTING {\
	ivec3 facingBlockPos = AABB_CENTER_INT + ivec3(round(ray.normal));\
	uint64_t chunkID = VOXEL.extra;\
	BlockData facingBlockData = GetBlockData(chunkID, facingBlockPos);\
	if (!IsBlockValid(facingBlockData)) {\
		facingBlockData = DefaultAirBlock; /* If invalid, assume a fully-sky-lit air block */\
	}\
	if (HasBlockLighting(facingBlockData)) {\
		const int nbAdjacentSides = 18;\
		const ivec3 adjacentSides[nbAdjacentSides] = {\
			ivec3( 0, 0, 1),\
			ivec3( 0, 1, 0),\
			ivec3( 1, 0, 0),\
			ivec3( 0, 0,-1),\
			ivec3( 0,-1, 0),\
			ivec3(-1, 0, 0),\
			ivec3( 0,-1,-1),\
			ivec3( 1, 0, 1),\
			ivec3( 0, 1, 1),\
			ivec3( 1, 1, 0),\
			ivec3(-1, 0,-1),\
			ivec3(-1,-1, 0),\
			ivec3( 0,-1, 1),\
			ivec3(-1, 0, 1),\
			ivec3(-1, 1, 0),\
			ivec3( 0, 1,-1),\
			ivec3( 1, 0,-1),\
			ivec3( 1,-1, 0),\
		};\
		vec4 lighting = vec4(GetBlockLighting(facingBlockData), 1);\
		for (int i = 0; i < nbAdjacentSides; ++i) {\
			if (dot(vec3(adjacentSides[i]), ray.normal) == 0) {\
				ivec3 adjacentBlockPos = facingBlockPos + adjacentSides[i];\
				uint64_t adjacentBlockChunkID = chunkID;\
				BlockData adjacentBlock = GetBlockData(adjacentBlockChunkID, adjacentBlockPos);\
				if (!IsBlockValid(adjacentBlock)) {\
					adjacentBlock = DefaultAirBlock;\
				}\
				vec3 p = (ray.localPosition - AABB_CENTER) - vec3(adjacentSides[i]);\
				if (HasBlockLighting(adjacentBlock)) {\
					lighting += vec4(GetBlockLighting(adjacentBlock) * (1 - clamp(max(sdfCube(p, 0.6), sdfSphere(p, 0.7)), 0, 1)), 1);\
				} else {\
					lighting.a++;\
				}\
			}\
		}\
		ray.color.rgb *= clamp(lighting.rgb/lighting.a, 0, 1);\
	} else {\
		ray.color.rgb = vec3(0);\
	}\
}

#endif // _RTCUBES_SHADER_BASE_INCLUDED_
#ifdef __cplusplus
#endif
#ifdef __cplusplus

// Vulkan4D Core Header

#define V4D_VERSION_MAJOR 0
#define V4D_VERSION_MINOR 0
#define V4D_VERSION_PATCH 0

// V4D Core class (Compiled into v4d.dll)
# include "Core.h"

#endif // __cplusplus
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
	#define BUFFER_REFERENCE_STRUCT(align) layout(buffer_reference, std430, buffer_reference_align = align) buffer
	#define BUFFER_REFERENCE_STRUCT_READONLY(align) layout(buffer_reference, std430, buffer_reference_align = align) buffer readonly
	#define BUFFER_REFERENCE_STRUCT_WRITEONLY(align) layout(buffer_reference, std430, buffer_reference_align = align) buffer writeonly
	#define BUFFER_REFERENCE_ADDR(type) type
	
#endif

#define MAX_BLOCK_XZ 16
#define MAX_BLOCK_Y 512
#define MAX_BLOCK_X MAX_BLOCK_XZ
#define MAX_BLOCK_Z MAX_BLOCK_XZ
#define MAX_BLOCKS_PER_CHUNK (MAX_BLOCK_X*MAX_BLOCK_Y*MAX_BLOCK_Z)

#define MAX_SKY_LIGHT_LEVEL 15
#define MAX_TORCH_LIGHT_LEVEL 15
#define MAX_WATER_DEPTH 63

// #define FACE_PLUS_X 0x01
// #define FACE_MINUS_X 0x02
// #define FACE_PLUS_Z 0x04
// #define FACE_MINUS_Z 0x08
// #define FACE_PLUS_Y 0x10
// #define FACE_MINUS_Y 0x20
// #define FACE_PLUSMINUS_X (FACE_PLUS_X|FACE_MINUS_X)
// #define FACE_PLUSMINUS_Z (FACE_PLUS_Z|FACE_MINUS_Z)
// #define FACE_PLUSMINUS_Y (FACE_PLUS_Y|FACE_MINUS_Y)
// #define FACE_ALL_BITS 0x3f

#ifdef __cplusplus
	union BlockIndex {
		uint32_t index32;
		struct Position {
			uint32_t x : 4; // 16
			uint32_t z : 4; // 16
			uint32_t y : 9; // 512
			uint32_t _ : 15; // padding
			Position(uint32_t x, uint32_t y, uint32_t z) : x(x), z(z), y(y), _(0) {}
		} pos;
		explicit BlockIndex(uint32_t index32) : index32(index32) {
			assert(index32 < MAX_BLOCKS_PER_CHUNK);
		}
		BlockIndex(uint16_t index16, bool extension) : index32(uint32_t(index16) | (uint32_t(extension?1:0) << 16)) {}
		BlockIndex(const glm::ivec3& p) : pos(p.x, p.y, p.z) {
			assert(p.x >= 0);
			assert(p.z >= 0);
			assert(p.y >= 0);
			assert(p.x < MAX_BLOCK_X);
			assert(p.z < MAX_BLOCK_Z);
			assert(p.y < MAX_BLOCK_Y);
			assert(index32 < MAX_BLOCKS_PER_CHUNK);
		}
		operator glm::ivec3() const {
			return {pos.x,pos.y,pos.z};
		}
		glm::ivec3 Position() const {
			return {pos.x,pos.y,pos.z};
		}
		uint16_t Index16() const {
			return index32 & 0xffff;
		}
		uint32_t Index32() const {
			assert(index32 < MAX_BLOCKS_PER_CHUNK);
			return index32;
		}
		bool IndexExtension() const {
			return index32 & 0x10000;
		}
	};
#else
	#define BlockIndex(x,y,z) (uint32_t(x) | (uint32_t(z) << 4) | (uint32_t(y) << 8))
#endif
STATIC_ASSERT_SIZE(BlockIndex, 4);

// Used in ClientChunkData for GPU calculation of lighting as well as Block-Specific stuff
#ifdef __cplusplus
	union BlockData {
		struct {
			// Only for opaque non-emissive blocks (dirt, wall, stone, trunc, ...)
			uint16_t extra : 14; // extra Block-Specific data (custom structure per block type)
			uint16_t lighting : 1; // will be False
			uint16_t flag : 1; // Multi-purpose flag (see note below)
		} opaqueBlock;
		struct {
			// Only for transparent or emissive blocks (air, glass, water, torch, lava, window, leafs, ...)
			uint16_t extra : 6; // extra Block-Specific data (custom structure per block type)
			uint16_t light_torch : 4; // 0:15 level of light by torches
			uint16_t light_sky : 4; // 0:15 level of light by sky
			uint16_t lighting : 1; // will be True
			uint16_t flag : 1; // Multi-purpose flag (see note below)
		};
		/*
			The flag bit has three purposes depending on which stage/struct it's used in.
				* TerrainBlock: For Server-Side when propagating information like lighting
					flag is set to 1 to mark it for propagation, then 0 when finished. After the propagation step, all blocks should have this set to 0.
				* VisibleBlock: For transferring chunk data to GPU (and generate AABBs)
					flag is used as the 17th bit for the block index to permit chunks of 512 blocks in Y. It's then set to 1 for the next step when writing it to ClientChunkData.
				* ClientChunkData: For the GPU to consume during rendering
					flag set to 1 means its information is valid, thus a value of zero would mean we don't know what this block contains (yet) and we cannot rely on this data.
		*/
		// uint16_t _rawData;
		// BlockData() : _rawData(0) {}
		// bool operator==(const BlockData& other) const {
		// 	if (other.lighting != lighting) return false;
		// 	return lighting? extra == other.extra : opaqueBlock.extra == other.opaqueBlock.extra;
		// }
		// bool operator!=(const BlockData& other) const {
		// 	return !(*this==other);
		// }
	};
#else
	#define BlockData uint16_t
	#define GetOpaqueBlockExtra(blockData)		(blockData & 0x3fff)
	#define GetTransparentBlockExtra(blockData)	(blockData & 0x003f)
	#define GetBlockLightTorch(blockData)		((blockData >> 6) & 0xf)
	#define GetBlockLightSky(blockData)		((blockData >> 10) & 0xf)
	#define HasBlockLighting(blockData)		((blockData & 0x4000) != 0)
	#define IsBlockValid(blockData)			((blockData & 0x8000) != 0)
	#define GetBlockLighting(blockData) (min(vec3(1), renderer.skyLightColor * pow(float(GetBlockLightSky(blockData)) / MAX_SKY_LIGHT_LEVEL, 4.0) + renderer.torchLightColor * pow(float(GetBlockLightTorch(blockData)) / 15, 4.0)))
	#define DefaultAirBlock (uint16_t(0xc000) | uint16_t(MAX_SKY_LIGHT_LEVEL << 10))
#endif
STATIC_ASSERT_SIZE(BlockData, 2);

#ifdef __cplusplus
	struct VisibleBlock {
		uint16_t index16; // Stores the first 16 bits of the block Index32, and stores the 17th bit in data.flag
		BlockData data;
		explicit VisibleBlock(BlockIndex index, BlockData data) : index16(index.Index16()), data(data) {
			this->data.flag = index.IndexExtension();
		}
		explicit VisibleBlock(uint32_t index, BlockData data) : VisibleBlock(BlockIndex(index), data) {}
		operator BlockIndex() const {
			return BlockIndex(index16, data.flag);
		}
	};
	STATIC_ASSERT_SIZE(VisibleBlock, 4);

	// For chunk storage on disk and server-side processing
	struct TerrainBlock {
		union {
			uint32_t _rawData;
			struct {
				uint16_t type : 12;// maximum of 4096 different types of blocks, total, including all mods
				uint16_t opaque : 1; // bool (cannot see through and doesn't let light pass)
				uint16_t solid : 1; // bool (has collider)
				uint16_t dynamic : 1; // bool (has custom logic that must be executed at every tick)
				uint16_t emissive : 1; // bool (can emit light)
				BlockData data;
			};
		};
		TerrainBlock() : _rawData(0) {}
		TerrainBlock(uint16_t type) : type(type) {}
		bool operator==(const TerrainBlock& other) const {
			return other.type == type;
		}
		bool operator!=(const TerrainBlock& other) const {
			return !(*this==other);
		}
		operator uint32_t() const {return _rawData;}
	};
	STATIC_ASSERT_SIZE(TerrainBlock, 4);
	
	struct NetworkBlock {
		int64_t chunkID;
		BlockIndex index;
		TerrainBlock block;
		NetworkBlock(int64_t chunkID, BlockIndex index, TerrainBlock block) : chunkID(chunkID), index(index), block(block) {}
	};
	STATIC_ASSERT_SIZE(NetworkBlock, 16);
#endif

// For use by GPU for processing of lighting
BUFFER_REFERENCE_STRUCT_READONLY(16) ClientChunkData {
	// Adjacent chunkData
	aligned_VkDeviceAddress plusX;
	aligned_VkDeviceAddress minusX;
	aligned_VkDeviceAddress plusZ;
	aligned_VkDeviceAddress minusZ;
	// blocks
	BlockData blocks[MAX_BLOCKS_PER_CHUNK];
	aligned_uint16_t blockTypes[MAX_BLOCKS_PER_CHUNK];
};
STATIC_ASSERT_ALIGNED16_SIZE(ClientChunkData, 32 + 4*MAX_BLOCKS_PER_CHUNK);

#ifndef __cplusplus
	//
	BlockData GetBlockData(inout uint64_t chunkID, inout ivec3 pos) {
		ClientChunkData chunk = ClientChunkData(chunkID); // Because of a bug in AMD drivers, we cannot pass the buffer_reference as an argument to a function, instead we pass its address
		while (pos.x < 0 && chunk.minusX != 0) {
			chunk = ClientChunkData(chunk.minusX);
			pos.x += MAX_BLOCK_X;
		}
		while (pos.z < 0 && chunk.minusZ != 0) {
			chunk = ClientChunkData(chunk.minusZ);
			pos.z += MAX_BLOCK_Z;
		}
		while (pos.x >= MAX_BLOCK_X && chunk.plusX != 0) {
			chunk = ClientChunkData(chunk.plusX);
			pos.x -= MAX_BLOCK_X;
		}
		while (pos.z >= MAX_BLOCK_Z && chunk.plusZ != 0) {
			chunk = ClientChunkData(chunk.plusZ);
			pos.z -= MAX_BLOCK_Z;
		}
		chunkID = uint64_t(chunk); // Because of the AMD bug mentioned above...
		if (pos.x >= 0 && pos.y >= 0 && pos.z >= 0 && pos.x < MAX_BLOCK_X && pos.y < MAX_BLOCK_Y && pos.z < MAX_BLOCK_Z) {
			return chunk.blocks[BlockIndex(pos.x, pos.y, pos.z)];
		}
		return BlockData(0); // invalid
	}
	#if defined(SHADER_RCHIT) || defined(SHADER_RAHIT) || defined(SHADER_RINT)
		BlockData GetBlockData() {
			ivec3 pos = AABB_CENTER_INT;
			if (pos.x >= 0 && pos.y >= 0 && pos.z >= 0 && pos.x < MAX_BLOCK_X && pos.y < MAX_BLOCK_Y && pos.z < MAX_BLOCK_Z) {
				return ClientChunkData(VOXEL.extra).blocks[BlockIndex(pos.x, pos.y, pos.z)];
			}
			return BlockData(0); // invalid
		}
	#endif
	uint16_t GetBlockType(inout uint64_t chunkID, inout ivec3 pos) {
		ClientChunkData chunk = ClientChunkData(chunkID); // Because of a bug in AMD drivers, we cannot pass the buffer_reference as an argument to a function, instead we pass its address
		while (pos.x < 0 && chunk.minusX != 0) {
			chunk = ClientChunkData(chunk.minusX);
			pos.x += MAX_BLOCK_X;
		}
		while (pos.z < 0 && chunk.minusZ != 0) {
			chunk = ClientChunkData(chunk.minusZ);
			pos.z += MAX_BLOCK_Z;
		}
		while (pos.x >= MAX_BLOCK_X && chunk.plusX != 0) {
			chunk = ClientChunkData(chunk.plusX);
			pos.x -= MAX_BLOCK_X;
		}
		while (pos.z >= MAX_BLOCK_Z && chunk.plusZ != 0) {
			chunk = ClientChunkData(chunk.plusZ);
			pos.z -= MAX_BLOCK_Z;
		}
		chunkID = uint64_t(chunk); // Because of the AMD bug mentioned above...
		if (pos.x >= 0 && pos.y >= 0 && pos.z >= 0 && pos.x < MAX_BLOCK_X && pos.y < MAX_BLOCK_Y && pos.z < MAX_BLOCK_Z) {
			return chunk.blockTypes[BlockIndex(pos.x, pos.y, pos.z)];
		}
		return uint16_t(0); // invalid
	}
#endif


void main() {
	CLOSEST_HIT_BEGIN
		CLOSEST_HIT_BOX_INTERSECTION_COMPUTE_NORMAL
		
		ClientChunkData chunk = ClientChunkData(VOXEL.extra);
		ivec3 thisBlockPos = AABB_CENTER_INT;
		BlockData thisBlockData = chunk.blocks[BlockIndex(thisBlockPos.x, thisBlockPos.y, thisBlockPos.z)];
		
		ray.color = vec4(vec3(GetBlockLighting(thisBlockData)), 0.5);
		ray.ior = 0;
		
	CLOSEST_HIT_END
	CLOSEST_HIT_BOX_AIM_WIREFRAME
	
	if (gl_HitKindEXT == BOX_INTERSECTION_KIND_INSIDE_FACE) {
		ray.hitDistance = max(camera.zNear, ray.hitDistance - camera.zNear - EPSILON*20);
	}
}
