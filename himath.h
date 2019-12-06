#ifndef HIMATH_H
#define HIMATH_H

#ifndef __cpluspllus
#include <stdbool.h>
#endif //__cpluspllus

#define HIMATH_PI 3.141592f

#ifdef __CUDACC__
#define HIMATH_ATTRIB __host__ __device__
#else
#define HIMATH_ATTRIB
#endif //__CUDACC__

HIMATH_ATTRIB bool almost_equal(float a, float b);
HIMATH_ATTRIB float radtodeg(float rad);
HIMATH_ATTRIB float degtorad(float deg);

typedef struct IVec2_
{
    int x, y;
} IVec2;

#ifdef __cplusplus
HIMATH_ATTRIB IVec2 operator-(IVec2 v);
HIMATH_ATTRIB IVec2 operator+(IVec2 a, IVec2 b);
HIMATH_ATTRIB IVec2 operator-(IVec2 a, IVec2 b);
HIMATH_ATTRIB IVec2 operator*(IVec2 a, IVec2 b);
HIMATH_ATTRIB IVec2 operator/(IVec2 a, IVec2 b);
HIMATH_ATTRIB IVec2& operator+=(IVec2& a, IVec2 b);
HIMATH_ATTRIB IVec2& operator-=(IVec2& a, IVec2 b);
HIMATH_ATTRIB IVec2& operator*=(IVec2& a, IVec2 b);
HIMATH_ATTRIB IVec2& operator/=(IVec2& a, IVec2 b);
#endif //__cplusplus
HIMATH_ATTRIB IVec2 ivec2_negate(IVec2 v);
HIMATH_ATTRIB IVec2 ivec2_add(IVec2 a, IVec2 b);
HIMATH_ATTRIB IVec2 ivec2_sub(IVec2 a, IVec2 b);
HIMATH_ATTRIB IVec2 ivec2_mul(IVec2 a, IVec2 b);
HIMATH_ATTRIB IVec2 ivec2_div(IVec2 a, IVec2 b);

typedef struct FVec2_
{
    float x, y;
} FVec2;

#ifdef __cplusplus

HIMATH_ATTRIB FVec2 operator-(FVec2 v);
HIMATH_ATTRIB FVec2 operator+(FVec2 a, FVec2 b);
HIMATH_ATTRIB FVec2 operator-(FVec2 a, FVec2 b);
HIMATH_ATTRIB FVec2 operator*(FVec2 a, FVec2 b);
HIMATH_ATTRIB FVec2 operator/(FVec2 a, FVec2 b);
HIMATH_ATTRIB FVec2& operator+=(FVec2& a, FVec2 b);
HIMATH_ATTRIB FVec2& operator-=(FVec2& a, FVec2 b);
HIMATH_ATTRIB FVec2& operator*=(FVec2& a, FVec2 b);
HIMATH_ATTRIB FVec2& operator/=(FVec2& a, FVec2 b);

#endif //__cplusplus

HIMATH_ATTRIB FVec2 fvec2_negate(FVec2 v);
HIMATH_ATTRIB FVec2 fvec2_add(FVec2 a, FVec2 b);
HIMATH_ATTRIB FVec2 fvec2_sub(FVec2 a, FVec2 b);
HIMATH_ATTRIB FVec2 fvec2_mul(FVec2 a, FVec2 b);
HIMATH_ATTRIB FVec2 fvec2_div(FVec2 a, FVec2 b);

typedef struct FVec3_
{
    float x, y, z;
} FVec3;

#ifdef __cplusplus
HIMATH_ATTRIB FVec3 operator-(FVec3 v);
HIMATH_ATTRIB FVec3 operator+(FVec3 a, FVec3 b);
HIMATH_ATTRIB FVec3 operator-(FVec3 a, FVec3 b);
HIMATH_ATTRIB FVec3 operator*(FVec3 a, FVec3 b);
HIMATH_ATTRIB FVec3 operator/(FVec3 a, FVec3 b);
HIMATH_ATTRIB FVec3 operator*(FVec3 v, float s);
HIMATH_ATTRIB FVec3 operator/(FVec3 v, float s);
HIMATH_ATTRIB FVec3& operator+=(FVec3& a, FVec3 b);
HIMATH_ATTRIB FVec3& operator-=(FVec3& a, FVec3 b);
HIMATH_ATTRIB FVec3& operator*=(FVec3& a, FVec3 b);
HIMATH_ATTRIB FVec3& operator/=(FVec3& a, FVec3 b);
#endif //__cplusplus

HIMATH_ATTRIB FVec3 fvec3_negate(FVec3 v);
HIMATH_ATTRIB FVec3 fvec3_add(FVec3 a, FVec3 b);
HIMATH_ATTRIB FVec3 fvec3_sub(FVec3 a, FVec3 b);
HIMATH_ATTRIB FVec3 fvec3_mul(FVec3 a, FVec3 b);
HIMATH_ATTRIB FVec3 fvec3_div(FVec3 a, FVec3 b);
HIMATH_ATTRIB FVec3 fvec3_mulf(FVec3 v, float s);
HIMATH_ATTRIB FVec3 fvec3_divf(FVec3 v, float s);
HIMATH_ATTRIB FVec3 fvec3_cross(FVec3 a, FVec3 b);
HIMATH_ATTRIB float fvec3_dot(FVec3 a, FVec3 b);
HIMATH_ATTRIB float fvec3_length_sq(FVec3 v);
HIMATH_ATTRIB float fvec3_length(FVec3 v);
HIMATH_ATTRIB FVec3 fvec3_normalize(FVec3 v);

typedef struct FVec4_
{
    float x, y, z, w;
} FVec4;

typedef union Mat4_
{
    float m[16];
    float mm[4][4];
} Mat4;

#ifdef __cplusplus
HIMATH_ATTRIB Mat4 operator*(const Mat4& a, const Mat4& b);
#endif //__cplusplus
// clang-format off
HIMATH_ATTRIB Mat4 mat4_make(float m00, float m01, float m02, float m03,
                               float m10, float m11, float m12, float m13,
                               float m20, float m21, float m22, float m23,
                               float m30, float m31, float m32, float m33);
// clang-format on
HIMATH_ATTRIB Mat4 mat4_mul(const Mat4* a, const Mat4* b);
HIMATH_ATTRIB Mat4 mat4_identity();
HIMATH_ATTRIB Mat4 mat4_scalev(FVec3 s);
HIMATH_ATTRIB Mat4 mat4_scale(float s);
HIMATH_ATTRIB Mat4 mat4_translation(FVec3 v);
HIMATH_ATTRIB Mat4 mat4_lookat(FVec3 eye, FVec3 target, FVec3 up);
HIMATH_ATTRIB Mat4 mat4_persp(float fov_y,
                              float aspect_ratio,
                              float near_z,
                              float far_z);

typedef struct Quat_
{
    union
    {
        float e[4];
        FVec4 v;
    };
} Quat;

#ifdef __cplusplus
HIMATH_ATTRIB Quat operator*(Quat a, Quat b);
HIMATH_ATTRIB Quat& operator*=(Quat& a, Quat b);
#endif //__cplusplus
HIMATH_ATTRIB Quat quat_mul(Quat a, Quat b);
HIMATH_ATTRIB Mat4 quat_to_matrix(Quat quat);
HIMATH_ATTRIB Quat quat_rotate(FVec3 axis, float angle_deg);

#ifdef HIMATH_IMPL

#include <math.h>

HIMATH_ATTRIB bool almost_equal(float a, float b)
{
    float diff = a - b;
    bool result = (diff > 0.f ? diff : -diff) < 0.001f;
    return result;
}

HIMATH_ATTRIB float radtodeg(float rad)
{
    return rad * 180.f / HIMATH_PI;
}

HIMATH_ATTRIB float degtorad(float deg)
{
    return deg * HIMATH_PI / 180.f;
}

#ifdef __cplusplus

HIMATH_ATTRIB IVec2 operator-(IVec2 v)
{
    return ivec2_negate(v);
}
HIMATH_ATTRIB IVec2 operator+(IVec2 a, IVec2 b)
{
    return ivec2_add(a, b);
}
HIMATH_ATTRIB IVec2 operator-(IVec2 a, IVec2 b)
{
    return ivec2_sub(a, b);
}
HIMATH_ATTRIB IVec2 operator*(IVec2 a, IVec2 b)
{
    return ivec2_mul(a, b);
}
HIMATH_ATTRIB IVec2 operator/(IVec2 a, IVec2 b)
{
    return ivec2_div(a, b);
}
HIMATH_ATTRIB IVec2& operator+=(IVec2& a, IVec2 b)
{
    return a = a + b;
}
HIMATH_ATTRIB IVec2& operator-=(IVec2& a, IVec2 b)
{
    return a = a - b;
}
HIMATH_ATTRIB IVec2& operator*=(IVec2& a, IVec2 b)
{
    return a = a * b;
}
HIMATH_ATTRIB IVec2& operator/=(IVec2& a, IVec2 b)
{
    return a = a / b;
}

#endif

HIMATH_ATTRIB IVec2 ivec2_negate(IVec2 v)
{
    IVec2 result = {-v.x, -v.y};
    return result;
}
HIMATH_ATTRIB IVec2 ivec2_add(IVec2 a, IVec2 b)
{
    IVec2 result = {a.x + b.x, a.y + b.y};
    return result;
}
HIMATH_ATTRIB IVec2 ivec2_sub(IVec2 a, IVec2 b)
{
    IVec2 result = {a.x - b.x, a.y - b.y};
    return result;
}
HIMATH_ATTRIB IVec2 ivec2_mul(IVec2 a, IVec2 b)
{
    IVec2 result = {a.x * b.x, a.y * b.y};
    return result;
}
HIMATH_ATTRIB IVec2 ivec2_div(IVec2 a, IVec2 b)
{
    IVec2 result = {a.x / b.x, a.y / b.y};
    return result;
}

#ifdef __cplusplus

HIMATH_ATTRIB FVec2 operator-(FVec2 v)
{
    return fvec2_negate(v);
}
HIMATH_ATTRIB FVec2 operator+(FVec2 a, FVec2 b)
{
    return fvec2_add(a, b);
}
HIMATH_ATTRIB FVec2 operator-(FVec2 a, FVec2 b)
{
    return fvec2_sub(a, b);
}
HIMATH_ATTRIB FVec2 operator*(FVec2 a, FVec2 b)
{
    return fvec2_mul(a, b);
}
HIMATH_ATTRIB FVec2 operator/(FVec2 a, FVec2 b)
{
    return fvec2_div(a, b);
}
HIMATH_ATTRIB FVec2& operator+=(FVec2& a, FVec2 b)
{
    return a = a + b;
}
HIMATH_ATTRIB FVec2& operator-=(FVec2& a, FVec2 b)
{
    return a = a - b;
}
HIMATH_ATTRIB FVec2& operator*=(FVec2& a, FVec2 b)
{
    return a = a * b;
}
HIMATH_ATTRIB FVec2& operator/=(FVec2& a, FVec2 b)
{
    return a = a / b;
}

#endif //__cplusplus

HIMATH_ATTRIB FVec2 fvec2_negate(FVec2 v)
{
    FVec2 result = {-v.x, -v.y};
    return result;
}
HIMATH_ATTRIB FVec2 fvec2_add(FVec2 a, FVec2 b)
{
    FVec2 result = {a.x + b.x, a.y + b.y};
    return result;
}
HIMATH_ATTRIB FVec2 fvec2_sub(FVec2 a, FVec2 b)
{
    FVec2 result = {a.x - b.x, a.y - b.y};
    return result;
}
HIMATH_ATTRIB FVec2 fvec2_mul(FVec2 a, FVec2 b)
{
    FVec2 result = {a.x * b.x, a.y * b.y};
    return result;
}
HIMATH_ATTRIB FVec2 fvec2_div(FVec2 a, FVec2 b)
{
    FVec2 result = {a.x / b.x, a.y / b.y};
    return result;
}

#ifdef __cplusplus

HIMATH_ATTRIB FVec3 operator-(FVec3 v)
{
    return fvec3_negate(v);
}
HIMATH_ATTRIB FVec3 operator+(FVec3 a, FVec3 b)
{
    return fvec3_add(a, b);
}
HIMATH_ATTRIB FVec3 operator-(FVec3 a, FVec3 b)
{
    return fvec3_sub(a, b);
}
HIMATH_ATTRIB FVec3 operator*(FVec3 a, FVec3 b)
{
    return fvec3_mul(a, b);
}
HIMATH_ATTRIB FVec3 operator/(FVec3 a, FVec3 b)
{
    return fvec3_div(a, b);
}
HIMATH_ATTRIB FVec3 operator*(FVec3 v, float s)
{
    return fvec3_mulf(v, s);
}
HIMATH_ATTRIB FVec3 operator/(FVec3 v, float s)
{
    return fvec3_divf(v, s);
}
HIMATH_ATTRIB FVec3& operator+=(FVec3& a, FVec3 b)
{
    return a = a + b;
}
HIMATH_ATTRIB FVec3& operator-=(FVec3& a, FVec3 b)
{
    return a = a - b;
}
HIMATH_ATTRIB FVec3& operator*=(FVec3& a, FVec3 b)
{
    return a = a * b;
}
HIMATH_ATTRIB FVec3& operator/=(FVec3& a, FVec3 b)
{
    return a = a / b;
}
HIMATH_ATTRIB FVec3& operator*=(FVec3& v, float s)
{
    return v = v * s;
}
HIMATH_ATTRIB FVec3& operator/=(FVec3& v, float s)
{
    return v = v / s;
}

#endif //__cplusplus

HIMATH_ATTRIB FVec3 fvec3_negate(FVec3 v)
{
    FVec3 result = {-v.x, -v.y, -v.z};
    return result;
}
HIMATH_ATTRIB FVec3 fvec3_add(FVec3 a, FVec3 b)
{
    FVec3 result = {a.x + b.x, a.y + b.y, a.z + b.z};
    return result;
}
HIMATH_ATTRIB FVec3 fvec3_sub(FVec3 a, FVec3 b)
{
    FVec3 result = {a.x - b.x, a.y - b.y, a.z - b.z};
    return result;
}
HIMATH_ATTRIB FVec3 fvec3_mul(FVec3 a, FVec3 b)
{
    FVec3 result = {a.x * b.x, a.y * b.y, a.z * b.z};
    return result;
}
HIMATH_ATTRIB FVec3 fvec3_div(FVec3 a, FVec3 b)
{
    FVec3 result = {a.x / b.x, a.y / b.y, a.z / b.z};
    return result;
}
HIMATH_ATTRIB FVec3 fvec3_mulf(FVec3 v, float s)
{
    FVec3 result = {v.x * s, v.y * s, v.z * s};
    return result;
}
HIMATH_ATTRIB FVec3 fvec3_divf(FVec3 v, float s)
{
    FVec3 result = {v.x / s, v.y / s, v.z / s};
    return result;
}
HIMATH_ATTRIB FVec3 fvec3_cross(FVec3 a, FVec3 b)
{
    FVec3 result = {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                    a.x * b.y - a.y * b.x};
    return result;
}
HIMATH_ATTRIB float fvec3_dot(FVec3 a, FVec3 b)
{
    float result = a.x * b.x + a.y * b.y + a.z * b.z;
    return result;
}
HIMATH_ATTRIB float fvec3_length_sq(FVec3 v)
{
    float result = fvec3_dot(v, v);
    return result;
}
HIMATH_ATTRIB float fvec3_length(FVec3 v)
{
    float result = sqrtf(fvec3_length_sq(v));
    return result;
}
HIMATH_ATTRIB FVec3 fvec3_normalize(FVec3 v)
{
    FVec3 result = fvec3_divf(v, fvec3_length(v));
    return result;
}

#ifdef __cplusplus

HIMATH_ATTRIB Mat4 operator*(const Mat4& a, const Mat4& b)
{
    return mat4_mul(&a, &b);
}

#endif //__cplusplus

// clang-format off
HIMATH_ATTRIB Mat4 mat4_make(float m00, float m01, float m02, float m03,
                               float m10, float m11, float m12, float m13,
                               float m20, float m21, float m22, float m23,
                               float m30, float m31, float m32, float m33)
// clang-format on
{
    Mat4 result = {{m00, m10, m20, m30, m01, m11, m21, m31, m02, m12, m22, m32,
                    m03, m13, m23, m33}};
    return result;
}

HIMATH_ATTRIB Mat4 mat4_mul(const Mat4* a, const Mat4* b)
{
    float(*am)[4] = (float(*)[4])a->m;
    float(*bm)[4] = (float(*)[4])b->m;
    Mat4 result = {{bm[0][0] * am[0][0] + bm[0][1] * am[1][0] +
                        bm[0][2] * am[2][0] + bm[0][3] * am[3][0],
                    bm[0][0] * am[0][1] + bm[0][1] * am[1][1] +
                        bm[0][2] * am[2][1] + bm[0][3] * am[3][1],
                    bm[0][0] * am[0][2] + bm[0][1] * am[1][2] +
                        bm[0][2] * am[2][2] + bm[0][3] * am[3][2],
                    bm[0][0] * am[0][3] + bm[0][1] * am[1][3] +
                        bm[0][2] * am[2][3] + bm[0][3] * am[3][3],
                    bm[1][0] * am[0][0] + bm[1][1] * am[1][0] +
                        bm[1][2] * am[2][0] + bm[1][3] * am[3][0],
                    bm[1][0] * am[0][1] + bm[1][1] * am[1][1] +
                        bm[1][2] * am[2][1] + bm[1][3] * am[3][1],
                    bm[1][0] * am[0][2] + bm[1][1] * am[1][2] +
                        bm[1][2] * am[2][2] + bm[1][3] * am[3][2],
                    bm[1][0] * am[0][3] + bm[1][1] * am[1][3] +
                        bm[1][2] * am[2][3] + bm[1][3] * am[3][3],
                    bm[2][0] * am[0][0] + bm[2][1] * am[1][0] +
                        bm[2][2] * am[2][0] + bm[2][3] * am[3][0],
                    bm[2][0] * am[0][1] + bm[2][1] * am[1][1] +
                        bm[2][2] * am[2][1] + bm[2][3] * am[3][1],
                    bm[2][0] * am[0][2] + bm[2][1] * am[1][2] +
                        bm[2][2] * am[2][2] + bm[2][3] * am[3][2],
                    bm[2][0] * am[0][3] + bm[2][1] * am[1][3] +
                        bm[2][2] * am[2][3] + bm[2][3] * am[3][3],
                    bm[3][0] * am[0][0] + bm[3][1] * am[1][0] +
                        bm[3][2] * am[2][0] + bm[3][3] * am[3][0],
                    bm[3][0] * am[0][1] + bm[3][1] * am[1][1] +
                        bm[3][2] * am[2][1] + bm[3][3] * am[3][1],
                    bm[3][0] * am[0][2] + bm[3][1] * am[1][2] +
                        bm[3][2] * am[2][2] + bm[3][3] * am[3][2],
                    bm[3][0] * am[0][3] + bm[3][1] * am[1][3] +
                        bm[3][2] * am[2][3] + bm[3][3] * am[3][3]}};
    return result;
}

HIMATH_ATTRIB Mat4 mat4_identity()
{
    Mat4 result = {{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}};
    return result;
}

HIMATH_ATTRIB Mat4 mat4_scalev(FVec3 s)
{
    Mat4 result = {{s.x, 0.f, 0.f, 0.f, 0.f, s.y, 0.f, 0.f, 0.f, 0.f, s.z, 0.f,
                    0.f, 0.f, 0.f, 1.f}};
    return result;
}

HIMATH_ATTRIB Mat4 mat4_scale(float s)
{
    Mat4 result = {{s, 0.f, 0.f, 0.f, 0.f, s, 0.f, 0.f, 0.f, 0.f, s, 0.f, 0.f,
                    0.f, 0.f, 1.f}};
    return result;
}

HIMATH_ATTRIB Mat4 mat4_translation(FVec3 v)
{
    Mat4 result = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, v.x, v.y, v.z, 1};
    return result;
}

HIMATH_ATTRIB Mat4 mat4_lookat(FVec3 eye, FVec3 target, FVec3 up)
{
    FVec3 k = fvec3_normalize(fvec3_sub(eye, target));
    FVec3 i = fvec3_normalize(fvec3_cross(up, k));
    FVec3 j = fvec3_normalize(fvec3_cross(k, i));
    Mat4 result = {
        i.x,
        j.x,
        k.x,
        0.f,
        i.y,
        j.y,
        k.y,
        0.f,
        i.z,
        j.z,
        k.z,
        0.f,
        -fvec3_dot(i, eye),
        -fvec3_dot(j, eye),
        -fvec3_dot(k, eye),
        1.f,
    };
    return result;
}

HIMATH_ATTRIB Mat4 mat4_persp(float fov_y,
                              float aspect_ratio,
                              float near_z,
                              float far_z)
{
    float theta = degtorad(fov_y) * 0.5f;
    float sin_fov = sinf(theta);
    float cos_fov = cosf(theta);

    float d = cos_fov / sin_fov;
    float da = d / aspect_ratio;
    float range = far_z / (near_z - far_z);
    float rnz = range * near_z;

    Mat4 result = {da,  0.f, 0.f,   0.f,  //
                   0.f, d,   0.f,   0.f,  //
                   0.f, 0.f, range, -1.f, //
                   0.f, 0.f, rnz,   0.f};
    return result;
}

#ifdef __cplusplus

HIMATH_ATTRIB Quat operator*(Quat a, Quat b)
{
    return quat_mul(a, b);
}

HIMATH_ATTRIB Quat& operator*=(Quat& a, Quat b)
{
    a = a * b;
    return a;
}

#endif //__cplusplus

HIMATH_ATTRIB Quat quat_mul(Quat a, Quat b)
{
    Quat result = {
        {a.v.x * b.v.x - a.v.y * b.v.y - a.v.z * b.v.z - a.v.w * b.v.w,
         a.v.x * b.v.y + a.v.y * b.v.x + a.v.z * b.v.w - a.v.w * b.v.z,
         a.v.x * b.v.z + a.v.z * b.v.x - a.v.y * b.v.w + a.v.w * b.v.y,
         a.v.x * b.v.w + a.v.w * b.v.x + a.v.y * b.v.z - a.v.z * b.v.y}};
    return result;
}

HIMATH_ATTRIB Mat4 quat_to_matrix(Quat quat)
{
    FVec4 q = quat.v;

    Mat4 result = {
        {1 - 2 * (q.z * q.z + q.w * q.w), 2 * (q.y * q.z + q.x * q.w),
         2 * (-q.x * q.z + q.y * q.w), 0.f, 2 * (q.y * q.z - q.x * q.w),
         1 - 2 * (q.y * q.y + q.w * q.w), 2 * (q.x * q.y + q.z * q.w), 0.f,
         2 * (q.x * q.z + q.y * q.w), 2 * (-q.x * q.y + q.z * q.w),
         1 - 2 * (q.y * q.y + q.z * q.z), 0.f, 0.f, 0.f, 0.f, 1.f}};
    return result;
}

HIMATH_ATTRIB Quat quat_rotate(FVec3 axis, float angle_deg)
{
    FVec3 axis_normalized = fvec3_normalize(axis);
    float half_radians = degtorad(angle_deg * 0.5f);
    float sin_theta = sinf(half_radians);
    float cos_theta = cosf(half_radians);
    Quat result = {{cos_theta, axis_normalized.x * sin_theta,
                    axis_normalized.y * sin_theta,
                    axis_normalized.z * sin_theta}};
    return result;
}

#endif // HIMATH_IMPL

#endif // HIMATH_H