#ifndef HIMATH_H
#define HIMATH_H

#ifndef __cpluspllus
#include <stdbool.h>
#endif //__cpluspllus

#define HIMATH_PI 3.141592f

#ifdef HIMATH_ENABLE_PREFIX
#define HM_SYM(name) himath_##name
#else
#define HM_SYM(name) name
#endif

bool HM_SYM(almost_equal)(float a, float b);
float HM_SYM(radtodeg)(float rad);
float HM_SYM(degtorad)(float deg);

typedef struct HM_SYM(IVec2_)
{
    int x, y;
} HM_SYM(IVec2);

#ifdef __cplusplus
HM_SYM(IVec2) operator+(HM_SYM(IVec2) a, HM_SYM(IVec2) b);
HM_SYM(IVec2) operator-(HM_SYM(IVec2) a, HM_SYM(IVec2) b);
HM_SYM(IVec2) operator*(HM_SYM(IVec2) a, HM_SYM(IVec2) b);
HM_SYM(IVec2) operator/(HM_SYM(IVec2) a, HM_SYM(IVec2) b);
HM_SYM(IVec2) & operator+=(HM_SYM(IVec2) & a, HM_SYM(IVec2) b);
HM_SYM(IVec2) & operator-=(HM_SYM(IVec2) & a, HM_SYM(IVec2) b);
HM_SYM(IVec2) & operator*=(HM_SYM(IVec2) & a, HM_SYM(IVec2) b);
HM_SYM(IVec2) & operator/=(HM_SYM(IVec2) & a, HM_SYM(IVec2) b);
#endif //__cplusplus
HM_SYM(IVec2) HM_SYM(ivec2_add)(HM_SYM(IVec2) a, HM_SYM(IVec2) b);
HM_SYM(IVec2) HM_SYM(ivec2_sub)(HM_SYM(IVec2) a, HM_SYM(IVec2) b);
HM_SYM(IVec2) HM_SYM(ivec2_mul)(HM_SYM(IVec2) a, HM_SYM(IVec2) b);
HM_SYM(IVec2) HM_SYM(ivec2_div)(HM_SYM(IVec2) a, HM_SYM(IVec2) b);

typedef struct HM_SYM(FVec2_)
{
    float x, y;
} HM_SYM(FVec2);

#ifdef __cplusplus

HM_SYM(FVec2) operator+(HM_SYM(FVec2) a, HM_SYM(FVec2) b);
HM_SYM(FVec2) operator-(HM_SYM(FVec2) a, HM_SYM(FVec2) b);
HM_SYM(FVec2) operator*(HM_SYM(FVec2) a, HM_SYM(FVec2) b);
HM_SYM(FVec2) operator/(HM_SYM(FVec2) a, HM_SYM(FVec2) b);
HM_SYM(FVec2) & operator+=(HM_SYM(FVec2) & a, HM_SYM(FVec2) b);
HM_SYM(FVec2) & operator-=(HM_SYM(FVec2) & a, HM_SYM(FVec2) b);
HM_SYM(FVec2) & operator*=(HM_SYM(FVec2) & a, HM_SYM(FVec2) b);
HM_SYM(FVec2) & operator/=(HM_SYM(FVec2) & a, HM_SYM(FVec2) b);

#endif //__cplusplus

HM_SYM(FVec2) HM_SYM(fvec2_add)(HM_SYM(FVec2) a, HM_SYM(FVec2) b);
HM_SYM(FVec2) HM_SYM(fvec2_sub)(HM_SYM(FVec2) a, HM_SYM(FVec2) b);
HM_SYM(FVec2) HM_SYM(fvec2_mul)(HM_SYM(FVec2) a, HM_SYM(FVec2) b);
HM_SYM(FVec2) HM_SYM(fvec2_div)(HM_SYM(FVec2) a, HM_SYM(FVec2) b);

typedef struct HM_SYM(FVec3_)
{
    float x, y, z;
} HM_SYM(FVec3);

#ifdef __cplusplus
HM_SYM(FVec3) operator+(HM_SYM(FVec3) a, HM_SYM(FVec3) b);
HM_SYM(FVec3) operator-(HM_SYM(FVec3) a, HM_SYM(FVec3) b);
HM_SYM(FVec3) operator*(HM_SYM(FVec3) a, HM_SYM(FVec3) b);
HM_SYM(FVec3) operator/(HM_SYM(FVec3) a, HM_SYM(FVec3) b);
HM_SYM(FVec3) operator*(HM_SYM(FVec3) v, float s);
HM_SYM(FVec3) operator/(HM_SYM(FVec3) v, float s);
HM_SYM(FVec3) & operator+=(HM_SYM(FVec3) & a, HM_SYM(FVec3) b);
HM_SYM(FVec3) & operator-=(HM_SYM(FVec3) & a, HM_SYM(FVec3) b);
HM_SYM(FVec3) & operator*=(HM_SYM(FVec3) & a, HM_SYM(FVec3) b);
HM_SYM(FVec3) & operator/=(HM_SYM(FVec3) & a, HM_SYM(FVec3) b);
#endif //__cplusplus

HM_SYM(FVec3) HM_SYM(fvec3_add)(HM_SYM(FVec3) a, HM_SYM(FVec3) b);
HM_SYM(FVec3) HM_SYM(fvec3_sub)(HM_SYM(FVec3) a, HM_SYM(FVec3) b);
HM_SYM(FVec3) HM_SYM(fvec3_mul)(HM_SYM(FVec3) a, HM_SYM(FVec3) b);
HM_SYM(FVec3) HM_SYM(fvec3_div)(HM_SYM(FVec3) a, HM_SYM(FVec3) b);
HM_SYM(FVec3) HM_SYM(fvec3_mulf)(HM_SYM(FVec3) v, float s);
HM_SYM(FVec3) HM_SYM(fvec3_divf)(HM_SYM(FVec3) v, float s);

HM_SYM(FVec3) HM_SYM(fvec3_cross)(HM_SYM(FVec3) a, HM_SYM(FVec3) b);
float HM_SYM(fvec3_dot)(HM_SYM(FVec3) a, HM_SYM(FVec3) b);
float HM_SYM(fvec3_length_sq)(HM_SYM(FVec3) v);
float HM_SYM(fvec3_length)(HM_SYM(FVec3) v);
HM_SYM(FVec3) HM_SYM(fvec3_normalize)(HM_SYM(FVec3) v);

typedef struct HM_SYM(FVec4_)
{
    float x, y, z, w;
} HM_SYM(FVec4);

typedef union HM_SYM(Mat4_)
{
    float m[16];
    float mm[4][4];
} HM_SYM(Mat4);

#ifdef __cplusplus
HM_SYM(Mat4) operator*(const HM_SYM(Mat4) & a, const HM_SYM(Mat4) & b);
#endif //__cplusplus
// clang-format off
HM_SYM(Mat4) HM_SYM(mat4_make)(float m00, float m01, float m02, float m03,
                               float m10, float m11, float m12, float m13,
                               float m20, float m21, float m22, float m23,
                               float m30, float m31, float m32, float m33);
// clang-format on
HM_SYM(Mat4) HM_SYM(mat4_mul)(const HM_SYM(Mat4) * a, const HM_SYM(Mat4) * b);
HM_SYM(Mat4) HM_SYM(mat4_identity)();
HM_SYM(Mat4) HM_SYM(mat4_scalev)(HM_SYM(FVec3) s);
HM_SYM(Mat4) HM_SYM(mat4_scale)(float s);
HM_SYM(Mat4) HM_SYM(mat4_translation)(HM_SYM(FVec3) v);
HM_SYM(Mat4)
HM_SYM(mat4_lookat)(HM_SYM(FVec3) eye, HM_SYM(FVec3) target, HM_SYM(FVec3) up);
HM_SYM(Mat4)
HM_SYM(mat4_persp)(float fov_y, float aspect_ratio, float near_z, float far_z);

typedef struct HM_SYM(Quat_)
{
    union
    {
        float e[4];
        HM_SYM(FVec4) v;
    };
} HM_SYM(Quat);

#ifdef __cplusplus
HM_SYM(Quat) operator*(HM_SYM(Quat) a, HM_SYM(Quat) b);
HM_SYM(Quat) & operator*=(HM_SYM(Quat) & a, HM_SYM(Quat) b);
#endif //__cplusplus
HM_SYM(Quat) HM_SYM(quat_mul)(HM_SYM(Quat) a, HM_SYM(Quat) b);
HM_SYM(Mat4) HM_SYM(quat_to_matrix)(HM_SYM(Quat) quat);
HM_SYM(Quat) HM_SYM(quat_rotate)(HM_SYM(FVec3) axis, float angle_deg);

#ifdef HIMATH_IMPL

#include <math.h>

bool HM_SYM(almost_equal)(float a, float b)
{
    float diff = a - b;
    bool result = (diff > 0.f ? diff : -diff) < 0.001f;
    return result;
}

float HM_SYM(radtodeg)(float rad)
{
    return rad * 180.f / HIMATH_PI;
}

float HM_SYM(degtorad)(float deg)
{
    return deg * HIMATH_PI / 180.f;
}

#ifdef __cplusplus

HM_SYM(IVec2) operator+(HM_SYM(IVec2) a, HM_SYM(IVec2) b)
{
    return HM_SYM(ivec2_add)(a, b);
}
HM_SYM(IVec2) operator-(HM_SYM(IVec2) a, HM_SYM(IVec2) b)
{
    return HM_SYM(ivec2_sub)(a, b);
}
HM_SYM(IVec2) operator*(HM_SYM(IVec2) a, HM_SYM(IVec2) b)
{
    return HM_SYM(ivec2_mul)(a, b);
}
HM_SYM(IVec2) operator/(HM_SYM(IVec2) a, HM_SYM(IVec2) b)
{
    return HM_SYM(ivec2_div)(a, b);
}
HM_SYM(IVec2) & operator+=(HM_SYM(IVec2) & a, HM_SYM(IVec2) b)
{
    return a = a + b;
}
HM_SYM(IVec2) & operator-=(HM_SYM(IVec2) & a, HM_SYM(IVec2) b)
{
    return a = a - b;
}
HM_SYM(IVec2) & operator*=(HM_SYM(IVec2) & a, HM_SYM(IVec2) b)
{
    return a = a * b;
}
HM_SYM(IVec2) & operator/=(HM_SYM(IVec2) & a, HM_SYM(IVec2) b)
{
    return a = a / b;
}

#endif

HM_SYM(IVec2) HM_SYM(ivec2_add)(HM_SYM(IVec2) a, HM_SYM(IVec2) b)
{
    HM_SYM(IVec2) result = {a.x + b.x, a.y + b.y};
    return result;
}
HM_SYM(IVec2) HM_SYM(ivec2_sub)(HM_SYM(IVec2) a, HM_SYM(IVec2) b)
{
    HM_SYM(IVec2) result = {a.x - b.x, a.y - b.y};
    return result;
}
HM_SYM(IVec2) HM_SYM(ivec2_mul)(HM_SYM(IVec2) a, HM_SYM(IVec2) b)
{
    HM_SYM(IVec2) result = {a.x * b.x, a.y * b.y};
    return result;
}
HM_SYM(IVec2) HM_SYM(ivec2_div)(HM_SYM(IVec2) a, HM_SYM(IVec2) b)
{
    HM_SYM(IVec2) result = {a.x / b.x, a.y / b.y};
    return result;
}

#ifdef __cplusplus

HM_SYM(FVec2) operator+(HM_SYM(FVec2) a, HM_SYM(FVec2) b)
{
    return HM_SYM(fvec2_add)(a, b);
}
HM_SYM(FVec2) operator-(HM_SYM(FVec2) a, HM_SYM(FVec2) b)
{
    return HM_SYM(fvec2_sub)(a, b);
}
HM_SYM(FVec2) operator*(HM_SYM(FVec2) a, HM_SYM(FVec2) b)
{
    return HM_SYM(fvec2_mul)(a, b);
}
HM_SYM(FVec2) operator/(HM_SYM(FVec2) a, HM_SYM(FVec2) b)
{
    return HM_SYM(fvec2_div)(a, b);
}
HM_SYM(FVec2) & operator+=(HM_SYM(FVec2) & a, HM_SYM(FVec2) b)
{
    return a = a + b;
}
HM_SYM(FVec2) & operator-=(HM_SYM(FVec2) & a, HM_SYM(FVec2) b)
{
    return a = a - b;
}
HM_SYM(FVec2) & operator*=(HM_SYM(FVec2) & a, HM_SYM(FVec2) b)
{
    return a = a * b;
}
HM_SYM(FVec2) & operator/=(HM_SYM(FVec2) & a, HM_SYM(FVec2) b)
{
    return a = a / b;
}

#endif //__cplusplus

HM_SYM(FVec2) HM_SYM(fvec2_add)(HM_SYM(FVec2) a, HM_SYM(FVec2) b)
{
    HM_SYM(FVec2) result = {a.x + b.x, a.y + b.y};
    return result;
}
HM_SYM(FVec2) HM_SYM(fvec2_sub)(HM_SYM(FVec2) a, HM_SYM(FVec2) b)
{
    HM_SYM(FVec2) result = {a.x - b.x, a.y - b.y};
    return result;
}
HM_SYM(FVec2) HM_SYM(fvec2_mul)(HM_SYM(FVec2) a, HM_SYM(FVec2) b)
{
    HM_SYM(FVec2) result = {a.x * b.x, a.y * b.y};
    return result;
}
HM_SYM(FVec2) HM_SYM(fvec2_div)(HM_SYM(FVec2) a, HM_SYM(FVec2) b)
{
    HM_SYM(FVec2) result = {a.x / b.x, a.y / b.y};
    return result;
}

#ifdef __cplusplus

HM_SYM(FVec3) operator+(HM_SYM(FVec3) a, HM_SYM(FVec3) b)
{
    return HM_SYM(fvec3_add)(a, b);
}
HM_SYM(FVec3) operator-(HM_SYM(FVec3) a, HM_SYM(FVec3) b)
{
    return HM_SYM(fvec3_sub)(a, b);
}
HM_SYM(FVec3) operator*(HM_SYM(FVec3) a, HM_SYM(FVec3) b)
{
    return HM_SYM(fvec3_mul)(a, b);
}
HM_SYM(FVec3) operator/(HM_SYM(FVec3) a, HM_SYM(FVec3) b)
{
    return HM_SYM(fvec3_div)(a, b);
}
HM_SYM(FVec3) operator*(HM_SYM(FVec3) v, float s)
{
    return HM_SYM(fvec3_mulf)(v, s);
}
HM_SYM(FVec3) operator/(HM_SYM(FVec3) v, float s)
{
    return HM_SYM(fvec3_divf)(v, s);
}
HM_SYM(FVec3) & operator+=(HM_SYM(FVec3) & a, HM_SYM(FVec3) b)
{
    return a = a + b;
}
HM_SYM(FVec3) & operator-=(HM_SYM(FVec3) & a, HM_SYM(FVec3) b)
{
    return a = a - b;
}
HM_SYM(FVec3) & operator*=(HM_SYM(FVec3) & a, HM_SYM(FVec3) b)
{
    return a = a * b;
}
HM_SYM(FVec3) & operator/=(HM_SYM(FVec3) & a, HM_SYM(FVec3) b)
{
    return a = a / b;
}
HM_SYM(FVec3) & operator*=(HM_SYM(FVec3) & v, float s)
{
    return v = v * s;
}
HM_SYM(FVec3) & operator/=(HM_SYM(FVec3) & v, float s)
{
    return v = v / s;
}

#endif //__cplusplus

HM_SYM(FVec3) HM_SYM(fvec3_add)(HM_SYM(FVec3) a, HM_SYM(FVec3) b)
{
    HM_SYM(FVec3) result = {a.x + b.x, a.y + b.y, a.z + b.z};
    return result;
}
HM_SYM(FVec3) HM_SYM(fvec3_sub)(HM_SYM(FVec3) a, HM_SYM(FVec3) b)
{
    HM_SYM(FVec3) result = {a.x - b.x, a.y - b.y, a.z - b.z};
    return result;
}
HM_SYM(FVec3) HM_SYM(fvec3_mul)(HM_SYM(FVec3) a, HM_SYM(FVec3) b)
{
    HM_SYM(FVec3) result = {a.x * b.x, a.y * b.y, a.z * b.z};
    return result;
}
HM_SYM(FVec3) HM_SYM(fvec3_div)(HM_SYM(FVec3) a, HM_SYM(FVec3) b)
{
    HM_SYM(FVec3) result = {a.x / b.x, a.y / b.y, a.z / b.z};
    return result;
}
HM_SYM(FVec3) HM_SYM(fvec3_mulf)(HM_SYM(FVec3) v, float s)
{
    HM_SYM(FVec3) result = {v.x * s, v.y * s, v.z * s};
    return result;
}
HM_SYM(FVec3) HM_SYM(fvec3_divf)(HM_SYM(FVec3) v, float s)
{
    HM_SYM(FVec3) result = {v.x / s, v.y / s, v.z / s};
    return result;
}
HM_SYM(FVec3) HM_SYM(fvec3_cross)(HM_SYM(FVec3) a, HM_SYM(FVec3) b)
{
    HM_SYM(FVec3)
    result = {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
              a.x * b.y - a.y * b.x};
    return result;
}
float HM_SYM(fvec3_dot)(HM_SYM(FVec3) a, HM_SYM(FVec3) b)
{
    float result = a.x * b.x + a.y * b.y + a.z * b.z;
    return result;
}
float HM_SYM(fvec3_length_sq)(HM_SYM(FVec3) v)
{
    float result = HM_SYM(fvec3_dot)(v, v);
    return result;
}
float HM_SYM(fvec3_length)(HM_SYM(FVec3) v)
{
    float result = sqrtf(HM_SYM(fvec3_length_sq)(v));
    return result;
}
HM_SYM(FVec3) HM_SYM(fvec3_normalize)(HM_SYM(FVec3) v)
{
    HM_SYM(FVec3) result = HM_SYM(fvec3_divf)(v, HM_SYM(fvec3_length)(v));
    return result;
}

#ifdef __cplusplus

HM_SYM(Mat4) operator*(const HM_SYM(Mat4) & a, const HM_SYM(Mat4) & b)
{
    return HM_SYM(mat4_mul)(&a, &b);
}

#endif //__cplusplus

// clang-format off
HM_SYM(Mat4) HM_SYM(mat4_make)(float m00, float m01, float m02, float m03,
                               float m10, float m11, float m12, float m13,
                               float m20, float m21, float m22, float m23,
                               float m30, float m31, float m32, float m33)
// clang-format on
{
    HM_SYM(Mat4)
    result = {{m00, m10, m20, m30, m01, m11, m21, m31, m02, m12, m22, m32, m03,
               m13, m23, m33}};
    return result;
}

HM_SYM(Mat4) HM_SYM(mat4_mul)(const HM_SYM(Mat4) * a, const HM_SYM(Mat4) * b)
{
    float(*am)[4] = (float(*)[4])a->m;
    float(*bm)[4] = (float(*)[4])b->m;
    HM_SYM(Mat4)
    result = {{bm[0][0] * am[0][0] + bm[0][1] * am[1][0] + bm[0][2] * am[2][0] +
                   bm[0][3] * am[3][0],
               bm[0][0] * am[0][1] + bm[0][1] * am[1][1] + bm[0][2] * am[2][1] +
                   bm[0][3] * am[3][1],
               bm[0][0] * am[0][2] + bm[0][1] * am[1][2] + bm[0][2] * am[2][2] +
                   bm[0][3] * am[3][2],
               bm[0][0] * am[0][3] + bm[0][1] * am[1][3] + bm[0][2] * am[2][3] +
                   bm[0][3] * am[3][3],
               bm[1][0] * am[0][0] + bm[1][1] * am[1][0] + bm[1][2] * am[2][0] +
                   bm[1][3] * am[3][0],
               bm[1][0] * am[0][1] + bm[1][1] * am[1][1] + bm[1][2] * am[2][1] +
                   bm[1][3] * am[3][1],
               bm[1][0] * am[0][2] + bm[1][1] * am[1][2] + bm[1][2] * am[2][2] +
                   bm[1][3] * am[3][2],
               bm[1][0] * am[0][3] + bm[1][1] * am[1][3] + bm[1][2] * am[2][3] +
                   bm[1][3] * am[3][3],
               bm[2][0] * am[0][0] + bm[2][1] * am[1][0] + bm[2][2] * am[2][0] +
                   bm[2][3] * am[3][0],
               bm[2][0] * am[0][1] + bm[2][1] * am[1][1] + bm[2][2] * am[2][1] +
                   bm[2][3] * am[3][1],
               bm[2][0] * am[0][2] + bm[2][1] * am[1][2] + bm[2][2] * am[2][2] +
                   bm[2][3] * am[3][2],
               bm[2][0] * am[0][3] + bm[2][1] * am[1][3] + bm[2][2] * am[2][3] +
                   bm[2][3] * am[3][3],
               bm[3][0] * am[0][0] + bm[3][1] * am[1][0] + bm[3][2] * am[2][0] +
                   bm[3][3] * am[3][0],
               bm[3][0] * am[0][1] + bm[3][1] * am[1][1] + bm[3][2] * am[2][1] +
                   bm[3][3] * am[3][1],
               bm[3][0] * am[0][2] + bm[3][1] * am[1][2] + bm[3][2] * am[2][2] +
                   bm[3][3] * am[3][2],
               bm[3][0] * am[0][3] + bm[3][1] * am[1][3] + bm[3][2] * am[2][3] +
                   bm[3][3] * am[3][3]}};
    return result;
}

HM_SYM(Mat4) HM_SYM(mat4_identity)()
{
    HM_SYM(Mat4) result = {{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}};
    return result;
}

HM_SYM(Mat4) HM_SYM(mat4_scalev)(HM_SYM(FVec3) s)
{
    HM_SYM(Mat4)
    result = {{s.x, 0.f, 0.f, 0.f, 0.f, s.y, 0.f, 0.f, 0.f, 0.f, s.z, 0.f, 0.f,
               0.f, 0.f, 1.f}};
    return result;
}

HM_SYM(Mat4) HM_SYM(mat4_scale)(float s)
{
    HM_SYM(Mat4)
    result = {{s, 0.f, 0.f, 0.f, 0.f, s, 0.f, 0.f, 0.f, 0.f, s, 0.f, 0.f, 0.f,
               0.f, 1.f}};
    return result;
}

HM_SYM(Mat4) HM_SYM(mat4_translation)(HM_SYM(FVec3) v)
{
    HM_SYM(Mat4)
    result = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, v.x, v.y, v.z, 1};
    return result;
}

HM_SYM(Mat4)
HM_SYM(mat4_lookat)(HM_SYM(FVec3) eye, HM_SYM(FVec3) target, HM_SYM(FVec3) up)
{
    HM_SYM(FVec3) k = HM_SYM(fvec3_normalize)(HM_SYM(fvec3_sub)(eye, target));
    HM_SYM(FVec3) i = HM_SYM(fvec3_normalize)(HM_SYM(fvec3_cross)(up, k));
    HM_SYM(FVec3) j = HM_SYM(fvec3_normalize)(HM_SYM(fvec3_cross)(k, i));
    HM_SYM(Mat4)
    result = {
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
        -HM_SYM(fvec3_dot)(i, eye),
        -HM_SYM(fvec3_dot)(j, eye),
        -HM_SYM(fvec3_dot)(k, eye),
        1.f,
    };
    return result;
}

HM_SYM(Mat4)
HM_SYM(mat4_persp)(float fov_y, float aspect_ratio, float near_z, float far_z)
{
    float theta = HM_SYM(degtorad)(fov_y) * 0.5f;
    float sin_fov = sinf(theta);
    float cos_fov = cosf(theta);

    float d = cos_fov / sin_fov;
    float da = d / aspect_ratio;
    float range = far_z / (near_z - far_z);
    float rnz = range * near_z;

    HM_SYM(Mat4)
    result = {da,  0.f, 0.f,   0.f,  //
              0.f, d,   0.f,   0.f,  //
              0.f, 0.f, range, -1.f, //
              0.f, 0.f, rnz,   0.f};
    return result;
}

#ifdef __cplusplus

HM_SYM(Quat) operator*(HM_SYM(Quat) a, HM_SYM(Quat) b)
{
    return HM_SYM(quat_mul)(a, b);
}

HM_SYM(Quat) & operator*=(HM_SYM(Quat) & a, HM_SYM(Quat) b)
{
    a = a * b;
    return a;
}

#endif //__cplusplus

HM_SYM(Quat) HM_SYM(quat_mul)(HM_SYM(Quat) a, HM_SYM(Quat) b)
{
    HM_SYM(Quat)
    result = {{a.v.x * b.v.x - a.v.y * b.v.y - a.v.z * b.v.z - a.v.w * b.v.w,
               a.v.x * b.v.y + a.v.y * b.v.x + a.v.z * b.v.w - a.v.w * b.v.z,
               a.v.x * b.v.z + a.v.z * b.v.x - a.v.y * b.v.w + a.v.w * b.v.y,
               a.v.x * b.v.w + a.v.w * b.v.x + a.v.y * b.v.z - a.v.z * b.v.y}};
    return result;
}

HM_SYM(Mat4) HM_SYM(quat_to_matrix)(HM_SYM(Quat) quat)
{
    HM_SYM(FVec4) q = quat.v;

    HM_SYM(Mat4)
    result = {{1 - 2 * (q.z * q.z + q.w * q.w), 2 * (q.y * q.z + q.x * q.w),
               2 * (-q.x * q.z + q.y * q.w), 0.f, 2 * (q.y * q.z - q.x * q.w),
               1 - 2 * (q.y * q.y + q.w * q.w), 2 * (q.x * q.y + q.z * q.w),
               0.f, 2 * (q.x * q.z + q.y * q.w), 2 * (-q.x * q.y + q.z * q.w),
               1 - 2 * (q.y * q.y + q.z * q.z), 0.f, 0.f, 0.f, 0.f, 1.f}};
    return result;
}

HM_SYM(Quat) HM_SYM(quat_rotate)(HM_SYM(FVec3) axis, float angle_deg)
{
    HM_SYM(FVec3) axis_normalized = HM_SYM(fvec3_normalize)(axis);
    float half_radians = HM_SYM(degtorad)(angle_deg * 0.5f);
    float sin_theta = sinf(half_radians);
    float cos_theta = cosf(half_radians);
    HM_SYM(Quat)
    result = {{cos_theta, axis_normalized.x * sin_theta,
               axis_normalized.y * sin_theta, axis_normalized.z * sin_theta}};
    return result;
}

#endif // HIMATH_IMPL

#endif // HIMATH_H