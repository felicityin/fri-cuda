/*
 * KoalaBear quartic extension
 *
 * Source: https://github.com/supranational/sppark.git
 * Status: MODIFIED from sppark/ff/baby_bear.hpp
 * Imported: 2025-09-05 by @felicityin
 *
 * LOCAL CHANGES (high level): 
 * - 2025-09-05: Change BabyBear to KoalaBear.
*/

// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cassert>
#include "kb31_t.cuh"

#define inline __device__ __forceinline__
#ifdef __GNUC__
#define asm __asm__ __volatile__
#else
#define asm asm volatile
#endif

# define inline __device__ __forceinline__

class __align__(16) kb31_4_t {
    union { kb31_t c[4]; uint32_t u[4]; };

    static const uint32_t P   = 0x7f000001; // The KoalaBear prime: 2^31 - 2^24 + 1 = 127 * 2^24 + 1
    static const uint32_t M   = 0x7effffff; // P - 2

    static const uint32_t BETA  = 0x5fffffa;   // (3<<32) % P
    static const uint32_t NBETA  = 0x79000007; // (-3<<32) % P

public:
    using mem_t = kb31_4_t;

    inline kb31_t& operator[](size_t i)             { return c[i]; }
    inline const kb31_t& operator[](size_t i) const { return c[i]; }
    inline size_t len() const                       { return 4; }

    inline kb31_4_t()           {}
    inline kb31_4_t(kb31_t a)   { c[0] = a; u[1] = u[2] = u[3] = 0; }

    // this is used in constant declaration, e.g. as kb31_4_t{1, 2, 3, 4}
    __host__ __device__ __forceinline__ kb31_4_t(int a) {
        c[0] = kb31_t{a}; u[1] = u[2] = u[3] = 0;
    }

    __host__ __device__ __forceinline__ kb31_4_t(int d, int f, int g, int h) {
        c[0] = kb31_t{d}; c[1] = kb31_t{f}; c[2] = kb31_t{g}; c[3] = kb31_t{h};
    }

    // Polynomial multiplication/squaring modulo x^4 - BETA
    inline kb31_4_t& sqr() {
        kb31_4_t ret;

# ifdef __CUDA_ARCH__
#  ifdef __GNUC__
#   define asm __asm__ __volatile__
#  else
#   define asm asm volatile
#  endif
        // +25% in comparison to multiplication by itself
        uint32_t u3x2 = u[3]<<1;
        uint32_t u1x2 = u[1]<<1;

        // ret[0] = a[0]*a[0] + BETA*(2*a[1]*a[3] + a[2]*a[2]);
        asm("{ .reg.b32 %lo, %hi, %m; .reg.pred %p;\n\t"
            "mul.lo.u32     %lo, %4, %2;      mul.hi.u32  %hi, %4, %2;\n\t"
            "mad.lo.cc.u32  %lo, %3, %3, %lo; madc.hi.u32 %hi, %3, %3, %hi;\n\t"
            "setp.ge.u32    %p, %hi, %5;\n\t"
            "@%p sub.u32    %hi, %hi, %5;\n\t"

            "mul.lo.u32     %m, %lo, %6;\n\t"
            "mad.lo.cc.u32  %lo, %m, %5, %lo; madc.hi.u32 %hi, %m, %5, %hi;\n\t"

            "mul.lo.u32     %lo, %hi, %7;     mul.hi.u32  %hi, %hi, %7;\n\t"
            "mad.lo.cc.u32  %lo, %1, %1, %lo; madc.hi.u32 %hi, %1, %1, %hi;\n\t"

            "mul.lo.u32     %m, %lo, %6;\n\t"
            "mad.lo.cc.u32  %lo, %m, %5, %lo; madc.hi.u32 %0, %m, %5, %hi;\n\t"
            "setp.ge.u32    %p, %0, %5;\n\t"
            "@%p sub.u32    %0, %0, %5;\n\t"
            "}" : "=r"(ret.u[0])
                : "r"(u[0]), "r"(u[1]), "r"(u[2]), "r"(u3x2),
                  "r"(P), "r"(M), "r"(BETA));

        // ret[1] = 2*(a[0]*a[1] + BETA*(a[2]*a[3]));
        asm("{ .reg.b32 %lo, %hi, %m; .reg.pred %p;\n\t"
            "mul.lo.u32     %lo, %4, %3;      mul.hi.u32  %hi, %4, %3;\n\t"

            "mul.lo.u32     %m, %lo, %6;\n\t"
            "mad.lo.cc.u32  %lo, %m, %5, %lo; madc.hi.u32 %hi, %m, %5, %hi;\n\t"

            "mul.lo.u32     %lo, %hi, %7;     mul.hi.u32  %hi, %hi, %7;\n\t"
            "mad.lo.cc.u32  %lo, %2, %1, %lo; madc.hi.u32 %hi, %2, %1, %hi;\n\t"
            "setp.ge.u32    %p, %hi, %5;\n\t"
            "@%p sub.u32    %hi, %hi, %5;\n\t"

            "mul.lo.u32     %m, %lo, %6;\n\t"
            "mad.lo.cc.u32  %lo, %m, %5, %lo; madc.hi.u32 %0, %m, %5, %hi;\n\t"
            "setp.ge.u32    %p, %0, %5;\n\t"
            "@%p sub.u32    %0, %0, %5;\n\t"
            "}" : "=r"(ret.u[1])
                : "r"(u[0]), "r"(u1x2), "r"(u[2]), "r"(u3x2),
                  "r"(P), "r"(M), "r"(BETA));

        // ret[2] = 2*a[0]*a[2] + a[1]*a[1] + BETA*(a[3]*a[3]);
        asm("{ .reg.b32 %lo, %hi, %m; .reg.pred %p;\n\t"
            "mul.lo.u32     %lo, %4, %4;      mul.hi.u32  %hi, %4, %4;\n\t"

            "mul.lo.u32     %m, %lo, %6;\n\t"
            "mad.lo.cc.u32  %lo, %m, %5, %lo; madc.hi.u32 %m, %m, %5, %hi;\n\t"

            "mul.lo.u32     %lo, %3, %1;      mul.hi.u32  %hi, %3, %1;\n\t"
            "mad.lo.cc.u32  %lo, %2, %2, %lo; madc.hi.u32 %hi, %2, %2, %hi;\n\t"
            "mad.lo.cc.u32  %lo, %m, %7, %lo; madc.hi.u32 %hi, %m, %7, %hi;\n\t"
            "setp.ge.u32    %p, %hi, %5;\n\t"
            "@%p sub.u32    %hi, %hi, %5;\n\t"

            "mul.lo.u32     %m, %lo, %6;\n\t"
            "mad.lo.cc.u32  %lo, %m, %5, %lo; madc.hi.u32 %0, %m, %5, %hi;\n\t"
            "setp.ge.u32    %p, %0, %5;\n\t"
            "@%p sub.u32    %0, %0, %5;\n\t"
            "}" : "=r"(ret.u[2])
                : "r"(u[0]<<1), "r"(u[1]), "r"(u[2]), "r"(u[3]),
                  "r"(P), "r"(M), "r"(BETA));

        // ret[3] = 2*(a[0]*a[3] + a[1]*a[2]);
        asm("{ .reg.b32 %lo, %hi, %m; .reg.pred %p;\n\t"
            "mul.lo.u32     %lo, %4, %1;      mul.hi.u32  %hi, %4, %1;\n\t"
            "mad.lo.cc.u32  %lo, %3, %2, %lo; madc.hi.u32 %hi, %3, %2, %hi;\n\t"
            "setp.ge.u32    %p, %hi, %5;\n\t"
            "@%p sub.u32    %hi, %hi, %5;\n\t"

            "mul.lo.u32     %m, %lo, %6;\n\t"
            "mad.lo.cc.u32  %lo, %m, %5, %lo; madc.hi.u32 %0, %m, %5, %hi;\n\t"
            "setp.ge.u32    %p, %0, %5;\n\t"
            "@%p sub.u32    %0, %0, %5;\n\t"
            "}" : "=r"(ret.u[3])
                : "r"(u[0]), "r"(u1x2), "r"(u[2]), "r"(u3x2),
                  "r"(P), "r"(M), "r"(BETA));
#  undef asm
# else
        union { uint64_t wl; uint32_t w[2]; };
        uint32_t u3x2 = u[3]<<1;
        uint32_t u1x2 = u[1]<<1;

        // ret[0] = a[0]*a[0] + BETA*(2*a[1]*a[3] + a[2]*a[2]);
        wl  = u[1] * (uint64_t)u3x2;
        wl += u[2] * (uint64_t)u[2];        final_sub(w[1]);
        wl += (w[0] * M) * (uint64_t)P;
        wl  = w[1] * (uint64_t)BETA;
        wl += u[0] * (uint64_t)u[0];
        wl += (w[0] * M) * (uint64_t)P;
        ret.u[0] = final_sub(w[1]);

        // ret[1] = 2*(a[0]*a[1] + BETA*(a[2]*a[3]));
        wl  = u[2] * (uint64_t)u3x2;
        wl += (w[0] * M) * (uint64_t)P;
        wl  = w[1] * (uint64_t)BETA;
        wl += u[0] * (uint64_t)u1x2;        final_sub(w[1]);
        wl += (w[0] * M) * (uint64_t)P;
        ret.u[1] = final_sub(w[1]);

        // ret[2] = 2*a[0]*a[2] + a[1]*a[1] + BETA*(a[3]*a[3]);
        wl  = u[3] * (uint64_t)u[3];
        wl += (w[0] * M) * (uint64_t)P;
        auto hi  = w[1];
        wl  = u[2] * (uint64_t)(u[0]<<1);
        wl += u[1] * (uint64_t)u[1];
        wl += hi * (uint64_t)BETA;          final_sub(w[1]);
        wl += (w[0] * M) * (uint64_t)P;
        ret.u[2] = final_sub(w[1]);

        // ret[3] = 2*(a[0]*a[3] + a[1]*a[2]);
        wl  = u[0] * (uint64_t)u3x2;
        wl += u[2] * (uint64_t)u1x2;        final_sub(w[1]);
        wl += (w[0] * M) * (uint64_t)P;
        ret.u[3] = final_sub(w[1]);
# endif

        return *this = ret;
    }

private:
    static inline uint32_t final_sub(uint32_t& u) {
        if (u >= P)
            u -= P;
        return u;
    }

    inline kb31_4_t& mul(const kb31_4_t& b) {
        kb31_4_t ret;

# ifdef __CUDA_ARCH__
#  ifdef __GNUC__
#   define asm __asm__ __volatile__
#  else
#   define asm asm volatile
#  endif
        // ret[0] = a[0]*b[0] + BETA*(a[1]*b[3] + a[2]*b[2] + a[3]*b[1]);
        asm("{ .reg.b32 %lo, %hi, %m; .reg.pred %p;\n\t"
            "mul.lo.u32    %lo, %4, %6;      mul.hi.u32  %hi, %4, %6;\n\t"
            "mad.lo.cc.u32 %lo, %3, %7, %lo; madc.hi.u32 %hi, %3, %7, %hi;\n\t"
            "mad.lo.cc.u32 %lo, %2, %8, %lo; madc.hi.u32 %hi, %2, %8, %hi;\n\t"
            "setp.ge.u32 %p, %hi, %9;\n\t"
            "@%p sub.u32 %hi, %hi, %9;\n\t"

            "mul.lo.u32    %m, %lo, %10;\n\t"
            "mad.lo.cc.u32 %lo, %m, %9, %lo; madc.hi.u32 %hi, %m, %9, %hi;\n\t"

            "mul.lo.u32    %lo, %hi, %11;    mul.hi.u32  %hi, %hi, %11;\n\t"
            "mad.lo.cc.u32 %lo, %1, %5, %lo; madc.hi.u32 %hi, %1, %5, %hi;\n\t"

            "mul.lo.u32    %m, %lo, %10;\n\t"
            "mad.lo.cc.u32 %lo, %m, %9, %lo; madc.hi.u32 %0, %m, %9, %hi;\n\t"
            "setp.ge.u32 %p, %0, %9;\n\t"
            "@%p sub.u32 %0, %0, %9;\n\t"
            "}" : "=r"(ret.u[0])
                : "r"(u[0]), "r"(u[1]), "r"(u[2]), "r"(u[3]),
                  "r"(b.u[0]), "r"(b.u[1]), "r"(b.u[2]), "r"(b.u[3]),
                  "r"(P), "r"(M), "r"(BETA));

        // ret[1] = a[0]*b[1] + a[1]*b[0] + BETA*(a[2]*b[3] + a[3]*b[2]);
        asm("{ .reg.b32 %lo, %hi, %m; .reg.pred %p;\n\t"
            "mul.lo.u32    %lo, %4, %7;      mul.hi.u32  %hi, %4, %7;\n\t"
            "mad.lo.cc.u32 %lo, %3, %8, %lo; madc.hi.u32 %hi, %3, %8, %hi;\n\t"

            "mul.lo.u32    %m, %lo, %10;\n\t"
            "mad.lo.cc.u32 %lo, %m, %9, %lo; madc.hi.u32 %hi, %m, %9, %hi;\n\t"

            "mul.lo.u32    %lo, %hi, %11;    mul.hi.u32  %hi, %hi, %11;\n\t"
            "mad.lo.cc.u32 %lo, %2, %5, %lo; madc.hi.u32 %hi, %2, %5, %hi;\n\t"
            "mad.lo.cc.u32 %lo, %1, %6, %lo; madc.hi.u32 %hi, %1, %6, %hi;\n\t"
            "setp.ge.u32 %p, %hi, %9;\n\t"
            "@%p sub.u32 %hi, %hi, %9;\n\t"

            "mul.lo.u32    %m, %lo, %10;\n\t"
            "mad.lo.cc.u32 %lo, %m, %9, %lo; madc.hi.u32 %0, %m, %9, %hi;\n\t"
            "setp.ge.u32 %p, %0, %9;\n\t"
            "@%p sub.u32 %0, %0, %9;\n\t"
            "}" : "=r"(ret.u[1])
                : "r"(u[0]), "r"(u[1]), "r"(u[2]), "r"(u[3]),
                  "r"(b.u[0]), "r"(b.u[1]), "r"(b.u[2]), "r"(b.u[3]),
                  "r"(P), "r"(M), "r"(BETA));

        // ret[2] = a[0]*b[2] + a[1]*b[1] + a[2]*b[0] + BETA*(a[3]*b[3]);
        asm("{ .reg.b32 %lo, %hi, %m; .reg.pred %p;\n\t"
            "mul.lo.u32    %lo, %4, %8;      mul.hi.u32  %hi, %4, %8;\n\t"

            "mul.lo.u32    %m, %lo, %10;\n\t"
            "mad.lo.cc.u32 %lo, %m, %9, %lo; madc.hi.u32 %hi, %m, %9, %hi;\n\t"

            "mul.lo.u32    %lo, %hi, %11;    mul.hi.u32  %hi, %hi, %11;\n\t"
            "mad.lo.cc.u32 %lo, %3, %5, %lo; madc.hi.u32 %hi, %3, %5, %hi;\n\t"
            "mad.lo.cc.u32 %lo, %2, %6, %lo; madc.hi.u32 %hi, %2, %6, %hi;\n\t"
            "mad.lo.cc.u32 %lo, %1, %7, %lo; madc.hi.u32 %hi, %1, %7, %hi;\n\t"
            "setp.ge.u32 %p, %hi, %9;\n\t"
            "@%p sub.u32 %hi, %hi, %9;\n\t"

            "mul.lo.u32    %m, %lo, %10;\n\t"
            "mad.lo.cc.u32 %lo, %m, %9, %lo; madc.hi.u32 %0, %m, %9, %hi;\n\t"
            "setp.ge.u32 %p, %0, %9;\n\t"
            "@%p sub.u32 %0, %0, %9;\n\t"
            "}" : "=r"(ret.u[2])
                : "r"(u[0]), "r"(u[1]), "r"(u[2]), "r"(u[3]),
                  "r"(b.u[0]), "r"(b.u[1]), "r"(b.u[2]), "r"(b.u[3]),
                  "r"(P), "r"(M), "r"(BETA));

        // ret[3] = a[0]*b[3] + a[1]*b[2] + a[2]*b[1] + a[3]*b[0];
        asm("{ .reg.b32 %lo, %hi, %m; .reg.pred %p;\n\t"
            "mul.lo.u32    %lo, %4, %5;      mul.hi.u32  %hi, %4, %5;\n\t"
            "mad.lo.cc.u32 %lo, %3, %6, %lo; madc.hi.u32 %hi, %3, %6, %hi;\n\t"
            "mad.lo.cc.u32 %lo, %2, %7, %lo; madc.hi.u32 %hi, %2, %7, %hi;\n\t"
            "mad.lo.cc.u32 %lo, %1, %8, %lo; madc.hi.u32 %hi, %1, %8, %hi;\n\t"
            "setp.ge.u32 %p, %hi, %9;\n\t"
            "@%p sub.u32 %hi, %hi, %9;\n\t"

            "mul.lo.u32    %m, %lo, %10;\n\t"
            "mad.lo.cc.u32 %lo, %m, %9, %lo; madc.hi.u32 %0, %m, %9, %hi;\n\t"
            "setp.ge.u32 %p, %0, %9;\n\t"
            "@%p sub.u32 %0, %0, %9;\n\t"
            "}" : "=r"(ret.u[3])
                : "r"(u[0]), "r"(u[1]), "r"(u[2]), "r"(u[3]),
                  "r"(b.u[0]), "r"(b.u[1]), "r"(b.u[2]), "r"(b.u[3]),
                  "r"(P), "r"(M), "r"(BETA));
#  undef asm
# else
        union { uint64_t wl; uint32_t w[2]; };

        // ret[0] = a[0]*b[0] + BETA*(a[1]*b[3] + a[2]*b[2] + a[3]*b[1]);
        wl  = u[1] * (uint64_t)b.u[3];
        wl += u[2] * (uint64_t)b.u[2];
        wl += u[3] * (uint64_t)b.u[1];      final_sub(w[1]);
        wl += (w[0] * M) * (uint64_t)P;
        wl  = w[1] * (uint64_t)BETA;
        wl += u[0] * (uint64_t)b.u[0];
        wl += (w[0] * M) * (uint64_t)P;
        ret.u[0] = final_sub(w[1]);

        // ret[1] = a[0]*b[1] + a[1]*b[0] + BETA*(a[2]*b[3] + a[3]*b[2]);
        wl  = u[2] * (uint64_t)b.u[3];
        wl += u[3] * (uint64_t)b.u[2];
        wl += (w[0] * M) * (uint64_t)P;
        wl  = w[1] * (uint64_t)BETA;
        wl += u[0] * (uint64_t)b.u[1];
        wl += u[1] * (uint64_t)b.u[0];      final_sub(w[1]);
        wl += (w[0] * M) * (uint64_t)P;
        ret.u[1] = final_sub(w[1]);

        // ret[2] = a[0]*b[2] + a[1]*b[1] + a[2]*b[0] + BETA*(a[3]*b[3]);
        wl  = u[3] * (uint64_t)b.u[3];
        wl += (w[0] * M) * (uint64_t)P;
        wl  = w[1] * (uint64_t)BETA;
        wl += u[0] * (uint64_t)b.u[2];
        wl += u[1] * (uint64_t)b.u[1];
        wl += u[2] * (uint64_t)b.u[0];      final_sub(w[1]);
        wl += (w[0] * M) * (uint64_t)P;
        ret.u[2] = final_sub(w[1]);

        // ret[3] = a[0]*b[3] + a[1]*b[2] + a[2]*b[1] + a[3]*b[0];
        wl  = u[0] * (uint64_t)b.u[3];
        wl += u[1] * (uint64_t)b.u[2];
        wl += u[2] * (uint64_t)b.u[1];
        wl += u[3] * (uint64_t)b.u[0];      final_sub(w[1]);
        wl += (w[0] * M) * (uint64_t)P;
        ret.u[3] = final_sub(w[1]);
# endif

        return *this = ret;
    }

public:

# ifdef __CUDACC_RDC__

    friend __device__ __noinline__ kb31_4_t operator*(kb31_4_t a, kb31_4_t b) {
        return a.mul(b);
    }

    inline kb31_4_t& operator*=(const kb31_4_t& b) {
        return *this = *this * b;
    }

    friend __device__ __noinline__ kb31_4_t operator*(kb31_4_t a, kb31_t b) {
        kb31_4_t ret;

        for (size_t i = 0; i < 4; i++)
            ret[i] = a[i] * b;

        return ret;
    }

    friend inline kb31_4_t operator*(kb31_t b, const kb31_4_t& a) {
        return a * b;
    }

    inline kb31_4_t& operator*=(kb31_t b) {
        return *this = *this * b;
    }

# else

    friend inline kb31_4_t operator*(kb31_4_t a, const kb31_4_t& b) {
        return a.mul(b);
    }

    inline kb31_4_t& operator*=(const kb31_4_t& b) {
        return mul(b);
    }

    inline kb31_4_t& operator*=(kb31_t b) {
        for (size_t i = 0; i < 4; i++)
            c[i] *= b;

        return *this;
    }

    friend inline kb31_4_t operator*(kb31_4_t a, kb31_t b) {
        return a *= b;
    }

    friend inline kb31_4_t operator*(kb31_t b, kb31_4_t a) {
        return a *= b;
    }

# endif

    friend inline kb31_4_t operator+(const kb31_4_t& a, const kb31_4_t& b) {
        kb31_4_t ret;

        for (size_t i = 0; i < 4; i++)
            ret[i] = a[i] + b[i];

        return ret;
    }

    inline kb31_4_t& operator+=(const kb31_4_t& b) {
        return *this = *this + b;
    }

    friend inline kb31_4_t operator+(const kb31_4_t& a, kb31_t b) {
        kb31_4_t ret;

        ret[0] = a[0] + b;
        ret[1] = a[1];
        ret[2] = a[2];
        ret[3] = a[3];

        return ret;
    }

    friend inline kb31_4_t operator+(kb31_t b, const kb31_4_t& a) {
        return a + b;
    }

    inline kb31_4_t& operator+=(kb31_t b) {
        c[0] += b;
        return *this;
    }

    friend inline kb31_4_t operator-(const kb31_4_t& a, const kb31_4_t& b) {
        kb31_4_t ret;

        for (size_t i = 0; i < 4; i++)
            ret[i] = a[i] - b[i];

        return ret;
    }

    inline kb31_4_t& operator-=(const kb31_4_t& b) {
        return *this = *this - b;
    }

    friend inline kb31_4_t operator-(const kb31_4_t& a, kb31_t b) {
        kb31_4_t ret;

        ret[0] = a[0] - b;
        ret[1] = a[1];
        ret[2] = a[2];
        ret[3] = a[3];

        return ret;
    }

    friend inline kb31_4_t operator-(kb31_t b, const kb31_4_t& a) {
        kb31_4_t ret;

        ret[0] = b - a[0];
        ret[1] = -a[1];
        ret[2] = -a[2];
        ret[3] = -a[3];

        return ret;
    }

    inline kb31_4_t& operator-=(kb31_t b) {
        c[0] -= b;
        return *this;
    }

private:
    inline kb31_t recip_b0() const {
        union { uint64_t wl; uint32_t w[2]; };

        // c[0]*c[0] - beta*(c[1]*kb31_t{u[3]<<1} - c[2]*c[2]);
        wl  = u[1] * (uint64_t)(u[3]<<1);
        wl += u[2] * (uint64_t)(P-u[2]);
        wl += (w[0] * M) * (uint64_t)P;   final_sub(w[1]);
        wl  = w[1] * (uint64_t)(P-BETA);
        wl += u[0] * (uint64_t)u[0];
        wl += (w[0] * M) * (uint64_t)P;

        return kb31_t{final_sub(w[1])};
    }

    inline kb31_t recip_b2() const {
        union { uint64_t wl; uint32_t w[2]; };

        // c[0]*kb31_t{u[2]<<1} - c[1]*c[1] - beta*(c[3]*c[3]);
        wl  = u[3] * (uint64_t)u[3];
        wl += (w[0] * M) * (uint64_t)P;   final_sub(w[1]);
        wl  = w[1] * (uint64_t)(P-BETA);
        wl += u[1] * (uint64_t)(P-u[1]);
        wl += u[0] * (uint64_t)(u[2]<<1);   final_sub(w[1]);
        wl += (w[0] * M) * (uint64_t)P;

        return kb31_t{final_sub(w[1])};
    }

    inline kb31_4_t recip_ret(kb31_t b0, kb31_t b2) const {
        kb31_4_t ret;
        union { uint64_t wl; uint32_t w[2]; };

        wl  = b2[0] * (uint64_t)BETA;
        wl += (w[0] * M) * (uint64_t)P;

        uint32_t beta_b2 = w[1];

        // ret[0] = c[0]*b0 - c[2]*beta_b2;
        wl  = u[0] * (uint64_t)b0[0];
        wl += (P-u[2]) * (uint64_t)beta_b2;
        wl += (w[0] * M) * (uint64_t)P;
        ret.u[0] = final_sub(w[1]);

        // ret[1] = c[3]*beta_b2 - c[1]b0;
        wl  = u[3] * (uint64_t)beta_b2;
        wl += (P-u[1]) * (uint64_t)b0[0];
        wl += (w[0] * M) * (uint64_t)P;
        ret.u[1] = final_sub(w[1]);

        // ret[2] = c[2]*b0 - c[0]*b2;
        wl  = u[2] * (uint64_t)b0[0];
        wl += (P-u[0]) * (uint64_t)b2[0];
        wl += (w[0] * M) * (uint64_t)P;
        ret.u[2] = final_sub(w[1]);

        // ret[3] = c[1]*b2 - c[3]*b0;
        wl  = u[1] * (uint64_t)b2[0];
        wl += (P-u[3]) * (uint64_t)b0[0];
        wl += (w[0] * M) * (uint64_t)P;
        ret.u[3] = final_sub(w[1]);

        return ret;
    }

public:
    inline kb31_4_t reciprocal() const {
        const kb31_t beta{BETA};

        kb31_t b0 = recip_b0();
        kb31_t b2 = recip_b2();

        kb31_t inv = b0*b0 - beta*b2*b2;
        inv = 1 / inv;

        b0 *= inv;
        b2 *= inv;

        return recip_ret(b0, b2);
    }

    friend inline kb31_4_t operator/(int one, const kb31_4_t& a) {
        assert(one == 1); return a.reciprocal();
    }

    friend inline kb31_4_t operator/(const kb31_4_t& a, const kb31_4_t& b) {
        return a * b.reciprocal();
    }

    friend inline kb31_4_t operator/(kb31_t a, const kb31_4_t& b) {
        return b.reciprocal() * a;
    }

    friend inline kb31_4_t operator/(const kb31_4_t& a, kb31_t b) {
        return a * b.reciprocal();
    }

    inline kb31_4_t& operator/=(const kb31_4_t& a) {
        return *this *= a.reciprocal();
    }

    inline kb31_4_t& operator/=(kb31_t a) {
        return *this *= a.reciprocal();
    }

    inline bool is_one() const {
        return c[0].is_one() & u[1]==0 & u[2]==0 & u[3]==0;
    }

    inline bool is_zero() const {
        return u[0]==0 & u[1]==0 & u[2]==0 & u[3]==0;
    }

    // raise to a variable power, variable in respect to threadIdx,
    // but mind the ^ operator's precedence!
    inline kb31_4_t& operator^=(uint32_t p) {
        kb31_4_t sqr = *this;

        if (!(p & 1)) {
            c[0] = kb31_t{1};
            c[1] = c[2] = c[3] = 0;
        }

        #pragma unroll 1
        while (p >>= 1) {
            sqr.sqr();
            if (p & 1)
                mul(sqr);
        }

        return *this;
    }

    friend inline kb31_4_t operator^(kb31_4_t a, uint32_t p) {
        return a ^= p;
    }

    inline kb31_4_t operator()(uint32_t p) {
        return *this^p;
    }

    // raise to a constant power, e.g. x^7, to be unrolled at compile time
    inline kb31_4_t& operator^=(int p) {
        assert(p >= 2);

        kb31_4_t sqr = *this;
        if ((p&1) == 0) {
            do {
                sqr.sqr();
                p >>= 1;
            } while ((p&1) == 0);
            *this = sqr;
        }
        for (p >>= 1; p; p >>= 1) {
            sqr.sqr();
            if (p&1)
                mul(sqr);
        }

        return *this;
    }

    friend inline kb31_4_t operator^(kb31_4_t a, int p) {
        return a ^= p;
    }

    inline kb31_4_t operator()(int p) {
        return *this^p;
    }
# undef inline

public:
    friend inline bool operator==(const kb31_4_t& a, const kb31_4_t& b) {
        return a.u[0]==b.u[0] & a.u[1]==b.u[1] & a.u[2]==b.u[2] & a.u[3]==b.u[3];
    }

    friend inline bool operator!=(const kb31_4_t& a, const kb31_4_t& b) {
        return a.u[0]!=b.u[0] | a.u[1]!=b.u[1] | a.u[2]!=b.u[2] | a.u[3]!=b.u[3];
     }

# if defined(_GLIBCXX_IOSTREAM) || defined(_IOSTREAM_) // non-standard
    friend std::ostream& operator<<(std::ostream& os, const kb31_4_t& a) {
        os << "[" << a.c[0] << ", " << a.c[1] << ", " << a.c[2] << ", " << a.c[3] << "]";
        return os;
    }
# endif
};

typedef kb31_4_t fr4_t; 
