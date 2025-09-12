/*
 * KoalaBear
 *
 * Source: https://github.com/supranational/sppark.git
 * Status: MODIFIED from sppark/ff/baby_bear.hpp
 * Imported: 2025-09-03 by @felicityin
 *
 * LOCAL CHANGES (high level): 
 *  - 2025-09-03: Change BabyBear to KoalaBear.
 *  - 2025-09-04: Adjust methods reciprocal() and heptaroot().​​
 *  - 2025-09-12: Fix method sqr_n(), and add more comments.
*/

// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cassert>
#include <cstdint>
#include <stdio.h>

#define inline __device__ __forceinline__
#ifdef __GNUC__
#define asm __asm__ __volatile__
#else
#define asm asm volatile
#endif

class kb31_t {
 public:
  using mem_t = kb31_t;
  uint32_t val;

  static const uint32_t P = 0x7f000001;  // The KoalaBear prime: 2^31 - 2^24 + 1 = 127 * 2^24 + 1
  static const uint32_t M = 0x7effffff;  // P - 2
  static const uint32_t ONE = 0x1fffffe; // in Montgomery form
  static const uint32_t TWO = 0x3fffffc; // in Montgomery form
  static const uint32_t MONTY_BITS = 32;
  static const uint32_t RR = 0x17f7efe4; // R = 2^MONTY_BITS, RR = R * R mod P
  static const uint32_t MONTY_MU = 0x81000001;   // MONTY_MU = P^{-1} (mod 2^MONTY_BITS)
  static const uint32_t MONTY_MASK = 0xffffffff; // MONTY_MASK = ((1ULL << MONTY_BITS) - 1);

  static constexpr size_t __device__ bit_length() { return 31; }

  inline uint32_t& operator[](size_t i) { return val; }

  inline uint32_t& operator*() { return val; }

  inline const uint32_t& operator[](size_t i) const { return val; }

  inline uint32_t operator*() const { return val; }

  inline size_t len() const { return 1; }

  constexpr kb31_t() : val(0) {}

  // Create a new field element from something already in MONTY form.
  inline constexpr kb31_t(const uint32_t a): val(a) {}

  inline kb31_t(const uint32_t* p) { val = *p; }

  inline constexpr kb31_t(int a) : val(((uint64_t)a << MONTY_BITS) % P) {}

  static inline const kb31_t zero() { return kb31_t(0); }

  static inline const kb31_t one() { return kb31_t(ONE); }

  static inline const kb31_t two() { return kb31_t(TWO); }

  static inline constexpr uint32_t to_monty(uint32_t x) {
    return (((uint64_t)x << MONTY_BITS) % P);
  }

  static inline uint32_t monty_reduce(uint64_t x) {
    uint64_t t = (x * (uint64_t)MONTY_MU) & (uint64_t)MONTY_MASK;
    uint64_t u = t * (uint64_t)P;
    uint64_t x_sub_u = x - u;
    bool over = x < u;
    uint32_t x_sub_u_hi = (uint32_t)(x_sub_u >> MONTY_BITS);
    uint32_t corr = over ? P : 0;
    return x_sub_u_hi + corr;
  }

  static inline uint32_t from_monty(uint32_t x) {
    return monty_reduce((uint64_t)x);
  }

  inline kb31_t& operator+=(const kb31_t b) {
    val += b.val;
    final_sub(val);
  
    return *this;
  }

  friend inline kb31_t operator+(kb31_t a, const kb31_t b) { 
    return a += b; 
  }

  inline kb31_t& operator<<=(uint32_t l) {
    while (l--) {
      val <<= 1;
      final_sub(val);
    }

    return *this;
  }

  friend inline kb31_t operator<<(kb31_t a, uint32_t l) { return a <<= l; }

  inline kb31_t& operator>>=(uint32_t r) {
    while (r--) {
      val += val & 1 ? P : 0;
      val >>= 1;
    }

    return *this;
  }

  friend inline kb31_t operator>>(kb31_t a, uint32_t r) { return a >>= r; }

  inline kb31_t& operator-=(const kb31_t b) {
    asm("{");
    asm(".reg.pred %brw;");
    asm("setp.lt.u32 %brw, %0, %1;" ::"r"(val), "r"(b.val));
    asm("sub.u32 %0, %0, %1;" : "+r"(val) : "r"(b.val));
    asm("@%brw add.u32 %0, %0, %1;" : "+r"(val) : "r"(P));
    asm("}");

    return *this;
  }

  friend inline kb31_t operator-(kb31_t a, const kb31_t b) { return a -= b; }

  inline kb31_t cneg(bool flag) {
    asm("{");
    asm(".reg.pred %flag;");
    asm("setp.ne.u32 %flag, %0, 0;" ::"r"(val));
    asm("@%flag setp.ne.u32 %flag, %0, 0;" ::"r"((int)flag));
    asm("@%flag sub.u32 %0, %1, %0;" : "+r"(val) : "r"(P));
    asm("}");

    return *this;
  }

  static inline kb31_t cneg(kb31_t a, bool flag) { return a.cneg(flag); }

  inline kb31_t operator-() const { return cneg(*this, true); }

  inline bool operator==(const kb31_t rhs) const { return val == rhs.val; }

  inline bool operator!=(const kb31_t rhs) const { return val != rhs.val; }

  inline bool is_one() const { return val == ONE; }

  inline bool is_zero() const { return val == 0; }

  inline void set_to_zero() { val = 0; }

  friend inline kb31_t czero(const kb31_t a, int set_z) {
    kb31_t ret;

    asm("{");
    asm(".reg.pred %set_z;");
    asm("setp.ne.s32 %set_z, %0, 0;" : : "r"(set_z));
    asm("selp.u32 %0, 0, %1, %set_z;" : "=r"(ret.val) : "r"(a.val));
    asm("}");

    return ret;
  }

  static inline kb31_t csel(const kb31_t a, const kb31_t b, int sel_a) {
    kb31_t ret;

    asm("{");
    asm(".reg.pred %sel_a;");
    asm("setp.ne.s32 %sel_a, %0, 0;" ::"r"(sel_a));
    asm("selp.u32 %0, %1, %2, %sel_a;"
        : "=r"(ret.val)
        : "r"(a.val), "r"(b.val));
    asm("}");

    return ret;
  }

 private:
  static inline void final_sub(uint32_t& val) {
    asm("{");
    asm(".reg.pred %p;");
    asm("setp.ge.u32 %p, %0, %1;" ::"r"(val), "r"(P));
    asm("@%p sub.u32 %0, %0, %1;" : "+r"(val) : "r"(P));
    asm("}");
  }

  inline kb31_t& mul(const kb31_t b) {
    uint32_t tmp[2], red;

    asm("mul.lo.u32 %0, %2, %3; mul.hi.u32 %1, %2, %3;"
        : "=r"(tmp[0]), "=r"(tmp[1])
        : "r"(val), "r"(b.val));
    asm("mul.lo.u32 %0, %1, %2;" : "=r"(red) : "r"(tmp[0]), "r"(M));
    asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.u32 %1, %2, %3, %4;"
        : "+r"(tmp[0]), "=r"(val)
        : "r"(red), "r"(P), "r"(tmp[1]));

    final_sub(val);

    return *this;
  }

  inline uint32_t mul_by_1() const {
    uint32_t tmp[2], red;

    asm("mul.lo.u32 %0, %1, %2;" : "=r"(red) : "r"(val), "r"(M));
    asm("mad.lo.cc.u32 %0, %2, %3, %4; madc.hi.u32 %1, %2, %3, 0;"
        : "=r"(tmp[0]), "=r"(tmp[1])
        : "r"(red), "r"(P), "r"(val));
    return tmp[1];
  }

 public:
  friend inline kb31_t operator*(kb31_t a, const kb31_t b) { return a.mul(b); }

  inline kb31_t& operator*=(const kb31_t a) { return mul(a); }

  // raise to a variable power, variable in respect to threadIdx,
  // but mind the ^ operator's precedence!
  inline kb31_t& operator^=(uint32_t p) {
    kb31_t sqr = *this;
    *this = csel(val, ONE, p & 1);

#pragma unroll 1
    while (p >>= 1) {
      sqr.mul(sqr);
      if (p & 1)
        mul(sqr);
    }

    return *this;
  }

  friend inline kb31_t operator^(kb31_t a, uint32_t p) { return a ^= p; }

  inline kb31_t operator()(uint32_t p) { return *this^p; }

  // raise to a constant power, e.g. x^7, to be unrolled at compile time
  inline kb31_t& operator^=(int p) {
    if (p < 2)
      asm("trap;");

    kb31_t sqr = *this;
    if ((p & 1) == 0) {
      do {
        sqr.mul(sqr);
        p >>= 1;
      } while ((p & 1) == 0);
      *this = sqr;
    }
    for (p >>= 1; p; p >>= 1) {
      sqr.mul(sqr);
      if (p & 1)
        mul(sqr);
    }

    return *this;
  }

  friend inline kb31_t operator^(kb31_t a, int p) { return a ^= p; }

  inline kb31_t operator()(int p) { return *this^p; }

  inline kb31_t square() { return *this * *this; }

  friend inline kb31_t sqr(kb31_t a) { return a.sqr(); }

  inline kb31_t& sqr() { return mul(*this); }

  inline void to() { mul(RR); }

  inline void from() { val = mul_by_1(); }

  template <size_t T>
  static inline kb31_t dot_product(const kb31_t a[T], const kb31_t b[T]) {
    uint32_t acc[2];
    size_t i = 1;

    asm("mul.lo.u32 %0, %2, %3; mul.hi.u32 %1, %2, %3;"
        : "=r"(acc[0]), "=r"(acc[1])
        : "r"(*a[0]), "r"(*b[0]));
    if ((T & 1) == 0) {
      asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.u32 %1, %2, %3, %1;"
          : "+r"(acc[0]), "+r"(acc[1])
          : "r"(*a[i]), "r"(*b[i]));
      i++;
    }
    for (; i < T; i += 2) {
      asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.u32 %1, %2, %3, %1;"
          : "+r"(acc[0]), "+r"(acc[1])
          : "r"(*a[i]), "r"(*b[i]));
      asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.u32 %1, %2, %3, %1;"
          : "+r"(acc[0]), "+r"(acc[1])
          : "r"(*a[i + 1]), "r"(*b[i + 1]));
      final_sub(acc[1]);
    }

    uint32_t red;
    asm("mul.lo.u32 %0, %1, %2;" : "=r"(red) : "r"(acc[0]), "r"(M));
    asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.u32 %1, %2, %3, %1;"
        : "+r"(acc[0]), "+r"(acc[1])
        : "r"(red), "r"(P));
    final_sub(acc[1]);

    return acc[1];
  }

  template <size_t T>
  static inline kb31_t dot_product(kb31_t a0, kb31_t b0, const kb31_t a[T - 1],
                                   const kb31_t* b, size_t stride_b = 1) {
    uint32_t acc[2];
    size_t i = 0;

    asm("mul.lo.u32 %0, %2, %3; mul.hi.u32 %1, %2, %3;"
        : "=r"(acc[0]), "=r"(acc[1])
        : "r"(*a0), "r"(*b0));
    if ((T & 1) == 0) {
      asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.u32 %1, %2, %3, %1;"
          : "+r"(acc[0]), "+r"(acc[1])
          : "r"(*a[i]), "r"(*b[0]));
      i++, b += stride_b;
    }
    for (; i < T - 1; i += 2) {
      asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.u32 %1, %2, %3, %1;"
          : "+r"(acc[0]), "+r"(acc[1])
          : "r"(*a[i]), "r"(*b[0]));
      b += stride_b;
      asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.u32 %1, %2, %3, %1;"
          : "+r"(acc[0]), "+r"(acc[1])
          : "r"(*a[i + 1]), "r"(*b[0]));
      b += stride_b;
      final_sub(acc[1]);
    }

    uint32_t red;
    asm("mul.lo.u32 %0, %1, %2;" : "=r"(red) : "r"(acc[0]), "r"(M));
    asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.u32 %1, %2, %3, %1;"
        : "+r"(acc[0]), "+r"(acc[1])
        : "r"(red), "r"(P));
    final_sub(acc[1]);

    return acc[1];
  }

public:
  static inline kb31_t sqr_n(kb31_t s, uint32_t n) {
#if 0
#pragma unroll 2
    while (n--)
        s.sqr();
#else  // +20% [for reciprocal()]
#pragma unroll 2
    while (n--) {
      uint32_t tmp[2], red;

      asm("mul.lo.u32 %0, %2, %2; mul.hi.u32 %1, %2, %2;"
          : "=r"(tmp[0]), "=r"(tmp[1])
          : "r"(s.val));
      asm("mul.lo.u32 %0, %1, %2;" : "=r"(red) : "r"(tmp[0]), "r"(M));
      asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.u32 %1, %2, %3, %4;"
          : "+r"(tmp[0]), "=r"(s.val)
          : "r"(red), "r"(P), "r"(tmp[1]));

      if (s.val >= P)
        final_sub(s.val);
    }
#endif
    return s;
  }

  static inline kb31_t sqr_n_mul(kb31_t s, uint32_t n, kb31_t m) {
    s = sqr_n(s, n);
    s.mul(m);

    return s;
  }

 public:

  // a^{-1} ≡ a^{P-2} mod P
  inline kb31_t reciprocal() const {
    kb31_t x03, x06, x0f, xff, ret = *this;

    x03 = sqr_n_mul(ret, 1, ret); // 0b11
    x0f = sqr_n_mul(x03, 2, x03); // 0b1111
    x06 = sqr_n(x03, 1);          // 0b110
    ret = sqr_n_mul(x0f, 3, x06); // 0b1111110
    xff = sqr_n_mul(x0f, 4, x0f); // 0b11111111
    ret = sqr_n_mul(ret, 8, xff); // 0b111111011111111
    ret = sqr_n_mul(ret, 8, xff); // 0b11111101111111111111111
    ret = sqr_n_mul(ret, 8, xff); // 0b1111110111111111111111111111111

    return ret;
  }

  friend inline kb31_t operator/(int one, kb31_t a) {
    if (one != 1)
      asm("trap;");
    return a.reciprocal();
  }

  friend inline kb31_t operator/(kb31_t a, kb31_t b) {
    return a * b.reciprocal();
  }

  inline kb31_t& operator/=(const kb31_t a) { return *this *= a.reciprocal(); }

  inline kb31_t heptaroot() const {
    kb31_t x01, x15, ret = *this;

    ret = sqr_n_mul(ret, 2, ret);  // 0b101
    x15 = sqr_n_mul(ret, 2, x01);  // 0b10101
    ret = sqr_n_mul(ret, 7, x15);  // 0b101010010101
    ret = sqr_n_mul(ret, 6, x15);  // 0b101010010101010101
    ret = sqr_n_mul(ret, 6, x15);  // 0b101010010101010101010101
    ret = sqr_n_mul(ret, 6, x15);  // 0b101010010101010101010101010101
    ret = sqr_n_mul(ret, 1, x01);  // 0b1010100101010101010101010101011

    return ret;
  }

  inline void shfl_bfly(uint32_t laneMask) {
    val = __shfl_xor_sync(0xFFFFFFFF, val, laneMask);
  }
};

#undef inline
#undef asm

typedef kb31_t fr_t; 
