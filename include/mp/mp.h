#include "hip/hip_runtime.h"
/*
	co-ecm
	Copyright (C) 2018  Jonas Wloka

	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
			the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	This program is distributed in the hope that it will be useful,
			but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef MONPROD_C_MP_H
#define MONPROD_C_MP_H


#ifndef __HIPCC__ /* when compiling with g++ ... */
#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif
#ifndef __global__
#define __global__
#endif
#endif

#include <stdio.h>
#include <stdint.h>
#include <malloc.h>
#include <hip/hip_runtime.h>
#include <stdbool.h>

#include "build_config.h"

#define MP_STRIDE BATCH_JOB_SIZE

#define _S_IDX(element, limb) (element + (MP_STRIDE * (limb)))

/**
 * Basic datatype for CUDA and host multi-precision numbers.
 */
typedef mp_limb mp_t[LIMBS];


typedef mp_limb mp_strided_t[MP_STRIDE * LIMBS];

/**
 * Array for basic datatypes. Needed in some places where arrays cannot be processed by Nvidias NVCC compiler.
 */
typedef mp_limb *mp_p;

#if (LIMB_BITS == 32)
#define _PRI_ulimb PRIu32
#define _PRI_xlimb PRIx32
#elif (LIMB_BITS == 64)
#define _PRI_ulimb PRIu64
#define _PRI_xlimb PRIx64
#endif

#ifdef __cplusplus
extern "C" {
#endif

/***************/
/* Device code */
/***************/

/*
 * Portable carry-chain arithmetic.
 *
 * The original code used NVIDIA PTX inline assembly with an implicit
 * carry flag register (add.cc / addc / addc.cc / sub.cc / subc / etc.).
 * AMD GCN/RDNA has no equivalent single-instruction carry chain, so
 * we emulate the carry flag with a thread-local variable.
 *
 * The _cc suffix means "set carry flag from this operation".
 * The c prefix (addc/subc/madc) means "consume the current carry flag".
 * Combined (addc_cc, subc_cc, madc_lo_cc, madc_hi_cc) means both.
 */

/* Carry flag: must be declared as a local variable in each device function
 * that uses the carry-chain macros, via: mp_limb __carry_flag = 0;
 * This ensures it lives in a register, not global device memory. */

/* --- Addition with carry chain --- */

/* add.cc: r = a + b, set carry */
#define __add_cc(r, a, b) do { \
	mp_limb __a = (a), __b = (b); \
	(r) = __a + __b; \
	__carry_flag = ((r) < __a) ? 1 : 0; \
} while(0)

/* addc: r = a + b + carry_in, do NOT set carry */
#define __addc(r, a, b) do { \
	mp_limb __a = (a), __b = (b), __cf = __carry_flag; \
	(r) = __a + __b + __cf; \
} while(0)

/* addc.cc: r = a + b + carry_in, set carry */
#define __addc_cc(r, a, b) do { \
	mp_limb __a = (a), __b = (b), __cf = __carry_flag; \
	mp_limb __t = __a + __b; \
	mp_limb __c1 = (__t < __a) ? 1 : 0; \
	(r) = __t + __cf; \
	__carry_flag = __c1 | (((r) < __t) ? 1 : 0); \
} while(0)

/* addcy: carry = 0 + 0 + carry_in (extract carry flag) */
#define __addcy(carry) do { (carry) = __carry_flag; } while(0)

/* addcy2: carry = carry + 0 + carry_in, set carry */
#define __addcy2(carry) do { \
	mp_limb __t = (carry) + __carry_flag; \
	__carry_flag = (__t < (carry)) ? 1 : 0; \
	(carry) = __t; \
} while(0)

/* --- Subtraction with borrow chain --- */

/* sub.cc: r = a - b, set borrow */
#define __sub_cc(r, a, b) do { \
	mp_limb __a = (a), __b = (b); \
	(r) = __a - __b; \
	__carry_flag = (__a < __b) ? 1 : 0; \
} while(0)

/* subc: r = a - b - borrow_in, do NOT set borrow */
#define __subc(r, a, b) do { \
	mp_limb __a = (a), __b = (b), __cf = __carry_flag; \
	(r) = __a - __b - __cf; \
} while(0)

/* subc.cc: r = a - b - borrow_in, set borrow */
#define __subc_cc(r, a, b) do { \
	mp_limb __a = (a), __b = (b), __cf = __carry_flag; \
	mp_limb __t = __a - __b; \
	mp_limb __c1 = (__a < __b) ? 1 : 0; \
	(r) = __t - __cf; \
	__carry_flag = __c1 | ((__t < __cf) ? 1 : 0); \
} while(0)

/* --- Multiplication --- */

#if (LIMB_BITS == 32)

/* mul.lo: r = low32(a * b) */
#define __mul_lo(r, a, b) do { \
	(r) = (mp_limb)((uint32_t)(a) * (uint32_t)(b)); \
} while(0)

/* mul.hi: r = high32(a * b) */
#define __mul_hi(r, a, b) do { \
	(r) = (mp_limb)(((uint64_t)(uint32_t)(a) * (uint64_t)(uint32_t)(b)) >> 32); \
} while(0)

/* mad.lo: r = low32(a * b) + c */
#define __mad_lo(r, a, b, c) do { \
	(r) = (mp_limb)((uint32_t)(a) * (uint32_t)(b)) + (c); \
} while(0)

/* mad.lo.cc: r = low32(a * b) + c, set carry (1-bit overflow from addition) */
#define __mad_lo_cc(r, a, b, c) do { \
	mp_limb __lo = (mp_limb)((uint32_t)(a) * (uint32_t)(b)); \
	mp_limb __c = (mp_limb)(c); \
	(r) = __lo + __c; \
	__carry_flag = ((r) < __lo) ? 1 : 0; \
} while(0)

/* mad.hi: r = high32(a * b) + c */
#define __mad_hi(r, a, b, c) do { \
	(r) = (mp_limb)(((uint64_t)(uint32_t)(a) * (uint64_t)(uint32_t)(b)) >> 32) + (c); \
} while(0)

/* mad.hi.cc: r = high32(a * b) + c, set carry */
#define __mad_hi_cc(r, a, b, c) do { \
	uint64_t __t = ((uint64_t)(uint32_t)(a) * (uint64_t)(uint32_t)(b)); \
	mp_limb __hi = (mp_limb)(__t >> 32); \
	mp_limb __s = __hi + (mp_limb)(c); \
	__carry_flag = (__s < __hi) ? 1 : 0; \
	(r) = __s; \
} while(0)

/* madc.hi: r = high32(a * b) + c + carry_in (no carry out) */
#define __madc_hi(r, a, b, c) do { \
	uint64_t __t = ((uint64_t)(uint32_t)(a) * (uint64_t)(uint32_t)(b)); \
	(r) = (mp_limb)(__t >> 32) + (mp_limb)(c) + __carry_flag; \
} while(0)

/* madc.lo.cc: r = low32(a * b) + c + carry_in, set carry (1-bit overflow) */
#define __madc_lo_cc(r, a, b, c) do { \
	mp_limb __lo = (mp_limb)((uint32_t)(a) * (uint32_t)(b)); \
	mp_limb __c = (mp_limb)(c); \
	mp_limb __cf = __carry_flag; \
	mp_limb __t1 = __lo + __c; \
	mp_limb __c1 = (__t1 < __lo) ? 1 : 0; \
	(r) = __t1 + __cf; \
	__carry_flag = __c1 | (((r) < __t1) ? 1 : 0); \
} while(0)

/* madc.hi.cc: r = high32(a * b) + c + carry_in, set carry */
#define __madc_hi_cc(r, a, b, c) do { \
	uint64_t __t = ((uint64_t)(uint32_t)(a) * (uint64_t)(uint32_t)(b)); \
	mp_limb __hi = (mp_limb)(__t >> 32); \
	uint64_t __s = (uint64_t)__hi + (uint64_t)(uint32_t)(c) + (uint64_t)__carry_flag; \
	(r) = (mp_limb)__s; \
	__carry_flag = (mp_limb)(__s >> 32); \
} while(0)

/* --- Funnel shifts (32-bit only) --- */
/* shf.r.clamp: r = funnel_shift_right(a, b, c) = (b:a) >> c */
#define __shf_r_clamp(r, a, b, c) do { \
	uint32_t __sh = (c); \
	if (__sh >= 32) (r) = (b) >> (__sh - 32); \
	else (r) = ((uint32_t)(a) >> __sh) | ((uint32_t)(b) << (32 - __sh)); \
} while(0)

/* shf.l.clamp: r = funnel_shift_left(a, b, c) = (b:a) << c */
#define __shf_l_clamp(r, a, b, c) do { \
	uint32_t __sh = (c); \
	if (__sh >= 32) (r) = (a) << (__sh - 32); \
	else (r) = ((uint32_t)(b) << __sh) | ((uint32_t)(a) >> (32 - __sh)); \
} while(0)

#elif (LIMB_BITS == 64)

#define __mul_lo(r, a, b) do { \
	(r) = (mp_limb)((uint64_t)(a) * (uint64_t)(b)); \
} while(0)

#define __mul_hi(r, a, b) do { \
	__uint128_t __t = (__uint128_t)(uint64_t)(a) * (__uint128_t)(uint64_t)(b); \
	(r) = (mp_limb)(__t >> 64); \
} while(0)

#define __mad_lo(r, a, b, c) do { \
	(r) = (mp_limb)((uint64_t)(a) * (uint64_t)(b)) + (c); \
} while(0)

#define __mad_lo_cc(r, a, b, c) do { \
	mp_limb __lo = (mp_limb)((uint64_t)(a) * (uint64_t)(b)); \
	mp_limb __c = (mp_limb)(c); \
	(r) = __lo + __c; \
	__carry_flag = ((r) < __lo) ? 1 : 0; \
} while(0)

#define __mad_hi(r, a, b, c) do { \
	__uint128_t __t = (__uint128_t)(uint64_t)(a) * (__uint128_t)(uint64_t)(b); \
	(r) = (mp_limb)(__t >> 64) + (c); \
} while(0)

#define __mad_hi_cc(r, a, b, c) do { \
	__uint128_t __t = (__uint128_t)(uint64_t)(a) * (__uint128_t)(uint64_t)(b); \
	mp_limb __hi = (mp_limb)(__t >> 64); \
	mp_limb __s = __hi + (mp_limb)(c); \
	__carry_flag = (__s < __hi) ? 1 : 0; \
	(r) = __s; \
} while(0)

#define __madc_hi(r, a, b, c) do { \
	__uint128_t __t = (__uint128_t)(uint64_t)(a) * (__uint128_t)(uint64_t)(b); \
	(r) = (mp_limb)(__t >> 64) + (mp_limb)(c) + __carry_flag; \
} while(0)

#define __madc_lo_cc(r, a, b, c) do { \
	mp_limb __lo = (mp_limb)((uint64_t)(a) * (uint64_t)(b)); \
	mp_limb __c = (mp_limb)(c); \
	mp_limb __cf = __carry_flag; \
	mp_limb __t1 = __lo + __c; \
	mp_limb __c1 = (__t1 < __lo) ? 1 : 0; \
	(r) = __t1 + __cf; \
	__carry_flag = __c1 | (((r) < __t1) ? 1 : 0); \
} while(0)

#define __madc_hi_cc(r, a, b, c) do { \
	__uint128_t __t = (__uint128_t)(uint64_t)(a) * (__uint128_t)(uint64_t)(b); \
	mp_limb __hi = (mp_limb)(__t >> 64); \
	__uint128_t __s = (__uint128_t)__hi + (__uint128_t)(uint64_t)(c) + (__uint128_t)__carry_flag; \
	(r) = (mp_limb)__s; \
	__carry_flag = (mp_limb)(__s >> 64); \
} while(0)

#endif /* LIMB_BITS */



/**
 * Print a mp_t number.
 * @param a	Number to print.
 */
__host__ __device__
void mp_print(const mp_t a);

/**
 * Print a mp_t number in hexadecimal.
 * @param a	Number to print.
 */
void mp_print_hex(const mp_t a);


void mp_print_hex_limbs(const mp_t a, size_t limbs);

/**
 * Allocate space for a mp_t on the CUDA device.
 *
 * @param a		Output parameter for the device memory address.
 */
__host__
void mp_dev_init(mp_p *a);

/**
 * Allocate space for a variable size mp_t on the CUDA device.
 *
 * @param a		Output parameter for the device memory address.
 * @param limbs	Number of limbs to allocate for this number.
 */
__host__
void mp_dev_init_limbs(mp_p *a, size_t limbs);

/**
 * Free a mp_t number.
 * @deprecated
 *
 * @param a		Number to deallocate.
 */
__host__ __device__
void mp_free(mp_t a);

/**
 * Set a mp_t to the value of a mp_limb.
 *
 * @param a		Number to set to \p s
 * @param s 	Value to set \p a to
 */
__host__ __device__
void mp_set_ui(mp_t a, mp_limb s);

/**
 * Set a := b.
 *
 * @param a
 * @param b
 */
__host__ __device__
void mp_copy(mp_t a, const mp_t b);

/**
 * Copy a mp_t number to the CUDA device.
 *
 * @param dev_a		Pointer to number in device memory.
 * @param b			Number to copy to the device.
 */
__host__
void mp_copy_to_dev(mp_p dev_a, const mp_t b);

/**
 * Copy variable size mp_t to the CUDA device.
 *
 * @param dev_a 	Pointer to number in device memory.
 * @param b 		Number to copy.
 * @param limbs 	Number of limbs in \p b.
 */
__host__
void mp_copy_to_dev_limbs(mp_p dev_a, const mp_t b, const size_t limbs);

/**
 * Copy mp_t number from device to host memory.
 *
 * @param a 		Host memory destination.
 * @param dev_b 	Device memory source.
 */
__host__
void mp_copy_from_dev(mp_t a, const mp_p dev_b);

/**
 * Set r := a * s;
 *
 * @param r
 * @param a
 * @param s
 * @return 	MSB of r exceeding the number of limbs in r.
 */
__host__ __device__
mp_limb mp_mul_ui(mp_t r, const mp_t a, const mp_limb s);

/**
 * Set r := LSB(a * b).
 * @param r
 * @param a
 * @param b
 * @return 	MSB of r exceeding the number of limbs in r.
 */
__host__ __device__
mp_limb mp_mul_limb(mp_limb *r, mp_limb a, mp_limb b);

/**
 * Set r := a + b
 *
 * @param r
 * @param a
 * @param b
 * @return 	Carry of a + b exceeding the number of limbs in r.
 */
__host__ __device__
mp_limb mp_add(mp_t r, const mp_t a, const mp_t b);

/**
 * Set r := a + b mod n
 *
 * @param r
 * @param a
 * @param b
 * @param n
 */
__host__ __device__
void mp_add_mod(mp_t r, const mp_t a, const mp_t b, const mp_t n);

/**
 * Add \p s to the \p limb -th limb of a, return carry of this limb only.
 *
 * Does not propagate carry to higher limbs.
 *
 * @param a
 * @param s
 * @param limb
 * @return
 */
__host__ __device__
mp_limb mp_limb_addc(mp_t a, const mp_limb s, const size_t limb);

/**
 * Set r:= a + b. Return carry.
 *
 * Propagates carry up to higher limbs.
 *
 * @param r
 * @param a
 * @param b
 * @return Carry exceedingt the number of limbs in r.
 */
__host__ __device__
mp_limb mp_add_ui(mp_t r, const mp_t a, const mp_limb b);

/**
 * Set r := a - b.
 *
 * @param r
 * @param a
 * @param b
 * @return Carry exceedingt the number of limbs in r.
 */
__host__ __device__
mp_limb mp_sub(mp_t r, const mp_t a, const mp_t b);

/**
 * Set r := a - b mod n.
 *
 * Effectively sets r := a + (n - b).
 * @param r
 * @param a
 * @param b
 * @param n
 */
__host__ __device__
void mp_sub_mod(mp_t r, const mp_t a, const mp_t b, const mp_t n);

/**
 * Set r := a - s.
 *
 * @param r
 * @param a
 * @param s
 */
__host__ __device__
void mp_sub_ui(mp_t r, const mp_t a, const mp_limb s);

/**
 * Set r := a * b.
 *
 * Discards any result that exceeds the number of limbs in r.
 *
 * @param r
 * @param a
 * @param b
 */
__host__ __device__
void mp_mul(mp_t r, const mp_t a, const mp_t b);

/**
 * Compare a and b.
 *
 * @param a
 * @param b
 * @return 0 if a == b, -1 if a < b, 1 if a > b
 */
__host__ __device__
int mp_cmp(const mp_t a, const mp_t b);

/**
 * Return 1 if a > b
 */
__host__ __device__
int mp_gt(const mp_t a, const mp_t b);

/**
 * Compare a and b.
 *
 * @param a
 * @param limbs_a Number of limbs in a
 * @param b
 * @param limbs_b Number of limbs in b
 * @return 0 if a == b, -1 if a < b, 1 if a > b
 */
__host__ __device__
int mp_cmp_limbs(const mp_t a, size_t limbs_a, const mp_t b, size_t limbs_b);

/**
 * Compare a and b.
 *
 * @param a
 * @param b
 * @return 0 if a == b, -1 if a < b, 1 if a > b
 */
__host__ __device__
int mp_cmp_ui(const mp_t a, const mp_limb b);

/**
 * Shift a by \p limbs to the left.
 * @param a
 * @param limbs
 */
__host__ __device__
void mp_sl_limbs(mp_t a, size_t limbs);

/**
 * Shift a by \p limbs to the right.
 * @param a
 * @param limbs
 */
__host__ __device__
void mp_sr_limbs(mp_t a, size_t limbs);

/**
 * Switch the values of a and b.
 * @param a
 * @param b
 */
__host__ __device__
void mp_switch(mp_t a, mp_t b);

__host__ __device__
bool mp_iseven(mp_t a);


#ifdef __cplusplus
}
#endif


/**
 * Set bit number \p bit in a to 1.
 */
#define mp_set_bit(a, bit) ((a)[(bit)/LIMB_BITS] |= ((mp_limb_t)1 << ((bit)%LIMB_BITS)))

/**
 * Return whether bit number \p bit in a is set.
 */
//#define mp_test_bit(a, bit) ((((a)[(bit)/LIMB_BITS]) >> ((bit)%LIMB_BITS)) & 1)
#define mp_test_bit(a, bit) !!(((a)[(bit)/LIMB_BITS]) & (1 << ((bit)%LIMB_BITS)))


#endif //MONPROD_C_MP_H
