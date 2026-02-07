# WebAssembly SIMD ISA

## The Core Type: `v128`

The `v128` struct is a single 128-bit wide value. It is the fundamental type for all operations below. It is layout-compatible with the WASM `v128` type.

---

## 1. Memory Access (Loads and Stores)

These instructions handle moving data between linear memory and `v128` registers.

### Quirks & Requirements

- Unsafety: All functions in this section are `unsafe` because they dereference raw pointers. You must ensure the pointer is valid for the size of the read/write.
- Alignment: These intrinsics perform unaligned (1-aligned) loads and stores. You do not need to ensure the pointer is aligned to 16 bytes.

### General Load/Store

| Instruction      | Signature                           | Description                                    |
| ---------------- | ----------------------------------- | ---------------------------------------------- |
| `v128_load`  | `unsafe fn(m: *const v128) -> v128` | Loads a 128-bit vector from the given address. |
| `v128_store` | `unsafe fn(m: *mut v128, a: v128)`  | Stores a 128-bit vector to the given address.  |

### Load and Extend

Loads a smaller data type (64 bits or less) and extends it to fill a 128-bit vector.

| Instruction                   | Signature                          | Description                             |
| ----------------------------- | ---------------------------------- | --------------------------------------- |
| `i16x8_load_extend_i8x8`  | `unsafe fn(m: *const i8) -> v128`  | Loads 8 `i8`s, sign-extends to `i16`s.  |
| `u16x8_load_extend_u8x8`  | `unsafe fn(m: *const u8) -> v128`  | Loads 8 `u8`s, zero-extends to `i16`s.  |
| `i32x4_load_extend_i16x4` | `unsafe fn(m: *const i16) -> v128` | Loads 4 `i16`s, sign-extends to `i32`s. |
| `u32x4_load_extend_u16x4` | `unsafe fn(m: *const u16) -> v128` | Loads 4 `u16`s, zero-extends to `i32`s. |
| `i64x2_load_extend_i32x2` | `unsafe fn(m: *const i32) -> v128` | Loads 2 `i32`s, sign-extends to `i64`s. |
| `u64x2_load_extend_u32x2` | `unsafe fn(m: *const u32) -> v128` | Loads 2 `u32`s, zero-extends to `i64`s. |

### Load and Splat

Loads a single scalar value and replicates it across all lanes.

| Instruction             | Signature                          | Description                           |
| ----------------------- | ---------------------------------- | ------------------------------------- |
| `v128_load8_splat`  | `unsafe fn(m: *const u8) -> v128`  | Loads 1 byte, replicates to 16 lanes. |
| `v128_load16_splat` | `unsafe fn(m: *const u16) -> v128` | Loads 2 bytes, replicates to 8 lanes. |
| `v128_load32_splat` | `unsafe fn(m: *const u32) -> v128` | Loads 4 bytes, replicates to 4 lanes. |
| `v128_load64_splat` | `unsafe fn(m: *const u64) -> v128` | Loads 8 bytes, replicates to 2 lanes. |

### Load Zero

Loads a single scalar into the lowest lane (lane 0) and sets all other bits to zero.

| Instruction            | Signature                          | Description                               |
| ---------------------- | ---------------------------------- | ----------------------------------------- |
| `v128_load32_zero` | `unsafe fn(m: *const u32) -> v128` | Loads `u32` into lane 0, zeroes the rest. |
| `v128_load64_zero` | `unsafe fn(m: *const u64) -> v128` | Loads `u64` into lane 0, zeroes the rest. |

### Lane-Specific Access

Reads or writes a specific lane `L` in a vector, leaving other lanes/memory untouched.

| Instruction             | Signature                                                   | Description                               |
| ----------------------- | ----------------------------------------------------------- | ----------------------------------------- |
| `v128_load8_lane`   | `unsafe fn<const L: usize>(v: v128, m: *const u8) -> v128`  | Loads byte from `m` into lane `L` of `v`. |
| `v128_load16_lane`  | `unsafe fn<const L: usize>(v: v128, m: *const u16) -> v128` | Loads u16 from `m` into lane `L` of `v`.  |
| `v128_load32_lane`  | `unsafe fn<const L: usize>(v: v128, m: *const u32) -> v128` | Loads u32 from `m` into lane `L` of `v`.  |
| `v128_load64_lane`  | `unsafe fn<const L: usize>(v: v128, m: *const u64) -> v128` | Loads u64 from `m` into lane `L` of `v`.  |
| `v128_store8_lane`  | `unsafe fn<const L: usize>(v: v128, m: *mut u8)`            | Stores byte from lane `L` of `v` to `m`.  |
| `v128_store16_lane` | `unsafe fn<const L: usize>(v: v128, m: *mut u16)`           | Stores u16 from lane `L` of `v` to `m`.   |
| `v128_store32_lane` | `unsafe fn<const L: usize>(v: v128, m: *mut u32)`           | Stores u32 from lane `L` of `v` to `m`.   |
| `v128_store64_lane` | `unsafe fn<const L: usize>(v: v128, m: *mut u64)`           | Stores u64 from lane `L` of `v` to `m`.   |

---

## 2. Construction and Splats

Functions to create vectors from immediate scalar values.

### Quirks

- These are `const fn`, meaning they can be used to initialize global constants.

| Instruction       | Signature                       | Description                     |
| ----------------- | ------------------------------- | ------------------------------- |
| `i8x16`       | `const fn(a0..a15: i8) -> v128` | Creates vector from 16 `i8`s.   |
| `u8x16`       | `const fn(a0..a15: u8) -> v128` | Creates vector from 16 `u8`s.   |
| `i16x8`       | `const fn(a0..a7: i16) -> v128` | Creates vector from 8 `i16`s.   |
| `u16x8`       | `const fn(a0..a7: u16) -> v128` | Creates vector from 8 `u16`s.   |
| `i32x4`       | `const fn(a0..a3: i32) -> v128` | Creates vector from 4 `i32`s.   |
| `u32x4`       | `const fn(a0..a3: u32) -> v128` | Creates vector from 4 `u32`s.   |
| `i64x2`       | `const fn(a0, a1: i64) -> v128` | Creates vector from 2 `i64`s.   |
| `u64x2`       | `const fn(a0, a1: u64) -> v128` | Creates vector from 2 `u64`s.   |
| `f32x4`       | `const fn(a0..a3: f32) -> v128` | Creates vector from 4 `f32`s.   |
| `f64x2`       | `const fn(a0, a1: f64) -> v128` | Creates vector from 2 `f64`s.   |
| `i8x16_splat` | `fn(a: i8) -> v128`             | Replicates `a` to all 16 lanes. |
| `u8x16_splat` | `fn(a: u8) -> v128`             | Replicates `a` to all 16 lanes. |
| `i16x8_splat` | `fn(a: i16) -> v128`            | Replicates `a` to all 8 lanes.  |
| `u16x8_splat` | `fn(a: u16) -> v128`            | Replicates `a` to all 8 lanes.  |
| `i32x4_splat` | `fn(a: i32) -> v128`            | Replicates `a` to all 4 lanes.  |
| `u32x4_splat` | `fn(a: u32) -> v128`            | Replicates `a` to all 4 lanes.  |
| `i64x2_splat` | `fn(a: i64) -> v128`            | Replicates `a` to all 2 lanes.  |
| `u64x2_splat` | `fn(a: u64) -> v128`            | Replicates `a` to all 2 lanes.  |
| `f32x4_splat` | `fn(a: f32) -> v128`            | Replicates `a` to all 4 lanes.  |
| `f64x2_splat` | `fn(a: f64) -> v128`            | Replicates `a` to all 2 lanes.  |

---

## 3. Shuffles and Swizzles

Rearranges lanes from two input vectors `a` (indices 0..N-1) and `b` (indices N..2N-1).

### Quirks

- Const Generics: `shuffle` functions require indices `I0`...`IN` to be known at compile time.
- Swizzle: `i8x16_swizzle` is dynamic; indices are read from a vector register. Indices out of bounds result in 0.

| Instruction         | Signature                                            | Description                              |
| ------------------- | ---------------------------------------------------- | ---------------------------------------- |
| `i8x16_shuffle` | `fn<const I0..I15: usize>(a: v128, b: v128) -> v128` | Shuffles 16 bytes. Indices must be < 32. |
| `u8x16_shuffle` | *Alias for `i8x16_shuffle*`                          | .                                        |
| `i16x8_shuffle` | `fn<const I0..I7: usize>(a: v128, b: v128) -> v128`  | Shuffles 8 shorts. Indices must be < 16. |
| `u16x8_shuffle` | *Alias for `i16x8_shuffle*`                          | .                                        |
| `i32x4_shuffle` | `fn<const I0..I3: usize>(a: v128, b: v128) -> v128`  | Shuffles 4 ints. Indices must be < 8.    |
| `u32x4_shuffle` | *Alias for `i32x4_shuffle*`                          | .                                        |
| `i64x2_shuffle` | `fn<const I0..I1: usize>(a: v128, b: v128) -> v128`  | Shuffles 2 longs. Indices must be < 4.   |
| `u64x2_shuffle` | *Alias for `i64x2_shuffle*`                          | .                                        |
| `i8x16_swizzle` | `fn(a: v128, s: v128) -> v128`                       | Rearranges `a` using indices in `s`.     |
| `u8x16_swizzle` | *Alias for `i8x16_swizzle*`                          | .                                        |

---

## 4. Lane Extraction and Replacement

Extracts a scalar from a lane or returns a new vector with one lane replaced.

### Quirks

- The lane index `N` is a `const` generic and must be within bounds for the specific type.

| Instruction              | Signature                                       | Description                   |
| ------------------------ | ----------------------------------------------- | ----------------------------- |
| `i8x16_extract_lane` | `fn<const N: usize>(a: v128) -> i8`             | Extracts `i8` from lane `N`.  |
| `u8x16_extract_lane` | `fn<const N: usize>(a: v128) -> u8`             | Extracts `u8` from lane `N`.  |
| `i8x16_replace_lane` | `fn<const N: usize>(a: v128, val: i8) -> v128`  | Replaces lane `N` with `val`. |
| `u8x16_replace_lane` | `fn<const N: usize>(a: v128, val: u8) -> v128`  | Replaces lane `N` with `val`. |
| `i16x8_extract_lane` | `fn<const N: usize>(a: v128) -> i16`            | Extracts `i16` from lane `N`. |
| `u16x8_extract_lane` | `fn<const N: usize>(a: v128) -> u16`            | Extracts `u16` from lane `N`. |
| `i16x8_replace_lane` | `fn<const N: usize>(a: v128, val: i16) -> v128` | Replaces lane `N` with `val`. |
| `u16x8_replace_lane` | `fn<const N: usize>(a: v128, val: u16) -> v128` | Replaces lane `N` with `val`. |
| `i32x4_extract_lane` | `fn<const N: usize>(a: v128) -> i32`            | Extracts `i32` from lane `N`. |
| `u32x4_extract_lane` | `fn<const N: usize>(a: v128) -> u32`            | Extracts `u32` from lane `N`. |
| `i32x4_replace_lane` | `fn<const N: usize>(a: v128, val: i32) -> v128` | Replaces lane `N` with `val`. |
| `u32x4_replace_lane` | `fn<const N: usize>(a: v128, val: u32) -> v128` | Replaces lane `N` with `val`. |
| `i64x2_extract_lane` | `fn<const N: usize>(a: v128) -> i64`            | Extracts `i64` from lane `N`. |
| `u64x2_extract_lane` | `fn<const N: usize>(a: v128) -> u64`            | Extracts `u64` from lane `N`. |
| `i64x2_replace_lane` | `fn<const N: usize>(a: v128, val: i64) -> v128` | Replaces lane `N` with `val`. |
| `u64x2_replace_lane` | `fn<const N: usize>(a: v128, val: u64) -> v128` | Replaces lane `N` with `val`. |
| `f32x4_extract_lane` | `fn<const N: usize>(a: v128) -> f32`            | Extracts `f32` from lane `N`. |
| `f32x4_replace_lane` | `fn<const N: usize>(a: v128, val: f32) -> v128` | Replaces lane `N` with `val`. |
| `f64x2_extract_lane` | `fn<const N: usize>(a: v128) -> f64`            | Extracts `f64` from lane `N`. |
| `f64x2_replace_lane` | `fn<const N: usize>(a: v128, val: f64) -> v128` | Replaces lane `N` with `val`. |

---

## 5. Integer Arithmetic (Standard)

Standard arithmetic operations with wrapping behavior on overflow.

| Instruction     | Signature                      | Description                        |
| --------------- | ------------------------------ | ---------------------------------- |
| `i8x16_add` | `fn(a: v128, b: v128) -> v128` | Lane-wise wrapping addition.       |
| `u8x16_add` | *Alias for `i8x16_add*`        | .                                  |
| `i16x8_add` | `fn(a: v128, b: v128) -> v128` | Lane-wise wrapping addition.       |
| `u16x8_add` | *Alias for `i16x8_add*`        | .                                  |
| `i32x4_add` | `fn(a: v128, b: v128) -> v128` | Lane-wise wrapping addition.       |
| `u32x4_add` | *Alias for `i32x4_add*`        | .                                  |
| `i64x2_add` | `fn(a: v128, b: v128) -> v128` | Lane-wise wrapping addition.       |
| `u64x2_add` | *Alias for `i64x2_add*`        | .                                  |
| `i8x16_sub` | `fn(a: v128, b: v128) -> v128` | Lane-wise wrapping subtraction.    |
| `u8x16_sub` | *Alias for `i8x16_sub*`        | .                                  |
| `i16x8_sub` | `fn(a: v128, b: v128) -> v128` | Lane-wise wrapping subtraction.    |
| `u16x8_sub` | *Alias for `i16x8_sub*`        | .                                  |
| `i32x4_sub` | `fn(a: v128, b: v128) -> v128` | Lane-wise wrapping subtraction.    |
| `u32x4_sub` | *Alias for `i32x4_sub*`        | .                                  |
| `i64x2_sub` | `fn(a: v128, b: v128) -> v128` | Lane-wise wrapping subtraction.    |
| `u64x2_sub` | *Alias for `i64x2_sub*`        | .                                  |
| `i16x8_mul` | `fn(a: v128, b: v128) -> v128` | Lane-wise multiplication.          |
| `u16x8_mul` | *Alias for `i16x8_mul*`        | .                                  |
| `i32x4_mul` | `fn(a: v128, b: v128) -> v128` | Lane-wise multiplication.          |
| `u32x4_mul` | *Alias for `i32x4_mul*`        | .                                  |
| `i64x2_mul` | `fn(a: v128, b: v128) -> v128` | Lane-wise multiplication.          |
| `u64x2_mul` | *Alias for `i64x2_mul*`        | .                                  |
| `i8x16_neg` | `fn(a: v128) -> v128`          | Negates values (multiplies by -1). |
| `i16x8_neg` | `fn(a: v128) -> v128`          | Negates values.                    |
| `i32x4_neg` | `fn(a: v128) -> v128`          | Negates values.                    |
| `i64x2_neg` | `fn(a: v128) -> v128`          | Negates values.                    |
| `i8x16_abs` | `fn(a: v128) -> v128`          | Lane-wise wrapping absolute value. |
| `i16x8_abs` | `fn(a: v128) -> v128`          | Lane-wise wrapping absolute value. |
| `i32x4_abs` | `fn(a: v128) -> v128`          | Lane-wise wrapping absolute value. |
| `i64x2_abs` | `fn(a: v128) -> v128`          | Lane-wise wrapping absolute value. |

---

## 6. Integer Arithmetic (Saturating)

Arithmetic that clamps results to the minimum/maximum representable value instead of wrapping.

| Instruction             | Signature                      | Description                            |
| ----------------------- | ------------------------------ | -------------------------------------- |
| `i8x16_add_sat`     | `fn(a: v128, b: v128) -> v128` | Signed saturation to `i8::MAX`.        |
| `u8x16_add_sat`     | `fn(a: v128, b: v128) -> v128` | Unsigned saturation to `u8::MAX`.      |
| `i16x8_add_sat`     | `fn(a: v128, b: v128) -> v128` | Signed saturation to `i16::MAX`.       |
| `u16x8_add_sat`     | `fn(a: v128, b: v128) -> v128` | Unsigned saturation to `u16::MAX`.     |
| `i8x16_sub_sat`     | `fn(a: v128, b: v128) -> v128` | Signed saturation to `i8::MIN`.        |
| `u8x16_sub_sat`     | `fn(a: v128, b: v128) -> v128` | Unsigned saturation to 0.              |
| `i16x8_sub_sat`     | `fn(a: v128, b: v128) -> v128` | Signed saturation to `i16::MIN`.       |
| `u16x8_sub_sat`     | `fn(a: v128, b: v128) -> v128` | Unsigned saturation to 0.              |
| `i16x8_q15mulr_sat` | `fn(a: v128, b: v128) -> v128` | Rounding multiplication in Q15 format. |

---

## 7. Extended and Pairwise Arithmetic

Operations producing wider results from narrower inputs.

| Instruction                       | Signature                      | Description                              |
| --------------------------------- | ------------------------------ | ---------------------------------------- |
| `i16x8_extadd_pairwise_i8x16` | `fn(a: v128) -> v128`          | Adds pairs of `i8`s, produces `i16`s.    |
| `i16x8_extadd_pairwise_u8x16` | `fn(a: v128) -> v128`          | Adds pairs of `u8`s, produces `i16`s.    |
| `u16x8_extadd_pairwise_u8x16` | _Alias for above_              | .                                        |
| `i32x4_extadd_pairwise_i16x8` | `fn(a: v128) -> v128`          | Adds pairs of `i16`s, produces `i32`s.   |
| `i32x4_extadd_pairwise_u16x8` | `fn(a: v128) -> v128`          | Adds pairs of `u16`s, produces `i32`s.   |
| `u32x4_extadd_pairwise_u16x8` | _Alias for above_              | .                                        |
| `i16x8_extmul_low_i8x16`      | `fn(a: v128, b: v128) -> v128` | Multiplies low `i8`s, produces `i16`s.   |
| `i16x8_extmul_high_i8x16`     | `fn(a: v128, b: v128) -> v128` | Multiplies high `i8`s, produces `i16`s.  |
| `i16x8_extmul_low_u8x16`      | `fn(a: v128, b: v128) -> v128` | Multiplies low `u8`s, produces `i16`s.   |
| `u16x8_extmul_low_u8x16`      | _Alias for above_              | .                                        |
| `i16x8_extmul_high_u8x16`     | `fn(a: v128, b: v128) -> v128` | Multiplies high `u8`s, produces `i16`s.  |
| `u16x8_extmul_high_u8x16`     | _Alias for above_              | .                                        |
| `i32x4_extmul_low_i16x8`      | `fn(a: v128, b: v128) -> v128` | Multiplies low `i16`s, produces `i32`s.  |
| `i32x4_extmul_high_i16x8`     | `fn(a: v128, b: v128) -> v128` | Multiplies high `i16`s, produces `i32`s. |
| `i32x4_extmul_low_u16x8`      | `fn(a: v128, b: v128) -> v128` | Multiplies low `u16`s, produces `i32`s.  |
| `u32x4_extmul_low_u16x8`      | _Alias for above_              | .                                        |
| `i32x4_extmul_high_u16x8`     | `fn(a: v128, b: v128) -> v128` | Multiplies high `u16`s, produces `i32`s. |
| `u32x4_extmul_high_u16x8`     | _Alias for above_              | .                                        |
| `i64x2_extmul_low_i32x4`      | `fn(a: v128, b: v128) -> v128` | Multiplies low `i32`s, produces `i64`s.  |
| `i64x2_extmul_high_i32x4`     | `fn(a: v128, b: v128) -> v128` | Multiplies high `i32`s, produces `i64`s. |
| `i64x2_extmul_low_u32x4`      | `fn(a: v128, b: v128) -> v128` | Multiplies low `u32`s, produces `i64`s.  |
| `u64x2_extmul_low_u32x4`      | _Alias for above_              | .                                        |
| `i64x2_extmul_high_u32x4`     | `fn(a: v128, b: v128) -> v128` | Multiplies high `u32`s, produces `i64`s. |
| `u64x2_extmul_high_u32x4`     | _Alias for above_              | .                                        |
| `i32x4_dot_i16x8`             | `fn(a: v128, b: v128) -> v128` | Multiplies `i16`s, adds adjacent pairs.  |

---

## 8. Min, Max, and Average

Pairwise comparison and averaging.

| Instruction      | Signature                      | Description                |
| ---------------- | ------------------------------ | -------------------------- |
| `i8x16_min`  | `fn(a: v128, b: v128) -> v128` | Signed minimum.            |
| `u8x16_min`  | `fn(a: v128, b: v128) -> v128` | Unsigned minimum.          |
| `i8x16_max`  | `fn(a: v128, b: v128) -> v128` | Signed maximum.            |
| `u8x16_max`  | `fn(a: v128, b: v128) -> v128` | Unsigned maximum.          |
| `u8x16_avgr` | `fn(a: v128, b: v128) -> v128` | Unsigned rounding average. |
| `i16x8_min`  | `fn(a: v128, b: v128) -> v128` | Signed minimum.            |
| `u16x8_min`  | `fn(a: v128, b: v128) -> v128` | Unsigned minimum.          |
| `i16x8_max`  | `fn(a: v128, b: v128) -> v128` | Signed maximum.            |
| `u16x8_max`  | `fn(a: v128, b: v128) -> v128` | Unsigned maximum.          |
| `u16x8_avgr` | `fn(a: v128, b: v128) -> v128` | Unsigned rounding average. |
| `i32x4_min`  | `fn(a: v128, b: v128) -> v128` | Signed minimum.            |
| `u32x4_min`  | `fn(a: v128, b: v128) -> v128` | Unsigned minimum.          |
| `i32x4_max`  | `fn(a: v128, b: v128) -> v128` | Signed maximum.            |
| `u32x4_max`  | `fn(a: v128, b: v128) -> v128` | Unsigned maximum.          |

---

## 9. Bitwise and Shifts

Logical operations and shifting.

### Quirks

- Shift Masking: The shift amount is a `u32`, but effectively masked to the lane width (e.g., `& 0x7` for `i8` or `& 0x1f` for `i32`).

| Instruction          | Signature                       | Description                                   |
| -------------------- | ------------------------------- | --------------------------------------------- |
| `v128_not`       | `fn(a: v128) -> v128`           | Bitwise inversion.                            |
| `v128_and`       | `fn(a: v128, b: v128) -> v128`  | Bitwise AND.                                  |
| `v128_andnot`    | `fn(a: v128, b: v128) -> v128`  | `a & (!b)`.                                   |
| `v128_or`        | `fn(a: v128, b: v128) -> v128`  | Bitwise OR.                                   |
| `v128_xor`       | `fn(a: v128, b: v128) -> v128`  | Bitwise XOR.                                  |
| `v128_bitselect` | `fn(v1, v2, c) -> v128`         | Select bits from `v1` if `c` is 1, else `v2`. |
| `i8x16_shl`      | `fn(a: v128, amt: u32) -> v128` | Left shift.                                   |
| `u8x16_shl`      | *Alias for `i8x16_shl*`         | .                                             |
| `i8x16_shr`      | `fn(a: v128, amt: u32) -> v128` | Arithmetic right shift (sign-extend).         |
| `u8x16_shr`      | `fn(a: v128, amt: u32) -> v128` | Logical right shift (zero-fill).              |
| `i16x8_shl`      | `fn(a: v128, amt: u32) -> v128` | Left shift.                                   |
| `u16x8_shl`      | *Alias for `i16x8_shl*`         | .                                             |
| `i16x8_shr`      | `fn(a: v128, amt: u32) -> v128` | Arithmetic right shift (sign-extend).         |
| `u16x8_shr`      | `fn(a: v128, amt: u32) -> v128` | Logical right shift (zero-fill).              |
| `i32x4_shl`      | `fn(a: v128, amt: u32) -> v128` | Left shift.                                   |
| `u32x4_shl`      | *Alias for `i32x4_shl*`         | .                                             |
| `i32x4_shr`      | `fn(a: v128, amt: u32) -> v128` | Arithmetic right shift (sign-extend).         |
| `u32x4_shr`      | `fn(a: v128, amt: u32) -> v128` | Logical right shift (zero-fill).              |
| `i64x2_shl`      | `fn(a: v128, amt: u32) -> v128` | Left shift.                                   |
| `u64x2_shl`      | *Alias for `i64x2_shl*`         | .                                             |
| `i64x2_shr`      | `fn(a: v128, amt: u32) -> v128` | Arithmetic right shift (sign-extend).         |
| `u64x2_shr`      | `fn(a: v128, amt: u32) -> v128` | Logical right shift (zero-fill).              |

---

## 10. Bit Analysis and Masking

Operations that analyze the bits across lanes or the entire vector.

| Instruction          | Signature                    | Description                                     |
| -------------------- | ---------------------------- | ----------------------------------------------- |
| `v128_any_true`  | `fn(a: v128) -> bool`        | Returns true if _any_ bit in the vector is set. |
| `i8x16_all_true` | `fn(a: v128) -> bool`        | Returns true if all lanes are non-zero.         |
| `u8x16_all_true` | *Alias for `i8x16_all_true*` | .                                               |
| `i16x8_all_true` | `fn(a: v128) -> bool`        | Returns true if all lanes are non-zero.         |
| `u16x8_all_true` | *Alias for `i16x8_all_true*` | .                                               |
| `i32x4_all_true` | `fn(a: v128) -> bool`        | Returns true if all lanes are non-zero.         |
| `u32x4_all_true` | *Alias for `i32x4_all_true*` | .                                               |
| `i64x2_all_true` | `fn(a: v128) -> bool`        | Returns true if all lanes are non-zero.         |
| `u64x2_all_true` | *Alias for `i64x2_all_true*` | .                                               |
| `i8x16_bitmask`  | `fn(a: v128) -> u16`         | Extracts high bit of each lane.                 |
| `u8x16_bitmask`  | *Alias for `i8x16_bitmask*`  | .                                               |
| `i16x8_bitmask`  | `fn(a: v128) -> u8`          | Extracts high bit of each lane.                 |
| `u16x8_bitmask`  | *Alias for `i16x8_bitmask*`  | .                                               |
| `i32x4_bitmask`  | `fn(a: v128) -> u8`          | Extracts high bit of each lane.                 |
| `u32x4_bitmask`  | *Alias for `i32x4_bitmask*`  | .                                               |
| `i64x2_bitmask`  | `fn(a: v128) -> u8`          | Extracts high bit of each lane.                 |
| `u64x2_bitmask`  | *Alias for `i64x2_bitmask*`  | .                                               |
| `i8x16_popcnt`   | `fn(v: v128) -> v128`        | Counts bits set to 1 in each lane.              |
| `u8x16_popcnt`   | *Alias for `i8x16_popcnt*`   | .                                               |

---

## 11. Comparisons

All comparisons return a vector where lanes are all ones (true) or all zeros (false).

| Instruction    | Signature                      | Description             |
| -------------- | ------------------------------ | ----------------------- |
| `i8x16_eq` | `fn(a: v128, b: v128) -> v128` | Check equality.         |
| `u8x16_eq` | *Alias for `i8x16_eq*`         | .                       |
| `i8x16_ne` | `fn(a: v128, b: v128) -> v128` | Check inequality.       |
| `u8x16_ne` | *Alias for `i8x16_ne*`         | .                       |
| `i8x16_lt` | `fn(a: v128, b: v128) -> v128` | Signed less than.       |
| `u8x16_lt` | `fn(a: v128, b: v128) -> v128` | Unsigned less than.     |
| `i8x16_gt` | `fn(a: v128, b: v128) -> v128` | Signed greater than.    |
| `u8x16_gt` | `fn(a: v128, b: v128) -> v128` | Unsigned greater than.  |
| `i8x16_le` | `fn(a: v128, b: v128) -> v128` | Signed less/equal.      |
| `u8x16_le` | `fn(a: v128, b: v128) -> v128` | Unsigned less/equal.    |
| `i8x16_ge` | `fn(a: v128, b: v128) -> v128` | Signed greater/equal.   |
| `u8x16_ge` | `fn(a: v128, b: v128) -> v128` | Unsigned greater/equal. |
| `i16x8_eq` | `fn(a: v128, b: v128) -> v128` | Check equality.         |
| `u16x8_eq` | *Alias for `i16x8_eq*`         | .                       |
| `i16x8_ne` | `fn(a: v128, b: v128) -> v128` | Check inequality.       |
| `u16x8_ne` | *Alias for `i16x8_ne*`         | .                       |
| `i16x8_lt` | `fn(a: v128, b: v128) -> v128` | Signed less than.       |
| `u16x8_lt` | `fn(a: v128, b: v128) -> v128` | Unsigned less than.     |
| `i16x8_gt` | `fn(a: v128, b: v128) -> v128` | Signed greater than.    |
| `u16x8_gt` | `fn(a: v128, b: v128) -> v128` | Unsigned greater than.  |
| `i16x8_le` | `fn(a: v128, b: v128) -> v128` | Signed less/equal.      |
| `u16x8_le` | `fn(a: v128, b: v128) -> v128` | Unsigned less/equal.    |
| `i16x8_ge` | `fn(a: v128, b: v128) -> v128` | Signed greater/equal.   |
| `u16x8_ge` | `fn(a: v128, b: v128) -> v128` | Unsigned greater/equal. |
| `i32x4_eq` | `fn(a: v128, b: v128) -> v128` | Check equality.         |
| `u32x4_eq` | *Alias for `i32x4_eq*`         | .                       |
| `i32x4_ne` | `fn(a: v128, b: v128) -> v128` | Check inequality.       |
| `u32x4_ne` | *Alias for `i32x4_ne*`         | .                       |
| `i32x4_lt` | `fn(a: v128, b: v128) -> v128` | Signed less than.       |
| `u32x4_lt` | `fn(a: v128, b: v128) -> v128` | Unsigned less than.     |
| `i32x4_gt` | `fn(a: v128, b: v128) -> v128` | Signed greater than.    |
| `u32x4_gt` | `fn(a: v128, b: v128) -> v128` | Unsigned greater than.  |
| `i32x4_le` | `fn(a: v128, b: v128) -> v128` | Signed less/equal.      |
| `u32x4_le` | `fn(a: v128, b: v128) -> v128` | Unsigned less/equal.    |
| `i32x4_ge` | `fn(a: v128, b: v128) -> v128` | Signed greater/equal.   |
| `u32x4_ge` | `fn(a: v128, b: v128) -> v128` | Unsigned greater/equal. |
| `i64x2_eq` | `fn(a: v128, b: v128) -> v128` | Check equality.         |
| `u64x2_eq` | *Alias for `i64x2_eq*`         | .                       |
| `i64x2_ne` | `fn(a: v128, b: v128) -> v128` | Check inequality.       |
| `u64x2_ne` | *Alias for `i64x2_ne*`         | .                       |
| `i64x2_lt` | `fn(a: v128, b: v128) -> v128` | Signed less than.       |
| `i64x2_gt` | `fn(a: v128, b: v128) -> v128` | Signed greater than.    |
| `i64x2_le` | `fn(a: v128, b: v128) -> v128` | Signed less/equal.      |
| `i64x2_ge` | `fn(a: v128, b: v128) -> v128` | Signed greater/equal.   |
| `f32x4_eq` | `fn(a: v128, b: v128) -> v128` | Check equality.         |
| `f32x4_ne` | `fn(a: v128, b: v128) -> v128` | Check inequality.       |
| `f32x4_lt` | `fn(a: v128, b: v128) -> v128` | Less than.              |
| `f32x4_gt` | `fn(a: v128, b: v128) -> v128` | Greater than.           |
| `f32x4_le` | `fn(a: v128, b: v128) -> v128` | Less/Equal.             |
| `f32x4_ge` | `fn(a: v128, b: v128) -> v128` | Greater/Equal.          |
| `f64x2_eq` | `fn(a: v128, b: v128) -> v128` | Check equality.         |
| `f64x2_ne` | `fn(a: v128, b: v128) -> v128` | Check inequality.       |
| `f64x2_lt` | `fn(a: v128, b: v128) -> v128` | Less than.              |
| `f64x2_gt` | `fn(a: v128, b: v128) -> v128` | Greater than.           |
| `f64x2_le` | `fn(a: v128, b: v128) -> v128` | Less/Equal.             |
| `f64x2_ge` | `fn(a: v128, b: v128) -> v128` | Greater/Equal.          |

---

## 12. Floating Point Arithmetic

Operations for 32-bit (`f32x4`) and 64-bit (`f64x2`) floats.

| Instruction         | Signature                      | Description                   |
| ------------------- | ------------------------------ | ----------------------------- |
| `f32x4_add`     | `fn(a: v128, b: v128) -> v128` | Addition.                     |
| `f32x4_sub`     | `fn(a: v128, b: v128) -> v128` | Subtraction.                  |
| `f32x4_mul`     | `fn(a: v128, b: v128) -> v128` | Multiplication.               |
| `f32x4_div`     | `fn(a: v128, b: v128) -> v128` | Division.                     |
| `f32x4_neg`     | `fn(a: v128) -> v128`          | Negation.                     |
| `f32x4_abs`     | `fn(a: v128) -> v128`          | Absolute value.               |
| `f32x4_sqrt`    | `fn(a: v128) -> v128`          | Square root.                  |
| `f32x4_min`     | `fn(a: v128, b: v128) -> v128` | Minimum.                      |
| `f32x4_max`     | `fn(a: v128, b: v128) -> v128` | Maximum.                      |
| `f32x4_pmin`    | `fn(a: v128, b: v128) -> v128` | Pseudo-min (`b < a ? b : a`). |
| `f32x4_pmax`    | `fn(a: v128, b: v128) -> v128` | Pseudo-max (`a < b ? b : a`). |
| `f32x4_ceil`    | `fn(a: v128) -> v128`          | Round up to integer.          |
| `f32x4_floor`   | `fn(a: v128) -> v128`          | Round down to integer.        |
| `f32x4_trunc`   | `fn(a: v128) -> v128`          | Truncate to integer.          |
| `f32x4_nearest` | `fn(a: v128) -> v128`          | Round nearest, ties to even.  |
| `f64x2_add`     | `fn(a: v128, b: v128) -> v128` | Addition.                     |
| `f64x2_sub`     | `fn(a: v128, b: v128) -> v128` | Subtraction.                  |
| `f64x2_mul`     | `fn(a: v128, b: v128) -> v128` | Multiplication.               |
| `f64x2_div`     | `fn(a: v128, b: v128) -> v128` | Division.                     |
| `f64x2_neg`     | `fn(a: v128) -> v128`          | Negation.                     |
| `f64x2_abs`     | `fn(a: v128) -> v128`          | Absolute value.               |
| `f64x2_sqrt`    | `fn(a: v128) -> v128`          | Square root.                  |
| `f64x2_min`     | `fn(a: v128, b: v128) -> v128` | Minimum.                      |
| `f64x2_max`     | `fn(a: v128, b: v128) -> v128` | Maximum.                      |
| `f64x2_pmin`    | `fn(a: v128, b: v128) -> v128` | Pseudo-min (`b < a ? b : a`). |
| `f64x2_pmax`    | `fn(a: v128, b: v128) -> v128` | Pseudo-max (`a < b ? b : a`). |
| `f64x2_ceil`    | `fn(a: v128) -> v128`          | Round up to integer.          |
| `f64x2_floor`   | `fn(a: v128) -> v128`          | Round down to integer.        |
| `f64x2_trunc`   | `fn(a: v128) -> v128`          | Truncate to integer.          |
| `f64x2_nearest` | `fn(a: v128) -> v128`          | Round nearest, ties to even.  |

---

## 13. Conversions, Widening and Narrowing

Casts between types, widening/narrowing lanes, and float<->int conversions.

| Instruction                      | Signature                      | Description                            |
| -------------------------------- | ------------------------------ | -------------------------------------- |
| `i8x16_narrow_i16x8`         | `fn(a: v128, b: v128) -> v128` | Narrow 2 `i16` vectors to `i8` (sat).  |
| `u8x16_narrow_i16x8`         | `fn(a: v128, b: v128) -> v128` | Narrow 2 `i16` vectors to `u8` (sat).  |
| `i16x8_narrow_i32x4`         | `fn(a: v128, b: v128) -> v128` | Narrow 2 `i32` vectors to `i16` (sat). |
| `u16x8_narrow_i32x4`         | `fn(a: v128, b: v128) -> v128` | Narrow 2 `i32` vectors to `u16` (sat). |
| `i16x8_extend_low_i8x16`     | `fn(a: v128) -> v128`          | Extend low `i8`s to `i16`s (sign).     |
| `i16x8_extend_high_i8x16`    | `fn(a: v128) -> v128`          | Extend high `i8`s to `i16`s (sign).    |
| `i16x8_extend_low_u8x16`     | `fn(a: v128) -> v128`          | Extend low `u8`s to `i16`s (zero).     |
| `u16x8_extend_low_u8x16`     | _Alias for above_              | .                                      |
| `i16x8_extend_high_u8x16`    | `fn(a: v128) -> v128`          | Extend high `u8`s to `i16`s (zero).    |
| `u16x8_extend_high_u8x16`    | _Alias for above_              | .                                      |
| `i32x4_extend_low_i16x8`     | `fn(a: v128) -> v128`          | Extend low `i16`s to `i32`s (sign).    |
| `i32x4_extend_high_i16x8`    | `fn(a: v128) -> v128`          | Extend high `i16`s to `i32`s (sign).   |
| `i32x4_extend_low_u16x8`     | `fn(a: v128) -> v128`          | Extend low `u16`s to `i32`s (zero).    |
| `u32x4_extend_low_u16x8`     | _Alias for above_              | .                                      |
| `i32x4_extend_high_u16x8`    | `fn(a: v128) -> v128`          | Extend high `u16`s to `i32`s (zero).   |
| `u32x4_extend_high_u16x8`    | _Alias for above_              | .                                      |
| `i64x2_extend_low_i32x4`     | `fn(a: v128) -> v128`          | Extend low `i32`s to `i64`s (sign).    |
| `i64x2_extend_high_i32x4`    | `fn(a: v128) -> v128`          | Extend high `i32`s to `i64`s (sign).   |
| `i64x2_extend_low_u32x4`     | `fn(a: v128) -> v128`          | Extend low `u32`s to `i64`s (zero).    |
| `u64x2_extend_low_u32x4`     | _Alias for above_              | .                                      |
| `i64x2_extend_high_u32x4`    | `fn(a: v128) -> v128`          | Extend high `u32`s to `i64`s (zero).   |
| `u64x2_extend_high_u32x4`    | _Alias for above_              | .                                      |
| `i32x4_trunc_sat_f32x4`      | `fn(a: v128) -> v128`          | `f32` to `i32` (saturating).           |
| `u32x4_trunc_sat_f32x4`      | `fn(a: v128) -> v128`          | `f32` to `u32` (saturating).           |
| `f32x4_convert_i32x4`        | `fn(a: v128) -> v128`          | `i32` to `f32`.                        |
| `f32x4_convert_u32x4`        | `fn(a: v128) -> v128`          | `u32` to `f32`.                        |
| `i32x4_trunc_sat_f64x2_zero` | `fn(a: v128) -> v128`          | `f64` to `i32` (low lanes, sat).       |
| `u32x4_trunc_sat_f64x2_zero` | `fn(a: v128) -> v128`          | `f64` to `u32` (low lanes, sat).       |
| `f64x2_convert_low_i32x4`    | `fn(a: v128) -> v128`          | Low `i32`s to `f64`s.                  |
| `f64x2_convert_low_u32x4`    | `fn(a: v128) -> v128`          | Low `u32`s to `f64`s.                  |
| `f32x4_demote_f64x2_zero`    | `fn(a: v128) -> v128`          | `f64` to `f32` (low lanes, zero high). |
| `f64x2_promote_low_f32x4`    | `fn(a: v128) -> v128`          | Low `f32`s to `f64`.                   |
