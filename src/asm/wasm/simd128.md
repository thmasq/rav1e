# WebAssembly SIMD ISA

### **Memory Access**

All memory intrinsics in this module perform **1-aligned** loads and stores (meaning there are no alignment requirements on the pointer, it does not need to be aligned to 16 bytes).

* **`v128.load`** - Loads a `v128` vector from the given heap address.
* **`v128.load8x8_s`** - Load eight 8-bit integers and sign-extend each to a 16-bit lane.
* **`v128.load8x8_u`** - Load eight 8-bit integers and zero-extend each to a 16-bit lane.
* **`v128.load16x4_s`** - Load four 16-bit integers and sign-extend each to a 32-bit lane.
* **`v128.load16x4_u`** - Load four 16-bit integers and zero-extend each to a 32-bit lane.
* **`v128.load32x2_s`** - Load two 32-bit integers and sign-extend each to a 64-bit lane.
* **`v128.load32x2_u`** - Load two 32-bit integers and zero-extend each to a 64-bit lane.
* **`v128.load8_splat`** - Load a single 8-bit element and splat to all lanes.
* **`v128.load16_splat`** - Load a single 16-bit element and splat to all lanes.
* **`v128.load32_splat`** - Load a single 32-bit element and splat to all lanes.
* **`v128.load64_splat`** - Load a single 64-bit element and splat to all lanes.
* **`v128.load32_zero`** - Load a 32-bit element into the lowest bits and set all other bits to zero.
* **`v128.load64_zero`** - Load a 64-bit element into the lowest bits and set all other bits to zero.
* **`v128.store`** - Stores a `v128` vector to the given heap address.
* **`v128.load{8,16,32,64}_lane`** - Loads a scalar value from memory and sets a specific lane of the vector to that value.
* **`v128.store{8,16,32,64}_lane`** - Stores a specific scalar lane from the vector into memory.

### **Constants & Creation**

* **`v128.const`** - Materializes a SIMD value from provided constant operands (implemented for all types: `i8x16`, `i16x8`, `i32x4`, `f32x4`, etc.).
* **`{type}.splat`** - Creates a vector with identical lanes replicated from a scalar value.
* *Types:* `i8x16`, `i16x8`, `i32x4`, `i64x2`, `f32x4`, `f64x2`.



### **Shuffle & Swizzle**

* **`i8x16.shuffle`** - Returns a new vector with lanes selected from two input vectors using immediate indices. Used to implement shuffles for all lane sizes (`i8`, `i16`, `i32`, `i64`).
* **`i8x16.swizzle`** - Returns a new vector with lanes selected from the first input vector using indices from the second input vector.

### **Lane Access**

* **`{type}.extract_lane_{s,u}`** - Extracts a scalar value from a specific lane index. `_s` for signed, `_u` for unsigned.
* **`{type}.replace_lane`** - Returns a new vector with a specific lane replaced by a scalar value.
* *Types:* `i8x16`, `i16x8`, `i32x4`, `i64x2`, `f32x4`, `f64x2`.



### **Integer Arithmetic & Logical**

#### **Comparisons**

* **`{type}.eq` / `ne**` - Lane-wise equal / not equal.
* **`{type}.lt_s` / `lt_u**` - Lane-wise less than (signed/unsigned).
* **`{type}.gt_s` / `gt_u**` - Lane-wise greater than (signed/unsigned).
* **`{type}.le_s` / `le_u**` - Lane-wise less than or equal (signed/unsigned).
* **`{type}.ge_s` / `ge_u**` - Lane-wise greater than or equal (signed/unsigned).
* *Types:* `i8x16`, `i16x8`, `i32x4`, `i64x2` (64-bit only has `eq`, `ne`, `lt_s`, `gt_s`, `le_s`, `ge_s`).



#### **Bitwise & Boolean**

* **`v128.not`** - Bitwise inversion.
* **`v128.and` / `or` / `xor**` - Bitwise AND, OR, XOR.
* **`v128.andnot`** - Bitwise AND of `a` and NOT `b`.
* **`v128.bitselect`** - Selects bits from `v1` or `v2` based on mask `c`.
* **`v128.any_true`** - Returns true if any bit in the vector is set.
* **`{type}.all_true`** - Returns true if all lanes are non-zero.
* **`{type}.bitmask`** - Extracts the high bit of each lane into a scalar integer mask.
* *Types for reduction:* `i8x16`, `i16x8`, `i32x4`, `i64x2`.



#### **Math Operations**

* **`{type}.abs`** - Lane-wise wrapping absolute value.
* **`{type}.neg`** - Lane-wise negation.
* **`i8x16.popcnt`** - Count population (number of set bits) per lane.
* **`{type}.add` / `sub**` - Lane-wise addition/subtraction.
* **`{type}.mul`** - Lane-wise multiplication.
* **`{type}.add_sat_{s,u}`** - Lane-wise saturating addition.
* **`{type}.sub_sat_{s,u}`** - Lane-wise saturating subtraction.
* **`{type}.min_{s,u}`** - Lane-wise minimum.
* **`{type}.max_{s,u}`** - Lane-wise maximum.
* **`{type}.avgr_u`** - Lane-wise rounding average.
* **`i16x8.q15mulr_sat_s`** - Lane-wise saturating rounding multiplication in Q15 format.
* **`i32x4.dot_i16x8_s`** - Multiply signed 16-bit integers and add adjacent pairs.

#### **Shifts**

* **`{type}.shl`** - Shift left.
* **`{type}.shr_s`** - Shift right arithmetic (sign-extend).
* **`{type}.shr_u`** - Shift right logical (zero-fill).
* *Types:* `i8x16`, `i16x8`, `i32x4`, `i64x2`.



#### **Narrowing & Extending**

* **`{type}.narrow_{src}_s` / `u**` - Narrow lanes with signed/unsigned saturation (e.g., `i8x16.narrow_i16x8_s`).
* **`{type}.extend_low_{src}_s` / `u**` - Extend low half of vector (e.g., `i16x8.extend_low_i8x16_s`).
* **`{type}.extend_high_{src}_s` / `u**` - Extend high half of vector.
* **`{type}.extadd_pairwise_{src}_{s,u}`** - Extended pairwise addition.
* **`{type}.extmul_low_{src}_{s,u}`** - Extended multiplication of low lanes.
* **`{type}.extmul_high_{src}_{s,u}`** - Extended multiplication of high lanes.

### **Floating Point (`f32x4`, `f64x2`)**

* **Arithmetic:** `add`, `sub`, `mul`, `div`, `sqrt`.
* **Rounding:** `ceil`, `floor`, `trunc`, `nearest` (rounds to even).
* **Sign:** `abs`, `neg`.
* **Comparison:** `eq`, `ne`, `lt`, `gt`, `le`, `ge`.
* **Min/Max:**
* `min`, `max` (propagates NaNs).
* `pmin`, `pmax` (pseudo-min/max, defined as `b < a ? b : a`).



### **Conversions**

* **`i32x4.trunc_sat_f32x4_{s,u}`** - Convert f32 to i32 with saturation.
* **`i32x4.trunc_sat_f64x2_{s,u}_zero`** - Convert f64 to i32 with saturation (zero high lanes).
* **`f32x4.convert_i32x4_{s,u}`** - Convert i32 to f32.
* **`f32x4.demote_f64x2_zero`** - Demote f64 to f32 (zero high lanes).
* **`f64x2.promote_low_f32x4`** - Promote low f32 lanes to f64.
* **`f64x2.convert_low_i32x4_{s,u}`** - Convert low i32 lanes to f64.
