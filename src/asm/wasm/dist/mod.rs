pub mod cdef_dist;
pub mod sse;

pub use self::cdef_dist::*;
pub use self::sse::*;

use crate::cpu_features::CpuFeatureLevel;
use crate::tiling::PlaneRegion;
use crate::util::{self, Pixel, PixelType};
use std::arch::wasm32::*;

#[inline(always)]
#[allow(clippy::let_and_return)]
pub fn get_sad<T: Pixel>(
  src: &PlaneRegion<'_, T>, dst: &PlaneRegion<'_, T>, w: usize, h: usize,
  bit_depth: usize, _cpu: CpuFeatureLevel,
) -> u32
where
  i32: util::math::CastFromPrimitive<T>,
{
  let call_rust =
    || -> u32 { crate::dist::rust::get_sad(dst, src, w, h, bit_depth, _cpu) };

  if T::type_enum() != PixelType::U8 {
    return call_rust();
  }

  let stride1 = src.plane_cfg.stride as usize;
  let stride2 = dst.plane_cfg.stride as usize;

  let mut ptr1 = src.data_ptr() as *const u8;
  let mut ptr2 = dst.data_ptr() as *const u8;

  let mut sum_u64 = 0u64;

  unsafe {
    for _ in 0..h {
      let mut x = 0;
      let mut row_sum = i32x4_splat(0);

      while x + 16 <= w {
        let a = v128_load(ptr1.add(x) as *const v128);
        let b = v128_load(ptr2.add(x) as *const v128);

        let diff = v128_or(u8x16_sub_sat(a, b), u8x16_sub_sat(b, a));

        let diff_lo = u16x8_extend_low_u8x16(diff);
        let diff_hi = u16x8_extend_high_u8x16(diff);

        let sum_lo = i32x4_extadd_pairwise_i16x8(diff_lo);
        let sum_hi = i32x4_extadd_pairwise_i16x8(diff_hi);

        row_sum = i32x4_add(row_sum, i32x4_add(sum_lo, sum_hi));
        x += 16;
      }

      sum_u64 += (i32x4_extract_lane::<0>(row_sum)
        + i32x4_extract_lane::<1>(row_sum)
        + i32x4_extract_lane::<2>(row_sum)
        + i32x4_extract_lane::<3>(row_sum)) as u64;

      while x < w {
        let a = *ptr1.add(x) as i32;
        let b = *ptr2.add(x) as i32;
        sum_u64 += (a - b).abs() as u64;
        x += 1;
      }

      ptr1 = ptr1.add(stride1);
      ptr2 = ptr2.add(stride2);
    }
  }

  sum_u64 as u32
}

#[inline(always)]
unsafe fn hadamard_butterfly(v: &mut [v128; 8]) {
  for i in (0..8).step_by(2) {
    let a = v[i];
    let b = v[i + 1];
    v[i] = i16x8_add(a, b);
    v[i + 1] = i16x8_sub(a, b);
  }
  for i in (0..8).step_by(4) {
    for j in 0..2 {
      let a = v[i + j];
      let b = v[i + j + 2];
      v[i + j] = i16x8_add(a, b);
      v[i + j + 2] = i16x8_sub(a, b);
    }
  }
  for i in 0..4 {
    let a = v[i];
    let b = v[i + 4];
    v[i] = i16x8_add(a, b);
    v[i + 4] = i16x8_sub(a, b);
  }
}

#[inline(always)]
unsafe fn transpose_8x8_i16(v: &mut [v128; 8]) {
  let t0 = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(v[0], v[1]);
  let t1 = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(v[2], v[3]);
  let t2 = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(v[4], v[5]);
  let t3 = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(v[6], v[7]);

  let t4 = i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(v[0], v[1]);
  let t5 = i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(v[2], v[3]);
  let t6 = i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(v[4], v[5]);
  let t7 = i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(v[6], v[7]);

  let m0 = i16x8_shuffle::<0, 1, 8, 9, 2, 3, 10, 11>(t0, t1);
  let m1 = i16x8_shuffle::<4, 5, 12, 13, 6, 7, 14, 15>(t0, t1);
  let m2 = i16x8_shuffle::<0, 1, 8, 9, 2, 3, 10, 11>(t2, t3);
  let m3 = i16x8_shuffle::<4, 5, 12, 13, 6, 7, 14, 15>(t2, t3);

  let m4 = i16x8_shuffle::<0, 1, 8, 9, 2, 3, 10, 11>(t4, t5);
  let m5 = i16x8_shuffle::<4, 5, 12, 13, 6, 7, 14, 15>(t4, t5);
  let m6 = i16x8_shuffle::<0, 1, 8, 9, 2, 3, 10, 11>(t6, t7);
  let m7 = i16x8_shuffle::<4, 5, 12, 13, 6, 7, 14, 15>(t6, t7);

  v[0] = i16x8_shuffle::<0, 1, 2, 3, 8, 9, 10, 11>(m0, m2);
  v[1] = i16x8_shuffle::<4, 5, 6, 7, 12, 13, 14, 15>(m0, m2);
  v[2] = i16x8_shuffle::<0, 1, 2, 3, 8, 9, 10, 11>(m1, m3);
  v[3] = i16x8_shuffle::<4, 5, 6, 7, 12, 13, 14, 15>(m1, m3);

  v[4] = i16x8_shuffle::<0, 1, 2, 3, 8, 9, 10, 11>(m4, m6);
  v[5] = i16x8_shuffle::<4, 5, 6, 7, 12, 13, 14, 15>(m4, m6);
  v[6] = i16x8_shuffle::<0, 1, 2, 3, 8, 9, 10, 11>(m5, m7);
  v[7] = i16x8_shuffle::<4, 5, 6, 7, 12, 13, 14, 15>(m5, m7);
}

#[inline(always)]
#[allow(clippy::let_and_return)]
pub fn get_satd<T: Pixel>(
  src: &PlaneRegion<'_, T>, dst: &PlaneRegion<'_, T>, w: usize, h: usize,
  bit_depth: usize, _cpu: CpuFeatureLevel,
) -> u32
where
  i32: util::math::CastFromPrimitive<T>,
{
  let call_rust =
    || -> u32 { crate::dist::rust::get_satd(dst, src, w, h, bit_depth, _cpu) };

  if T::type_enum() != PixelType::U8 || w != 8 || h != 8 {
    return call_rust();
  }

  let stride1 = src.plane_cfg.stride as usize;
  let stride2 = dst.plane_cfg.stride as usize;
  let ptr1 = src.data_ptr() as *const u8;
  let ptr2 = dst.data_ptr() as *const u8;

  let dist = unsafe {
    let mut v = [i32x4_splat(0); 8];
    for i in 0..8 {
      let r1 = v128_load64_zero(ptr1.add(i * stride1) as *const u64);
      let r2 = v128_load64_zero(ptr2.add(i * stride2) as *const u64);

      let r1_16 = u16x8_extend_low_u8x16(r1);
      let r2_16 = u16x8_extend_low_u8x16(r2);
      v[i] = i16x8_sub(r1_16, r2_16);
    }

    hadamard_butterfly(&mut v);
    transpose_8x8_i16(&mut v);
    hadamard_butterfly(&mut v);

    let mut sum = i32x4_splat(0);
    for i in 0..8 {
      let abs = i16x8_abs(v[i]);
      sum = i32x4_add(sum, i32x4_extadd_pairwise_i16x8(abs));
    }

    let total = (i32x4_extract_lane::<0>(sum)
      + i32x4_extract_lane::<1>(sum)
      + i32x4_extract_lane::<2>(sum)
      + i32x4_extract_lane::<3>(sum)) as u32;

    (total + 4) >> 3
  };

  dist
}
