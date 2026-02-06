pub mod cdef_dist;
pub mod sse;

pub use self::cdef_dist::*;
pub use self::sse::*;

use crate::cpu_features::CpuFeatureLevel;
use crate::tiling::PlaneRegion;
use crate::util::{self, Pixel, PixelType};
use std::arch::wasm32::*;

#[inline(always)]
unsafe fn sum128_i32(v: v128) -> u32 {
  let a = i32x4_extract_lane::<0>(v);
  let b = i32x4_extract_lane::<1>(v);
  let c = i32x4_extract_lane::<2>(v);
  let d = i32x4_extract_lane::<3>(v);
  a.wrapping_add(b).wrapping_add(c).wrapping_add(d) as u32
}

#[inline(always)]
unsafe fn load_pixels<T: Pixel, const W: usize>(ptr: *const u8) -> v128 {
  match W {
    4 => {
      if T::type_enum() == PixelType::U8 {
        let val = v128_load32_zero(ptr as *const u32);
        u16x8_extend_low_u8x16(val)
      } else {
        v128_load64_zero(ptr as *const u64)
      }
    }
    8 => {
      if T::type_enum() == PixelType::U8 {
        let val = v128_load64_zero(ptr as *const u64);
        u16x8_extend_low_u8x16(val)
      } else {
        v128_load(ptr as *const v128)
      }
    }
    _ => {
      if T::type_enum() == PixelType::U8 {
        v128_load(ptr as *const v128)
      } else {
        unreachable!()
      }
    }
  }
}

#[inline(always)]
unsafe fn sad_wxh<T: Pixel, const W: usize, const H: usize>(
  src: &PlaneRegion<'_, T>, dst: &PlaneRegion<'_, T>,
) -> u32 {
  let src_ptr = src.data_ptr();
  let dst_ptr = dst.data_ptr();
  let src_stride = src.plane_cfg.stride as usize;
  let dst_stride = dst.plane_cfg.stride as usize;

  let sum;

  if T::type_enum() == PixelType::U8 {
    let mut sum_lanes = i32x4_splat(0);

    for i in 0..H {
      let s_row = (src_ptr as *const u8).add(i * src_stride);
      let d_row = (dst_ptr as *const u8).add(i * dst_stride);

      if W == 4 {
        let s = v128_load32_zero(s_row as *const u32);
        let d = v128_load32_zero(d_row as *const u32);
        let diff = v128_or(u8x16_sub_sat(s, d), u8x16_sub_sat(d, s));
        let diff_lo = u16x8_extend_low_u8x16(diff);
        sum_lanes = i32x4_add(sum_lanes, i32x4_extadd_pairwise_i16x8(diff_lo));
      } else if W == 8 {
        let s = v128_load64_zero(s_row as *const u64);
        let d = v128_load64_zero(d_row as *const u64);
        let diff = v128_or(u8x16_sub_sat(s, d), u8x16_sub_sat(d, s));
        let diff_lo = u16x8_extend_low_u8x16(diff);
        sum_lanes = i32x4_add(sum_lanes, i32x4_extadd_pairwise_i16x8(diff_lo));
      } else {
        for j in (0..W).step_by(16) {
          let s = v128_load(s_row.add(j) as *const v128);
          let d = v128_load(d_row.add(j) as *const v128);
          let diff = v128_or(u8x16_sub_sat(s, d), u8x16_sub_sat(d, s));
          let diff_lo = u16x8_extend_low_u8x16(diff);
          let diff_hi = u16x8_extend_high_u8x16(diff);
          sum_lanes =
            i32x4_add(sum_lanes, i32x4_extadd_pairwise_i16x8(diff_lo));
          sum_lanes =
            i32x4_add(sum_lanes, i32x4_extadd_pairwise_i16x8(diff_hi));
        }
      }
    }
    sum = sum128_i32(sum_lanes);
  } else {
    let mut sum_lanes = i32x4_splat(0);

    for i in 0..H {
      let s_row = (src_ptr as *const u8).add(i * src_stride);
      let d_row = (dst_ptr as *const u8).add(i * dst_stride);

      if W == 4 {
        let s = v128_load64_zero(s_row as *const u64);
        let d = v128_load64_zero(d_row as *const u64);
        let diff = i16x8_sub(s, d);
        let abs_diff = i16x8_abs(diff);
        sum_lanes =
          i32x4_add(sum_lanes, i32x4_extadd_pairwise_i16x8(abs_diff));
      } else if W == 8 {
        let s = v128_load(s_row as *const v128);
        let d = v128_load(d_row as *const v128);
        let diff = i16x8_sub(s, d);
        let abs_diff = i16x8_abs(diff);
        sum_lanes =
          i32x4_add(sum_lanes, i32x4_extadd_pairwise_i16x8(abs_diff));
      } else {
        for j in (0..W).step_by(8) {
          let s = v128_load(s_row.add(j * 2) as *const v128);
          let d = v128_load(d_row.add(j * 2) as *const v128);
          let diff = i16x8_sub(s, d);
          let abs_diff = i16x8_abs(diff);
          sum_lanes =
            i32x4_add(sum_lanes, i32x4_extadd_pairwise_i16x8(abs_diff));
        }
      }
    }
    sum = sum128_i32(sum_lanes);
  }

  sum
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
unsafe fn hadamard_4x4(v: &mut [v128; 4]) {
  let p0 = i16x8_add(v[0], v[2]);
  let p1 = i16x8_add(v[1], v[3]);
  let p2 = i16x8_sub(v[0], v[2]);
  let p3 = i16x8_sub(v[1], v[3]);

  v[0] = i16x8_add(p0, p1);
  v[1] = i16x8_sub(p0, p1);
  v[2] = i16x8_add(p2, p3);
  v[3] = i16x8_sub(p2, p3);
}

#[inline(always)]
unsafe fn transpose_4x4_packed(v: &mut [v128; 4]) {
  let t0 = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(v[0], v[1]);
  let t1 = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(v[2], v[3]);

  let m0 = i32x4_shuffle::<0, 2, 1, 3>(t0, t0);
  let m1 = i32x4_shuffle::<0, 2, 1, 3>(t1, t1);

  v[0] = i32x4_shuffle::<0, 4, 1, 5>(m0, m1);
  v[1] = i32x4_shuffle::<1, 5, 2, 6>(m0, m1);
  v[2] = i32x4_shuffle::<2, 6, 3, 7>(m0, m1);
  v[3] = i32x4_shuffle::<3, 7, 0, 0>(m0, m1);
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
unsafe fn satd_wxh<T: Pixel, const W: usize, const H: usize>(
  src: &PlaneRegion<'_, T>, dst: &PlaneRegion<'_, T>,
) -> u32 {
  let src_ptr = src.data_ptr();
  let dst_ptr = dst.data_ptr();
  let src_stride = src.plane_cfg.stride as usize;
  let dst_stride = dst.plane_cfg.stride as usize;

  let mut sum = i32x4_splat(0);

  if W >= 8 && H >= 8 {
    for i in (0..H).step_by(8) {
      for j in (0..W).step_by(8) {
        let mut v = [i32x4_splat(0); 8];
        for k in 0..8 {
          let s_row = (src_ptr as *const u8)
            .add((i + k) * src_stride + j * std::mem::size_of::<T>());
          let d_row = (dst_ptr as *const u8)
            .add((i + k) * dst_stride + j * std::mem::size_of::<T>());
          let s = load_pixels::<T, 8>(s_row);
          let d = load_pixels::<T, 8>(d_row);
          v[k] = i16x8_sub(s, d);
        }

        hadamard_butterfly(&mut v);
        transpose_8x8_i16(&mut v);
        hadamard_butterfly(&mut v);

        for k in 0..8 {
          sum = i32x4_add(sum, i32x4_extadd_pairwise_i16x8(i16x8_abs(v[k])));
        }
      }
    }
    let total = sum128_i32(sum);
    (total + 2) >> 2
  } else {
    for i in (0..H).step_by(4) {
      for j in (0..W).step_by(4) {
        let mut v = [i32x4_splat(0); 4];
        for k in 0..4 {
          let s_row = (src_ptr as *const u8)
            .add((i + k) * src_stride + j * std::mem::size_of::<T>());
          let d_row = (dst_ptr as *const u8)
            .add((i + k) * dst_stride + j * std::mem::size_of::<T>());
          let s = load_pixels::<T, 4>(s_row);
          let d = load_pixels::<T, 4>(d_row);
          v[k] = i16x8_sub(s, d);
        }

        hadamard_4x4(&mut v);
        transpose_4x4_packed(&mut v);
        hadamard_4x4(&mut v);

        for k in 0..4 {
          let mask = u32x4(0xFFFF, 0xFFFF, 0, 0);
          let valid = v128_and(v[k], mask);
          sum = i32x4_add(sum, i32x4_extadd_pairwise_i16x8(i16x8_abs(valid)));
        }
      }
    }
    let total = sum128_i32(sum);
    (total + 2) >> 2
  }
}

macro_rules! dist_dispatch {
    ($w:ident, $h:ident, $bit_depth:ident, $cpu:ident, $src:ident, $dst:ident, $simd_func:ident, $fallback_func:ident, $T:ty, $(($W:literal, $H:literal)),*) => {
        match ($w, $h) {
            $(
                ($W, $H) => $simd_func::<$T, $W, $H>($src, $dst),
            )*
            _ => crate::dist::rust::$fallback_func($dst, $src, $w, $h, $bit_depth, $cpu),
        }
    };
}

#[inline(always)]
#[allow(clippy::let_and_return)]
pub fn get_sad<T: Pixel>(
  src: &PlaneRegion<'_, T>, dst: &PlaneRegion<'_, T>, w: usize, h: usize,
  bit_depth: usize, _cpu: CpuFeatureLevel,
) -> u32
where
  i32: util::math::CastFromPrimitive<T>,
{
  let call_dist = unsafe {
    dist_dispatch!(
      w,
      h,
      bit_depth,
      _cpu,
      src,
      dst,
      sad_wxh,
      get_sad,
      T,
      (4, 4),
      (4, 8),
      (4, 16),
      (8, 4),
      (8, 8),
      (8, 16),
      (8, 32),
      (16, 4),
      (16, 8),
      (16, 16),
      (16, 32),
      (16, 64),
      (32, 8),
      (32, 16),
      (32, 32),
      (32, 64),
      (64, 16),
      (64, 32),
      (64, 64),
      (64, 128),
      (128, 64),
      (128, 128)
    )
  };
  call_dist
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
  let call_dist = unsafe {
    dist_dispatch!(
      w,
      h,
      bit_depth,
      _cpu,
      src,
      dst,
      satd_wxh,
      get_satd,
      T,
      (4, 4),
      (4, 8),
      (4, 16),
      (8, 4),
      (8, 8),
      (8, 16),
      (8, 32),
      (16, 4),
      (16, 8),
      (16, 16),
      (16, 32),
      (16, 64),
      (32, 8),
      (32, 16),
      (32, 32),
      (32, 64),
      (64, 16),
      (64, 32),
      (64, 64),
      (64, 128),
      (128, 64),
      (128, 128)
    )
  };
  call_dist
}
