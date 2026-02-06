use crate::activity::apply_ssim_boost;
use crate::cpu_features::CpuFeatureLevel;
use crate::tiling::PlaneRegion;
use crate::util::{self, Pixel, PixelType};
use std::arch::wasm32::*;

#[inline(always)]
unsafe fn sum128_i32(v: v128) -> i32 {
  let a = i32x4_extract_lane::<0>(v);
  let b = i32x4_extract_lane::<1>(v);
  let c = i32x4_extract_lane::<2>(v);
  let d = i32x4_extract_lane::<3>(v);
  a.wrapping_add(b).wrapping_add(c).wrapping_add(d)
}

#[inline(always)]
unsafe fn cdef_dist_8x8_simd<T: Pixel>(
  src: &PlaneRegion<'_, T>, dst: &PlaneRegion<'_, T>,
) -> (u32, u32, u32, u32, u32) {
  let mut sum_s: i32 = 0;
  let mut sum_d: i32 = 0;
  let mut sum_s2: i32 = 0;
  let mut sum_d2: i32 = 0;
  let mut sum_sd: i32 = 0;

  let src_ptr = src.data_ptr() as *const u16;
  let dst_ptr = dst.data_ptr() as *const u16;
  let src_stride = src.plane_cfg.stride as usize;
  let dst_stride = dst.plane_cfg.stride as usize;

  for i in 0..8 {
    let s_row = if T::type_enum() == PixelType::U8 {
      let s_u8 = v128_load64_zero(src_ptr.add(i * src_stride) as *const u64);
      u16x8_extend_low_u8x16(s_u8)
    } else {
      v128_load(src_ptr.add(i * src_stride) as *const v128)
    };

    let d_row = if T::type_enum() == PixelType::U8 {
      let d_u8 = v128_load64_zero(dst_ptr.add(i * dst_stride) as *const u64);
      u16x8_extend_low_u8x16(d_u8)
    } else {
      v128_load(dst_ptr.add(i * dst_stride) as *const v128)
    };

    let ones = i16x8_splat(1);
    let s_sum_row = i32x4_dot_i16x8(s_row, ones);
    let d_sum_row = i32x4_dot_i16x8(d_row, ones);
    sum_s += sum128_i32(s_sum_row);
    sum_d += sum128_i32(d_sum_row);

    let s2_row = i32x4_dot_i16x8(s_row, s_row);
    let d2_row = i32x4_dot_i16x8(d_row, d_row);
    let sd_row = i32x4_dot_i16x8(s_row, d_row);

    sum_s2 += sum128_i32(s2_row);
    sum_d2 += sum128_i32(d2_row);
    sum_sd += sum128_i32(sd_row);
  }

  (sum_s as u32, sum_d as u32, sum_s2 as u32, sum_d2 as u32, sum_sd as u32)
}

#[inline(always)]
unsafe fn cdef_dist_4x4_simd<T: Pixel>(
  src: &PlaneRegion<'_, T>, dst: &PlaneRegion<'_, T>,
) -> (u32, u32, u32, u32, u32) {
  let mut sum_s: i32 = 0;
  let mut sum_d: i32 = 0;
  let mut sum_s2: i32 = 0;
  let mut sum_d2: i32 = 0;
  let mut sum_sd: i32 = 0;

  let src_ptr = src.data_ptr() as *const u16;
  let dst_ptr = dst.data_ptr() as *const u16;
  let src_stride = src.plane_cfg.stride as usize;
  let dst_stride = dst.plane_cfg.stride as usize;

  for i in 0..4 {
    let s_row = if T::type_enum() == PixelType::U8 {
      let s_u8 = v128_load32_zero(src_ptr.add(i * src_stride) as *const u32);
      u16x8_extend_low_u8x16(s_u8)
    } else {
      v128_load64_zero(src_ptr.add(i * src_stride) as *const u64)
    };

    let d_row = if T::type_enum() == PixelType::U8 {
      let d_u8 = v128_load32_zero(dst_ptr.add(i * dst_stride) as *const u32);
      u16x8_extend_low_u8x16(d_u8)
    } else {
      v128_load64_zero(dst_ptr.add(i * dst_stride) as *const u64)
    };

    let ones = i16x8_splat(1);
    let s_sum_row = i32x4_dot_i16x8(s_row, ones);
    let d_sum_row = i32x4_dot_i16x8(d_row, ones);
    sum_s += sum128_i32(s_sum_row);
    sum_d += sum128_i32(d_sum_row);

    let s2_row = i32x4_dot_i16x8(s_row, s_row);
    let d2_row = i32x4_dot_i16x8(d_row, d_row);
    let sd_row = i32x4_dot_i16x8(s_row, d_row);

    sum_s2 += sum128_i32(s2_row);
    sum_d2 += sum128_i32(d2_row);
    sum_sd += sum128_i32(sd_row);
  }

  (sum_s as u32, sum_d as u32, sum_s2 as u32, sum_d2 as u32, sum_sd as u32)
}

#[allow(clippy::let_and_return)]
pub fn cdef_dist_kernel<T: Pixel>(
  src: &PlaneRegion<'_, T>, dst: &PlaneRegion<'_, T>, w: usize, h: usize,
  bit_depth: usize, _cpu: CpuFeatureLevel,
) -> u32
where
  u32: util::math::CastFromPrimitive<T>,
{
  debug_assert!(src.plane_cfg.xdec == 0);
  debug_assert!(src.plane_cfg.ydec == 0);
  debug_assert!(dst.plane_cfg.xdec == 0);
  debug_assert!(dst.plane_cfg.ydec == 0);

  let call_rust = || -> u32 {
    crate::dist::rust::cdef_dist_kernel(dst, src, w, h, bit_depth, _cpu)
  };

  let (sum_s, sum_d, sum_s2, sum_d2, sum_sd) = unsafe {
    if w == 8 && h == 8 {
      cdef_dist_8x8_simd(src, dst)
    } else if w == 4 && h == 4 {
      cdef_dist_4x4_simd(src, dst)
    } else {
      return call_rust();
    }
  };

  let sse = (sum_d2 as u64) + (sum_s2 as u64) - 2 * (sum_sd as u64);

  let sum_s = sum_s as u64;
  let sum_d = sum_d as u64;

  let (svar, dvar) = if w == 8 && h == 8 {
    let svar =
      (sum_s2 as u64).saturating_sub((sum_s * sum_s + 32) >> 6) as u32;
    let dvar =
      (sum_d2 as u64).saturating_sub((sum_d * sum_d + 32) >> 6) as u32;
    (svar, dvar)
  } else {
    let svar = (sum_s2 as u64).saturating_sub((sum_s * sum_s + 8) >> 4) as u32;
    let dvar = (sum_d2 as u64).saturating_sub((sum_d * sum_d + 8) >> 4) as u32;
    (svar << 2, dvar << 2)
  };

  let dist = apply_ssim_boost(sse as u32, svar, dvar, bit_depth);

  dist
}
