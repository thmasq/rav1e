pub mod cdef_dist;
pub mod sse;

pub use self::cdef_dist::*;
pub use self::sse::*;

use crate::cpu_features::CpuFeatureLevel;
use crate::partition::BlockSize;
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
    if W == 4 {
      let mut sum0 = i32x4_splat(0);
      let mut sum1 = i32x4_splat(0);

      for i in (0..H).step_by(2) {
        let s_row0 = (src_ptr as *const u8).add(i * src_stride);
        let d_row0 = (dst_ptr as *const u8).add(i * dst_stride);
        let s_row1 = s_row0.add(src_stride);
        let d_row1 = d_row0.add(dst_stride);

        let s0 = v128_load32_zero(s_row0 as *const u32);
        let d0 = v128_load32_zero(d_row0 as *const u32);
        let diff0 = v128_or(u8x16_sub_sat(s0, d0), u8x16_sub_sat(d0, s0));
        sum0 = i32x4_add(
          sum0,
          i32x4_extadd_pairwise_i16x8(u16x8_extend_low_u8x16(diff0)),
        );

        let s1 = v128_load32_zero(s_row1 as *const u32);
        let d1 = v128_load32_zero(d_row1 as *const u32);
        let diff1 = v128_or(u8x16_sub_sat(s1, d1), u8x16_sub_sat(d1, s1));
        sum1 = i32x4_add(
          sum1,
          i32x4_extadd_pairwise_i16x8(u16x8_extend_low_u8x16(diff1)),
        );
      }
      sum = sum128_i32(i32x4_add(sum0, sum1));
    } else if W == 8 {
      let mut sum0 = i32x4_splat(0);
      let mut sum1 = i32x4_splat(0);

      for i in (0..H).step_by(2) {
        let s_row0 = (src_ptr as *const u8).add(i * src_stride);
        let d_row0 = (dst_ptr as *const u8).add(i * dst_stride);
        let s_row1 = s_row0.add(src_stride);
        let d_row1 = d_row0.add(dst_stride);

        let s0 = v128_load64_zero(s_row0 as *const u64);
        let d0 = v128_load64_zero(d_row0 as *const u64);
        let diff0 = v128_or(u8x16_sub_sat(s0, d0), u8x16_sub_sat(d0, s0));
        sum0 = i32x4_add(
          sum0,
          i32x4_extadd_pairwise_i16x8(u16x8_extend_low_u8x16(diff0)),
        );

        let s1 = v128_load64_zero(s_row1 as *const u64);
        let d1 = v128_load64_zero(d_row1 as *const u64);
        let diff1 = v128_or(u8x16_sub_sat(s1, d1), u8x16_sub_sat(d1, s1));
        sum1 = i32x4_add(
          sum1,
          i32x4_extadd_pairwise_i16x8(u16x8_extend_low_u8x16(diff1)),
        );
      }
      sum = sum128_i32(i32x4_add(sum0, sum1));
    } else {
      let mut sum_lo = i32x4_splat(0);
      let mut sum_hi = i32x4_splat(0);

      for i in 0..H {
        let s_row = (src_ptr as *const u8).add(i * src_stride);
        let d_row = (dst_ptr as *const u8).add(i * dst_stride);

        for j in (0..W).step_by(16) {
          let s = v128_load(s_row.add(j) as *const v128);
          let d = v128_load(d_row.add(j) as *const v128);

          let diff = v128_or(u8x16_sub_sat(s, d), u8x16_sub_sat(d, s));
          sum_lo = i32x4_add(
            sum_lo,
            i32x4_extadd_pairwise_i16x8(u16x8_extend_low_u8x16(diff)),
          );
          sum_hi = i32x4_add(
            sum_hi,
            i32x4_extadd_pairwise_i16x8(u16x8_extend_high_u8x16(diff)),
          );
        }
      }
      sum = sum128_i32(i32x4_add(sum_lo, sum_hi));
    }
  } else {
    if W == 4 {
      let mut sum0 = i32x4_splat(0);
      let mut sum1 = i32x4_splat(0);

      for i in (0..H).step_by(2) {
        let s_row0 = (src_ptr as *const u8).add(i * src_stride);
        let d_row0 = (dst_ptr as *const u8).add(i * dst_stride);
        let s_row1 = s_row0.add(src_stride);
        let d_row1 = d_row0.add(dst_stride);

        let s0 = v128_load64_zero(s_row0 as *const u64);
        let d0 = v128_load64_zero(d_row0 as *const u64);
        sum0 = i32x4_add(
          sum0,
          i32x4_extadd_pairwise_i16x8(i16x8_abs(i16x8_sub(s0, d0))),
        );

        let s1 = v128_load64_zero(s_row1 as *const u64);
        let d1 = v128_load64_zero(d_row1 as *const u64);
        sum1 = i32x4_add(
          sum1,
          i32x4_extadd_pairwise_i16x8(i16x8_abs(i16x8_sub(s1, d1))),
        );
      }
      sum = sum128_i32(i32x4_add(sum0, sum1));
    } else if W == 8 {
      let mut sum0 = i32x4_splat(0);
      let mut sum1 = i32x4_splat(0);

      for i in (0..H).step_by(2) {
        let s_row0 = (src_ptr as *const u8).add(i * src_stride);
        let d_row0 = (dst_ptr as *const u8).add(i * dst_stride);
        let s_row1 = s_row0.add(src_stride);
        let d_row1 = d_row0.add(dst_stride);

        let s0 = v128_load(s_row0 as *const v128);
        let d0 = v128_load(d_row0 as *const v128);
        sum0 = i32x4_add(
          sum0,
          i32x4_extadd_pairwise_i16x8(i16x8_abs(i16x8_sub(s0, d0))),
        );

        let s1 = v128_load(s_row1 as *const v128);
        let d1 = v128_load(d_row1 as *const v128);
        sum1 = i32x4_add(
          sum1,
          i32x4_extadd_pairwise_i16x8(i16x8_abs(i16x8_sub(s1, d1))),
        );
      }
      sum = sum128_i32(i32x4_add(sum0, sum1));
    } else {
      let mut sum0 = i32x4_splat(0);
      let mut sum1 = i32x4_splat(0);

      for i in (0..H).step_by(2) {
        let s_row0 = (src_ptr as *const u8).add(i * src_stride);
        let d_row0 = (dst_ptr as *const u8).add(i * dst_stride);
        let s_row1 = s_row0.add(src_stride);
        let d_row1 = d_row0.add(dst_stride);

        for j in (0..W).step_by(8) {
          let offset = j * 2;

          let s0 = v128_load(s_row0.add(offset) as *const v128);
          let d0 = v128_load(d_row0.add(offset) as *const v128);
          let diff0 = i16x8_abs(i16x8_sub(s0, d0));
          sum0 = i32x4_add(sum0, i32x4_extadd_pairwise_i16x8(diff0));

          let s1 = v128_load(s_row1.add(offset) as *const v128);
          let d1 = v128_load(d_row1.add(offset) as *const v128);
          let diff1 = i16x8_abs(i16x8_sub(s1, d1));
          sum1 = i32x4_add(sum1, i32x4_extadd_pairwise_i16x8(diff1));
        }
      }
      sum = sum128_i32(i32x4_add(sum0, sum1));
    }
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

        let s_base = (src_ptr as *const u8)
          .add(i * src_stride + j * std::mem::size_of::<T>());
        let d_base = (dst_ptr as *const u8)
          .add(i * dst_stride + j * std::mem::size_of::<T>());

        for k in 0..8 {
          let s_row = s_base.add(k * src_stride);
          let d_row = d_base.add(k * dst_stride);
          let s = load_pixels::<T, 8>(s_row);
          let d = load_pixels::<T, 8>(d_row);
          v[k] = i16x8_sub(s, d);
        }

        hadamard_butterfly(&mut v);
        transpose_8x8_i16(&mut v);
        hadamard_butterfly(&mut v);

        let abs0 = i32x4_extadd_pairwise_i16x8(i16x8_abs(v[0]));
        let abs1 = i32x4_extadd_pairwise_i16x8(i16x8_abs(v[1]));
        let abs2 = i32x4_extadd_pairwise_i16x8(i16x8_abs(v[2]));
        let abs3 = i32x4_extadd_pairwise_i16x8(i16x8_abs(v[3]));
        let abs4 = i32x4_extadd_pairwise_i16x8(i16x8_abs(v[4]));
        let abs5 = i32x4_extadd_pairwise_i16x8(i16x8_abs(v[5]));
        let abs6 = i32x4_extadd_pairwise_i16x8(i16x8_abs(v[6]));
        let abs7 = i32x4_extadd_pairwise_i16x8(i16x8_abs(v[7]));

        let sum01 = i32x4_add(abs0, abs1);
        let sum23 = i32x4_add(abs2, abs3);
        let sum45 = i32x4_add(abs4, abs5);
        let sum67 = i32x4_add(abs6, abs7);

        let sum0123 = i32x4_add(sum01, sum23);
        let sum4567 = i32x4_add(sum45, sum67);

        sum = i32x4_add(sum, i32x4_add(sum0123, sum4567));
      }
    }
  } else {
    let mask = i32x4(-1, -1, 0, 0);

    for i in (0..H).step_by(4) {
      for j in (0..W).step_by(4) {
        let mut v = [i32x4_splat(0); 4];

        let s_base = (src_ptr as *const u8)
          .add(i * src_stride + j * std::mem::size_of::<T>());
        let d_base = (dst_ptr as *const u8)
          .add(i * dst_stride + j * std::mem::size_of::<T>());

        for k in 0..4 {
          let s_row = s_base.add(k * src_stride);
          let d_row = d_base.add(k * dst_stride);
          let s = load_pixels::<T, 4>(s_row);
          let d = load_pixels::<T, 4>(d_row);
          v[k] = i16x8_sub(s, d);
        }

        hadamard_4x4(&mut v);
        transpose_4x4_packed(&mut v);
        hadamard_4x4(&mut v);

        let abs0 =
          i32x4_extadd_pairwise_i16x8(i16x8_abs(v128_and(v[0], mask)));
        let abs1 =
          i32x4_extadd_pairwise_i16x8(i16x8_abs(v128_and(v[1], mask)));
        let abs2 =
          i32x4_extadd_pairwise_i16x8(i16x8_abs(v128_and(v[2], mask)));
        let abs3 =
          i32x4_extadd_pairwise_i16x8(i16x8_abs(v128_and(v[3], mask)));

        let sum01 = i32x4_add(abs0, abs1);
        let sum23 = i32x4_add(abs2, abs3);

        sum = i32x4_add(sum, i32x4_add(sum01, sum23));
      }
    }
  }

  let total = sum128_i32(sum);
  (total + 2) >> 2
}

type DistFn<T> = unsafe fn(&PlaneRegion<'_, T>, &PlaneRegion<'_, T>) -> u32;

const DIST_FNS_LENGTH: usize = 32;

#[inline(always)]
const fn to_index(bsize: BlockSize) -> usize {
  bsize as usize & (DIST_FNS_LENGTH - 1)
}

macro_rules! generate_dist_table {
  ($T:ty, $method:ident) => {{
    let mut out: [Option<DistFn<$T>>; DIST_FNS_LENGTH] =
      [None; DIST_FNS_LENGTH];
    use BlockSize::*;
    out[BLOCK_4X4 as usize] = Some($method::<$T, 4, 4>);
    out[BLOCK_4X8 as usize] = Some($method::<$T, 4, 8>);
    out[BLOCK_4X16 as usize] = Some($method::<$T, 4, 16>);
    out[BLOCK_8X4 as usize] = Some($method::<$T, 8, 4>);
    out[BLOCK_8X8 as usize] = Some($method::<$T, 8, 8>);
    out[BLOCK_8X16 as usize] = Some($method::<$T, 8, 16>);
    out[BLOCK_8X32 as usize] = Some($method::<$T, 8, 32>);
    out[BLOCK_16X4 as usize] = Some($method::<$T, 16, 4>);
    out[BLOCK_16X8 as usize] = Some($method::<$T, 16, 8>);
    out[BLOCK_16X16 as usize] = Some($method::<$T, 16, 16>);
    out[BLOCK_16X32 as usize] = Some($method::<$T, 16, 32>);
    out[BLOCK_16X64 as usize] = Some($method::<$T, 16, 64>);
    out[BLOCK_32X8 as usize] = Some($method::<$T, 32, 8>);
    out[BLOCK_32X16 as usize] = Some($method::<$T, 32, 16>);
    out[BLOCK_32X32 as usize] = Some($method::<$T, 32, 32>);
    out[BLOCK_32X64 as usize] = Some($method::<$T, 32, 64>);
    out[BLOCK_64X16 as usize] = Some($method::<$T, 64, 16>);
    out[BLOCK_64X32 as usize] = Some($method::<$T, 64, 32>);
    out[BLOCK_64X64 as usize] = Some($method::<$T, 64, 64>);
    out[BLOCK_64X128 as usize] = Some($method::<$T, 64, 128>);
    out[BLOCK_128X64 as usize] = Some($method::<$T, 128, 64>);
    out[BLOCK_128X128 as usize] = Some($method::<$T, 128, 128>);
    out
  }};
}

static SAD_FNS_U8: [Option<DistFn<u8>>; DIST_FNS_LENGTH] =
  generate_dist_table!(u8, sad_wxh);
static SAD_FNS_U16: [Option<DistFn<u16>>; DIST_FNS_LENGTH] =
  generate_dist_table!(u16, sad_wxh);

static SATD_FNS_U8: [Option<DistFn<u8>>; DIST_FNS_LENGTH] =
  generate_dist_table!(u8, satd_wxh);
static SATD_FNS_U16: [Option<DistFn<u16>>; DIST_FNS_LENGTH] =
  generate_dist_table!(u16, satd_wxh);

#[inline(always)]
#[allow(clippy::let_and_return)]
pub fn get_sad<T: Pixel>(
  src: &PlaneRegion<'_, T>, dst: &PlaneRegion<'_, T>, w: usize, h: usize,
  bit_depth: usize, _cpu: CpuFeatureLevel,
) -> u32
where
  i32: util::math::CastFromPrimitive<T>,
{
  let bsize_opt = BlockSize::from_width_and_height_opt(w, h);

  if let Ok(bsize) = bsize_opt {
    unsafe {
      if T::type_enum() == PixelType::U8 {
        if let Some(func) = SAD_FNS_U8[to_index(bsize)] {
          let src_u8 = std::mem::transmute(src);
          let dst_u8 = std::mem::transmute(dst);
          return func(src_u8, dst_u8);
        }
      } else {
        if let Some(func) = SAD_FNS_U16[to_index(bsize)] {
          let src_u16 = std::mem::transmute(src);
          let dst_u16 = std::mem::transmute(dst);
          return func(src_u16, dst_u16);
        }
      }
    }
  }

  crate::dist::rust::get_sad(dst, src, w, h, bit_depth, _cpu)
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
  let bsize_opt = BlockSize::from_width_and_height_opt(w, h);

  if let Ok(bsize) = bsize_opt {
    unsafe {
      if T::type_enum() == PixelType::U8 {
        if let Some(func) = SATD_FNS_U8[to_index(bsize)] {
          let src_u8 = std::mem::transmute(src);
          let dst_u8 = std::mem::transmute(dst);
          return func(src_u8, dst_u8);
        }
      } else {
        if let Some(func) = SATD_FNS_U16[to_index(bsize)] {
          let src_u16 = std::mem::transmute(src);
          let dst_u16 = std::mem::transmute(dst);
          return func(src_u16, dst_u16);
        }
      }
    }
  }

  crate::dist::rust::get_satd(dst, src, w, h, bit_depth, _cpu)
}
