use crate::cpu_features::CpuFeatureLevel;
use crate::encoder::IMPORTANCE_BLOCK_SIZE;
use crate::rdo::DistortionScale;
use crate::tiling::PlaneRegion;
use crate::util::{self, Pixel, PixelType};
use std::arch::wasm32::*;

unsafe fn sum128_i32(v: v128) -> i32 {
  let hi = i32x4_shuffle::<2, 3, 0, 0>(v, v);
  let sum = i32x4_add(v, hi);
  let hi2 = i32x4_shuffle::<1, 0, 0, 0>(sum, sum);
  let final_sum = i32x4_add(sum, hi2);
  i32x4_extract_lane::<0>(final_sum)
}

#[inline(always)]
#[allow(clippy::let_and_return)]
pub fn get_weighted_sse<T: Pixel>(
  src: &PlaneRegion<'_, T>, dst: &PlaneRegion<'_, T>, scale: &[u32],
  scale_stride: usize, w: usize, h: usize, bit_depth: usize,
  _cpu: CpuFeatureLevel,
) -> u64
where
  i32: util::math::CastFromPrimitive<T>,
{
  assert_eq!(IMPORTANCE_BLOCK_SIZE >> 1, 4);

  let _call_rust = || -> u64 {
    crate::dist::rust::get_weighted_sse(
      dst,
      src,
      scale,
      scale_stride,
      w,
      h,
      bit_depth,
      _cpu,
    )
  };

  let den = DistortionScale::new(1, 1 << 8).0 as u64;

  let dist = unsafe {
    let mut acc = 0u64;
    let src_ptr = src.data_ptr();
    let dst_ptr = dst.data_ptr();
    let src_stride = src.plane_cfg.stride as usize;
    let dst_stride = dst.plane_cfg.stride as usize;

    for r in (0..h).step_by(4) {
      for c in (0..w).step_by(4) {
        let s_val = *scale.as_ptr().add((r >> 2) * scale_stride + (c >> 2));

        let mut block_sse = 0u64;

        for i in 0..4 {
          let s_row_ptr = src_ptr.add((r + i) * src_stride + c);
          let d_row_ptr = dst_ptr.add((r + i) * dst_stride + c);

          let (s, d) = if T::type_enum() == PixelType::U8 {
            let s = v128_load32_zero(s_row_ptr as *const u32);
            let d = v128_load32_zero(d_row_ptr as *const u32);
            (u16x8_extend_low_u8x16(s), u16x8_extend_low_u8x16(d))
          } else {
            let s = v128_load64_zero(s_row_ptr as *const u64);
            let d = v128_load64_zero(d_row_ptr as *const u64);
            (s, d)
          };

          let diff = i16x8_sub(s, d);
          let sq = i32x4_dot_i16x8(diff, diff);
          let sum_sq = sum128_i32(sq) as u64;
          block_sse += sum_sq;
        }

        acc += block_sse * (s_val as u64);
      }
    }

    (acc + (den >> 1)) / den
  };

  dist
}
