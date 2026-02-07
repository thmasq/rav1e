use crate::cpu_features::CpuFeatureLevel;
use crate::frame::PlaneSlice;
use crate::mc::{FilterMode, SUBPEL_FILTERS, SUBPEL_FILTER_SIZE};
use crate::tiling::PlaneRegionMut;
use crate::util::{self, Pixel, PixelType};
use core::arch::wasm32::*;
use std::mem::transmute;

use crate::mc::rust;

#[inline(always)]
fn get_filter(
  mode: FilterMode, frac: i32, length: usize,
) -> [i32; SUBPEL_FILTER_SIZE] {
  let filter_idx = if mode == FilterMode::BILINEAR || length > 4 {
    mode as usize
  } else {
    (mode as usize).min(1) + 4
  };
  SUBPEL_FILTERS[filter_idx][frac as usize]
}

#[inline(always)]
unsafe fn load_8tap_coeffs(filter: &[i32; 8]) -> [v128; 8] {
  [
    i16x8_splat(filter[0] as i16),
    i16x8_splat(filter[1] as i16),
    i16x8_splat(filter[2] as i16),
    i16x8_splat(filter[3] as i16),
    i16x8_splat(filter[4] as i16),
    i16x8_splat(filter[5] as i16),
    i16x8_splat(filter[6] as i16),
    i16x8_splat(filter[7] as i16),
  ]
}

#[inline(always)]
unsafe fn filter_8tap_h_u8_8wide(
  src: *const u8, coeffs: &[v128; 8],
) -> (v128, v128) {
  let mut sum_lo = i32x4_splat(0);
  let mut sum_hi = i32x4_splat(0);

  for k in 0..8 {
    let p_i16 = u16x8_load_extend_u8x8(src.offset(k as isize));

    sum_lo = i32x4_add(sum_lo, i32x4_extmul_low_i16x8(p_i16, coeffs[k]));
    sum_hi = i32x4_add(sum_hi, i32x4_extmul_high_i16x8(p_i16, coeffs[k]));
  }
  (sum_lo, sum_hi)
}

#[target_feature(enable = "simd128")]
unsafe fn put_8tap_u8(
  dst: &mut PlaneRegionMut<'_, u8>, src: PlaneSlice<'_, u8>, width: usize,
  height: usize, col_frac: i32, row_frac: i32, mode_x: FilterMode,
  mode_y: FilterMode, _bit_depth: usize,
) {
  let intermediate_bits = 4;
  let max_sample_val = 255;

  let y_filter = get_filter(mode_y, row_frac, height);
  let x_filter = get_filter(mode_x, col_frac, width);

  let x_coeffs = load_8tap_coeffs(&x_filter);
  let y_coeffs = load_8tap_coeffs(&y_filter);

  match (col_frac, row_frac) {
    (0, 0) => {
      for r in 0..height {
        let src_slice = src.row(r);
        let dst_slice = &mut dst[r];
        dst_slice[..width].copy_from_slice(&src_slice[..width]);
      }
    }
    (0, _) => {
      let offset_slice = src.go_up(3);
      let dst_ptr = dst.data_ptr_mut();
      let dst_stride = dst.plane_cfg.stride;
      let src_stride = offset_slice.plane.geometry().stride.get();
      let src_ptr_base = offset_slice.as_ptr();

      for r in 0..height {
        let src_row_ptr =
          src_ptr_base.offset((r as isize) * (src_stride as isize));
        let dst_row_ptr = dst_ptr.offset((r as isize) * (dst_stride as isize));

        let mut c = 0;
        while c + 8 <= width {
          let mut sum_lo = i32x4_splat(0);
          let mut sum_hi = i32x4_splat(0);

          for k in 0..8 {
            let p_ptr = src_row_ptr
              .offset((k as isize) * (src_stride as isize) + c as isize);
            let p_u8 = v128_load64_zero(p_ptr as *const u64);
            let p_i16 = u16x8_extend_low_u8x16(p_u8);

            let term = i16x8_mul(p_i16, y_coeffs[k]);
            sum_lo = i32x4_add(sum_lo, i32x4_extend_low_i16x8(term));
            sum_hi = i32x4_add(sum_hi, i32x4_extend_high_i16x8(term));
          }

          let round = i32x4_splat(1 << 6);
          sum_lo = i32x4_add(sum_lo, round);
          sum_hi = i32x4_add(sum_hi, round);

          sum_lo = i32x4_shr(sum_lo, 7);
          sum_hi = i32x4_shr(sum_hi, 7);

          let res_i16 = i16x8_narrow_i32x4(sum_lo, sum_hi);
          let res_u8 = u8x16_narrow_i16x8(res_i16, res_i16);

          v128_store64_lane::<0>(res_u8, dst_row_ptr.add(c) as *mut u64);

          c += 8;
        }

        for i in c..width {
          let mut sum = 0;
          for k in 0..8 {
            let val = *src_row_ptr
              .offset((k as isize) * (src_stride as isize) + i as isize)
              as i32;
            sum += val * y_filter[k];
          }
          *dst_row_ptr.add(i) =
            util::round_shift(sum, 7).clamp(0, max_sample_val) as u8;
        }
      }
    }
    (_, 0) => {
      let offset_slice = src.go_left(3);
      let dst_ptr = dst.data_ptr_mut();
      let dst_stride = dst.plane_cfg.stride;
      let src_stride = offset_slice.plane.geometry().stride.get();
      let src_ptr_base = offset_slice.as_ptr();

      for r in 0..height {
        let src_row_ptr =
          src_ptr_base.offset((r as isize) * (src_stride as isize));
        let dst_row_ptr = dst_ptr.offset((r as isize) * (dst_stride as isize));

        let mut c = 0;
        while c + 8 <= width {
          let (mut sum_lo, mut sum_hi) =
            filter_8tap_h_u8_8wide(src_row_ptr.add(c), &x_coeffs);

          let inner_shift = 7 - intermediate_bits;
          let round_inner = i32x4_splat(1 << (inner_shift - 1));
          sum_lo = i32x4_add(sum_lo, round_inner);
          sum_hi = i32x4_add(sum_hi, round_inner);
          sum_lo = i32x4_shr(sum_lo, inner_shift);
          sum_hi = i32x4_shr(sum_hi, inner_shift);

          let outer_shift = intermediate_bits;
          let round_outer = i32x4_splat(1 << (outer_shift - 1));
          sum_lo = i32x4_add(sum_lo, round_outer);
          sum_hi = i32x4_add(sum_hi, round_outer);
          sum_lo = i32x4_shr(sum_lo, outer_shift);
          sum_hi = i32x4_shr(sum_hi, outer_shift);

          let res_i16 = i16x8_narrow_i32x4(sum_lo, sum_hi);
          let res_u8 = u8x16_narrow_i16x8(res_i16, res_i16);
          v128_store64_lane::<0>(res_u8, dst_row_ptr.add(c) as *mut u64);

          c += 8;
        }

        for i in c..width {
          let mut sum = 0;
          for k in 0..8 {
            let val = *src_row_ptr.add(i + k) as i32;
            sum += val * x_filter[k];
          }
          *dst_row_ptr.add(i) = util::round_shift(
            util::round_shift(sum, 7 - intermediate_bits as usize),
            intermediate_bits as usize,
          )
          .clamp(0, max_sample_val) as u8;
        }
      }
    }
    (_, _) => {
      let mut intermediate = [0i16; 8 * 135];
      let offset_slice = src.go_left(3).go_up(3);
      let src_stride = offset_slice.plane.geometry().stride.get();
      let src_ptr_base = offset_slice.as_ptr();
      let dst_ptr = dst.data_ptr_mut();
      let dst_stride = dst.plane_cfg.stride;

      for cg in (0..width).step_by(8) {
        let process_width = (width - cg).min(8);

        for r in 0..height + 7 {
          let src_row = src_ptr_base
            .offset((r as isize) * (src_stride as isize) + cg as isize);
          let int_row = &mut intermediate[r * 8..];

          if process_width == 8 {
            let (mut s_lo, mut s_hi) =
              filter_8tap_h_u8_8wide(src_row, &x_coeffs);

            let shift = 7 - intermediate_bits;
            let round = i32x4_splat(1 << (shift - 1));
            s_lo = i32x4_shr(i32x4_add(s_lo, round), shift);
            s_hi = i32x4_shr(i32x4_add(s_hi, round), shift);

            let res = i16x8_narrow_i32x4(s_lo, s_hi);
            v128_store(int_row.as_mut_ptr() as *mut v128, res);
          } else {
            for c in 0..process_width {
              let mut sum = 0;
              for k in 0..8 {
                sum += *src_row.add(c + k) as i32 * x_filter[k];
              }
              int_row[c] =
                util::round_shift(sum, 7 - intermediate_bits as usize) as i16;
            }
          }
        }

        for r in 0..height {
          let dst_row =
            dst_ptr.offset((r as isize) * (dst_stride as isize) + cg as isize);

          if process_width == 8 {
            let mut sum_lo = i32x4_splat(0);
            let mut sum_hi = i32x4_splat(0);

            for k in 0..8 {
              let int_ptr = intermediate.as_ptr().add((r + k) * 8);
              let val = v128_load(int_ptr as *const v128);

              let term_lo = i32x4_extmul_low_i16x8(val, y_coeffs[k]);
              let term_hi = i32x4_extmul_high_i16x8(val, y_coeffs[k]);

              sum_lo = i32x4_add(sum_lo, term_lo);
              sum_hi = i32x4_add(sum_hi, term_hi);
            }

            let shift = 7 + intermediate_bits;
            let round = i32x4_splat(1 << (shift - 1));
            sum_lo = i32x4_shr(i32x4_add(sum_lo, round), shift);
            sum_hi = i32x4_shr(i32x4_add(sum_hi, round), shift);

            let res_i16 = i16x8_narrow_i32x4(sum_lo, sum_hi);
            let res_u8 = u8x16_narrow_i16x8(res_i16, res_i16);
            v128_store64_lane::<0>(res_u8, dst_row as *mut u64);
          } else {
            for c in 0..process_width {
              let mut sum = 0;
              for k in 0..8 {
                sum += intermediate[(r + k) * 8 + c] as i32 * y_filter[k];
              }
              *dst_row.add(c) =
                util::round_shift(sum, 7 + intermediate_bits as usize)
                  .clamp(0, max_sample_val) as u8;
            }
          }
        }
      }
    }
  }
}

#[target_feature(enable = "simd128")]
unsafe fn prep_8tap_u8(
  tmp: &mut [i16], src: PlaneSlice<'_, u8>, width: usize, height: usize,
  col_frac: i32, row_frac: i32, mode_x: FilterMode, mode_y: FilterMode,
) {
  let y_filter = get_filter(mode_y, row_frac, height);
  let x_filter = get_filter(mode_x, col_frac, width);
  let x_coeffs = load_8tap_coeffs(&x_filter);
  let y_coeffs = load_8tap_coeffs(&y_filter);

  match (col_frac, row_frac) {
    (0, 0) => {
      let src_ptr = src.as_ptr();
      let src_stride = src.plane.geometry().stride.get();

      for r in 0..height {
        let src_row = src_ptr.offset((r as isize) * (src_stride as isize));
        let tmp_row = tmp.as_mut_ptr().add(r * width);
        let mut c = 0;

        while c + 8 <= width {
          let val = u16x8_load_extend_u8x8(src_row.add(c));
          let shifted = i16x8_shl(val, 4);
          v128_store(tmp_row.add(c) as *mut v128, shifted);
          c += 8;
        }

        for i in c..width {
          *tmp_row.add(i) = (*src_row.add(i) as i16) << 4;
        }
      }
    }
    (0, _) => {
      let offset_slice = src.go_up(3);
      let src_stride = offset_slice.plane.geometry().stride.get();
      let src_ptr_base = offset_slice.as_ptr();

      for r in 0..height {
        let src_row_ptr =
          src_ptr_base.offset((r as isize) * (src_stride as isize));
        let tmp_row_ptr = tmp.as_mut_ptr().add(r * width);

        let mut c = 0;
        while c + 8 <= width {
          let mut sum_lo = i32x4_splat(0);
          let mut sum_hi = i32x4_splat(0);
          for k in 0..8 {
            let p_ptr = src_row_ptr
              .offset((k as isize) * (src_stride as isize) + c as isize);

            let p_i16 = u16x8_load_extend_u8x8(p_ptr);

            sum_lo =
              i32x4_add(sum_lo, i32x4_extmul_low_i16x8(p_i16, y_coeffs[k]));
            sum_hi =
              i32x4_add(sum_hi, i32x4_extmul_high_i16x8(p_i16, y_coeffs[k]));
          }

          let round = i32x4_splat(1 << 2);
          sum_lo = i32x4_shr(i32x4_add(sum_lo, round), 3);
          sum_hi = i32x4_shr(i32x4_add(sum_hi, round), 3);

          let res = i16x8_narrow_i32x4(sum_lo, sum_hi);
          v128_store(tmp_row_ptr.add(c) as *mut v128, res);
          c += 8;
        }
        for i in c..width {
          let mut sum = 0;
          for k in 0..8 {
            sum += *src_row_ptr
              .offset((k as isize) * (src_stride as isize) + i as isize)
              as i32
              * y_filter[k];
          }
          *tmp_row_ptr.add(i) = util::round_shift(sum, 3) as i16;
        }
      }
    }
    (_, 0) => {
      let offset_slice = src.go_left(3);
      let src_stride = offset_slice.plane.geometry().stride.get();
      let src_ptr_base = offset_slice.as_ptr();

      for r in 0..height {
        let src_row_ptr =
          src_ptr_base.offset((r as isize) * (src_stride as isize));
        let tmp_row_ptr = tmp.as_mut_ptr().add(r * width);

        let mut c = 0;
        while c + 8 <= width {
          let (mut s_lo, mut s_hi) =
            filter_8tap_h_u8_8wide(src_row_ptr.add(c), &x_coeffs);
          let round = i32x4_splat(1 << 2);
          s_lo = i32x4_shr(i32x4_add(s_lo, round), 3);
          s_hi = i32x4_shr(i32x4_add(s_hi, round), 3);
          let res = i16x8_narrow_i32x4(s_lo, s_hi);
          v128_store(tmp_row_ptr.add(c) as *mut v128, res);
          c += 8;
        }
        for i in c..width {
          let mut sum = 0;
          for k in 0..8 {
            sum += *src_row_ptr.add(i + k) as i32 * x_filter[k];
          }
          *tmp_row_ptr.add(i) = util::round_shift(sum, 3) as i16;
        }
      }
    }
    (_, _) => {
      let mut intermediate = [0i16; 8 * 135];
      let offset_slice = src.go_left(3).go_up(3);
      let src_stride = offset_slice.plane.geometry().stride.get();
      let src_ptr_base = offset_slice.as_ptr();

      for cg in (0..width).step_by(8) {
        let process_width = (width - cg).min(8);

        for r in 0..height + 7 {
          let src_row = src_ptr_base
            .offset((r as isize) * (src_stride as isize) + cg as isize);
          let int_row = &mut intermediate[r * 8..];

          if process_width == 8 {
            let (mut s_lo, mut s_hi) =
              filter_8tap_h_u8_8wide(src_row, &x_coeffs);
            let round = i32x4_splat(1 << 2);
            s_lo = i32x4_shr(i32x4_add(s_lo, round), 3);
            s_hi = i32x4_shr(i32x4_add(s_hi, round), 3);
            v128_store(
              int_row.as_mut_ptr() as *mut v128,
              i16x8_narrow_i32x4(s_lo, s_hi),
            );
          } else {
            for c in 0..process_width {
              let mut sum = 0;
              for k in 0..8 {
                sum += *src_row.add(c + k) as i32 * x_filter[k];
              }
              int_row[c] = util::round_shift(sum, 3) as i16;
            }
          }
        }

        for r in 0..height {
          let tmp_row = tmp.as_mut_ptr().add(r * width + cg);

          if process_width == 8 {
            let mut sum_lo = i32x4_splat(0);
            let mut sum_hi = i32x4_splat(0);
            for k in 0..8 {
              let int_ptr = intermediate.as_ptr().add((r + k) * 8);
              let val = v128_load(int_ptr as *const v128);

              let term_lo = i32x4_extmul_low_i16x8(val, y_coeffs[k]);
              let term_hi = i32x4_extmul_high_i16x8(val, y_coeffs[k]);

              sum_lo = i32x4_add(sum_lo, term_lo);
              sum_hi = i32x4_add(sum_hi, term_hi);
            }
            let round = i32x4_splat(1 << 6);
            sum_lo = i32x4_shr(i32x4_add(sum_lo, round), 7);
            sum_hi = i32x4_shr(i32x4_add(sum_hi, round), 7);
            v128_store(
              tmp_row as *mut v128,
              i16x8_narrow_i32x4(sum_lo, sum_hi),
            );
          } else {
            for c in 0..process_width {
              let mut sum = 0;
              for k in 0..8 {
                sum += intermediate[(r + k) * 8 + c] as i32 * y_filter[k];
              }
              *tmp_row.add(c) = util::round_shift(sum, 7) as i16;
            }
          }
        }
      }
    }
  }
}

pub fn put_8tap<T: Pixel>(
  dst: &mut PlaneRegionMut<'_, T>, src: PlaneSlice<'_, T>, width: usize,
  height: usize, col_frac: i32, row_frac: i32, mode_x: FilterMode,
  mode_y: FilterMode, bit_depth: usize, cpu: CpuFeatureLevel,
) {
  if T::type_enum() == PixelType::U8 {
    unsafe {
      debug_assert_eq!(height & 1, 0);
      debug_assert!(width.is_power_of_two() && (2..=128).contains(&width));
      debug_assert!(dst.rect().width >= width && dst.rect().height >= height);
      debug_assert!(src.accessible(width + 4, height + 4));

      let dst_u8 = &mut *(dst as *mut PlaneRegionMut<'_, T>
        as *mut PlaneRegionMut<'_, u8>);
      // SAFETY: Checked T == u8, PlaneSlice internal layout is compatible for reinterpretation.
      let src_u8 = transmute::<PlaneSlice<'_, T>, PlaneSlice<'_, u8>>(src);
      put_8tap_u8(
        dst_u8, src_u8, width, height, col_frac, row_frac, mode_x, mode_y,
        bit_depth,
      );
    }
  } else {
    rust::put_8tap(
      dst, src, width, height, col_frac, row_frac, mode_x, mode_y, bit_depth,
      cpu,
    );
  }
}

pub fn prep_8tap<T: Pixel>(
  tmp: &mut [i16], src: PlaneSlice<'_, T>, width: usize, height: usize,
  col_frac: i32, row_frac: i32, mode_x: FilterMode, mode_y: FilterMode,
  bit_depth: usize, cpu: CpuFeatureLevel,
) {
  if T::type_enum() == PixelType::U8 {
    unsafe {
      assert_eq!(height & 1, 0);
      assert!(width.is_power_of_two() && (2..=128).contains(&width));

      // SAFETY: Checked T == u8.
      let src_u8 = transmute::<PlaneSlice<'_, T>, PlaneSlice<'_, u8>>(src);
      prep_8tap_u8(
        tmp, src_u8, width, height, col_frac, row_frac, mode_x, mode_y,
      );
    }
  } else {
    rust::prep_8tap(
      tmp, src, width, height, col_frac, row_frac, mode_x, mode_y, bit_depth,
      cpu,
    );
  }
}

#[target_feature(enable = "simd128")]
unsafe fn mc_avg_u8(
  dst: &mut PlaneRegionMut<'_, u8>, tmp1: &[i16], tmp2: &[i16], width: usize,
  height: usize,
) {
  let dst_stride = dst.plane_cfg.stride;
  let dst_ptr = dst.data_ptr_mut();

  let bias_vec = i16x8_splat(16);

  for r in 0..height {
    let dst_row = dst_ptr.offset((r as isize) * (dst_stride as isize));
    let t1_row = &tmp1[r * width..];
    let t2_row = &tmp2[r * width..];

    let mut c = 0;
    while c + 8 <= width {
      let v1 = v128_load(t1_row.as_ptr().add(c) as *const v128);
      let v2 = v128_load(t2_row.as_ptr().add(c) as *const v128);

      let sum = i16x8_add(i16x8_add(v1, v2), bias_vec);
      let res_i16 = i16x8_shr(sum, 5);
      let res_u8 = u8x16_narrow_i16x8(res_i16, res_i16);

      v128_store64_lane::<0>(res_u8, dst_row.add(c) as *mut u64);
      c += 8;
    }

    for i in c..width {
      let sum = t1_row[i] + t2_row[i] + 16;
      *dst_row.add(i) = (sum >> 5).clamp(0, 255) as u8;
    }
  }
}

pub fn mc_avg<T: Pixel>(
  dst: &mut PlaneRegionMut<'_, T>, tmp1: &[i16], tmp2: &[i16], width: usize,
  height: usize, bit_depth: usize, cpu: CpuFeatureLevel,
) {
  if T::type_enum() == PixelType::U8 {
    unsafe {
      let dst_u8 = &mut *(dst as *mut PlaneRegionMut<'_, T>
        as *mut PlaneRegionMut<'_, u8>);
      mc_avg_u8(dst_u8, tmp1, tmp2, width, height);
    }
  } else {
    rust::mc_avg(dst, tmp1, tmp2, width, height, bit_depth, cpu);
  }
}
