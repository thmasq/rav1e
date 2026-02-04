use crate::cdef::rust::pad_into_tmp16;
use crate::cdef::{rust::CDEF_DIV_TABLE, CDEF_HAVE_ALL, CDEF_VERY_LARGE};
use crate::cpu_features::CpuFeatureLevel;
use crate::tiling::{PlaneRegion, PlaneRegionMut};
use crate::util;
use crate::util::{msb, Pixel, PixelType};
use core::arch::wasm32::*;
use std::cmp;

#[inline(always)]
unsafe fn v128_constrain(
  diff: v128, threshold: v128, shift: u32, damping: i32,
) -> v128 {
  if damping == 0 {
    return i16x8_splat(0);
  }

  let abs_diff = i16x8_abs(diff);

  let shifted = u16x8_shr(abs_diff, shift);
  let diff_term = i16x8_sub_sat(threshold, shifted);
  let magnitude = i16x8_min(abs_diff, i16x8_max(i16x8_splat(0), diff_term));

  let sign_mask = i16x8_shr(diff, 15);
  let xored = v128_xor(magnitude, sign_mask);
  i16x8_sub(xored, sign_mask)
}

#[inline(always)]
unsafe fn cdef_filter_8x8_u16(
  dst: *mut u8, dst_stride: isize, src: *const u16, src_stride: isize,
  pri_strength: i32, sec_strength: i32, dir: usize, damping: i32,
  bit_depth: usize, xdec: usize, ydec: usize, is_hbd: bool,
) {
  let xsize = 8 >> xdec;
  let ysize = 8 >> ydec;

  let coeff_shift = bit_depth.saturating_sub(8);

  let pri_damping = cmp::max(0, damping - msb(pri_strength));
  let sec_damping = cmp::max(0, damping - msb(sec_strength));

  let pri_thresh = pri_strength;
  let sec_thresh = sec_strength;

  let pri_thresh_vec = i16x8_splat(pri_thresh as i16);
  let sec_thresh_vec = i16x8_splat(sec_thresh as i16);

  let cdef_pri_taps = [[4, 2], [3, 3]];
  let cdef_sec_taps = [[2, 1], [2, 1]];

  let pri_taps = cdef_pri_taps[((pri_strength >> coeff_shift) & 1) as usize];
  let sec_taps = cdef_sec_taps[((pri_strength >> coeff_shift) & 1) as usize];

  let pri_tap0 = i16x8_splat(pri_taps[0] as i16);
  let pri_tap1 = i16x8_splat(pri_taps[1] as i16);
  let sec_tap0 = i16x8_splat(sec_taps[0] as i16);
  let sec_tap1 = i16x8_splat(sec_taps[1] as i16);

  let cdef_directions = [
    [-1 * src_stride + 1, -2 * src_stride + 2],
    [0 * src_stride + 1, -1 * src_stride + 2],
    [0 * src_stride + 1, 0 * src_stride + 2],
    [0 * src_stride + 1, 1 * src_stride + 2],
    [1 * src_stride + 1, 2 * src_stride + 2],
    [1 * src_stride + 0, 2 * src_stride + 1],
    [1 * src_stride + 0, 2 * src_stride + 0],
    [1 * src_stride + 0, 2 * src_stride - 1],
  ];

  let dirs = [
    cdef_directions[dir],
    cdef_directions[(dir + 2) & 7],
    cdef_directions[(dir + 6) & 7],
  ];

  for y in 0..ysize {
    let ptr_in =
      (src as *const u8).offset((y as isize) * src_stride) as *const u16;
    let dst_row = dst.offset((y as isize) * dst_stride);

    let px = v128_load(ptr_in as *const v128);

    let mut min_px = px;
    let mut max_px = px;
    let mut sum = i16x8_splat(0);

    let process_tap = |offset: isize,
                       tap_vec: v128,
                       thresh_vec: v128,
                       shift: i32,
                       min_px: &mut v128,
                       max_px: &mut v128,
                       sum: &mut v128| {
      let p_ptr = (ptr_in as *const u8).offset(offset) as *const u16;
      let p_val = v128_load(p_ptr as *const v128);

      *max_px = u16x8_max(*max_px, p_val);
      *min_px = u16x8_min(*min_px, p_val);

      let diff = i16x8_sub(p_val, px);
      let constrained =
        v128_constrain(diff, thresh_vec, shift as u32, damping);

      let term = i16x8_mul(tap_vec, constrained);
      *sum = i16x8_add(*sum, term);
    };

    process_tap(
      dirs[0][0],
      pri_tap0,
      pri_thresh_vec,
      pri_damping,
      &mut min_px,
      &mut max_px,
      &mut sum,
    );
    process_tap(
      -dirs[0][0],
      pri_tap0,
      pri_thresh_vec,
      pri_damping,
      &mut min_px,
      &mut max_px,
      &mut sum,
    );
    process_tap(
      dirs[0][1],
      pri_tap1,
      pri_thresh_vec,
      pri_damping,
      &mut min_px,
      &mut max_px,
      &mut sum,
    );
    process_tap(
      -dirs[0][1],
      pri_tap1,
      pri_thresh_vec,
      pri_damping,
      &mut min_px,
      &mut max_px,
      &mut sum,
    );

    for i in 1..3 {
      for j in 0..2 {
        let tap = if j == 0 { sec_tap0 } else { sec_tap1 };
        process_tap(
          dirs[i][j],
          tap,
          sec_thresh_vec,
          sec_damping,
          &mut min_px,
          &mut max_px,
          &mut sum,
        );
        process_tap(
          -dirs[i][j],
          tap,
          sec_thresh_vec,
          sec_damping,
          &mut min_px,
          &mut max_px,
          &mut sum,
        );
      }
    }

    let sum_sign = i16x8_shr(sum, 15);
    let bias = i16x8_add(i16x8_splat(8), sum_sign);
    let rounded = i16x8_add(sum, bias);
    let offset = i16x8_shr(rounded, 4);
    let new_val = i16x8_add(px, offset);

    let clamped = i16x8_max(min_px, i16x8_min(max_px, new_val));

    if is_hbd {
      if xsize == 8 {
        v128_store(dst_row as *mut v128, clamped);
      } else {
        v128_store64_lane::<0>(clamped, dst_row as *mut u64);
      }
    } else {
      let res_u8 = u8x16_narrow_i16x8(clamped, clamped);
      if xsize == 8 {
        v128_store64_lane::<0>(res_u8, dst_row as *mut u64);
      } else {
        v128_store32_lane::<0>(res_u8, dst_row as *mut u32);
      }
    }
  }
}

#[target_feature(enable = "simd128")]
pub(crate) unsafe fn cdef_filter_block<T: Pixel>(
  dst: &mut PlaneRegionMut<'_, T>, input: *const T, istride: isize,
  pri_strength: i32, sec_strength: i32, dir: usize, damping: i32,
  bit_depth: usize, xdec: usize, ydec: usize, edges: u8,
  _cpu: CpuFeatureLevel,
) where
  i32: util::math::CastFromPrimitive<T>,
{
  if edges == CDEF_HAVE_ALL && T::type_enum() == PixelType::U8 {
    let dst_u8 =
      &mut *(dst as *mut PlaneRegionMut<'_, T> as *mut PlaneRegionMut<'_, u8>);
    return cdef_filter_block_8bit_fast(
      dst_u8,
      input as *const u8,
      istride,
      pri_strength,
      sec_strength,
      dir,
      damping,
      xdec,
      ydec,
    );
  }

  let mut tmp_buf_storage = [CDEF_VERY_LARGE; 16 * 16];
  let tmp_stride = 16isize;

  let (src_ptr, src_stride) =
    if edges == CDEF_HAVE_ALL && T::type_enum() == PixelType::U16 {
      (input as *const u16, istride)
    } else {
      let tmp_ptr = tmp_buf_storage.as_mut_ptr().offset(2 * tmp_stride + 2);

      pad_into_tmp16(
        tmp_ptr,
        tmp_stride,
        input,
        istride,
        8 >> xdec,
        8 >> ydec,
        edges,
      );
      (tmp_ptr.offset(2 * tmp_stride + 2) as *const u16, tmp_stride)
    };

  cdef_filter_8x8_u16(
    dst.data_ptr_mut() as *mut u8,
    T::to_asm_stride(dst.plane_cfg.stride),
    src_ptr,
    src_stride * (std::mem::size_of::<T>() as isize),
    pri_strength,
    sec_strength,
    dir,
    damping,
    bit_depth,
    xdec,
    ydec,
    T::type_enum() == PixelType::U16,
  );
}

#[target_feature(enable = "simd128")]
unsafe fn cdef_filter_block_8bit_fast(
  dst: &mut PlaneRegionMut<'_, u8>, input: *const u8, istride: isize,
  pri_strength: i32, sec_strength: i32, dir: usize, damping: i32, xdec: usize,
  ydec: usize,
) {
  let xsize = 8 >> xdec;
  let ysize = 8 >> ydec;

  let pri_damping = cmp::max(0, damping - msb(pri_strength));
  let sec_damping = cmp::max(0, damping - msb(sec_strength));

  let pri_thresh_vec = i16x8_splat(pri_strength as i16);
  let sec_thresh_vec = i16x8_splat(sec_strength as i16);

  let cdef_pri_taps = [[4, 2], [3, 3]];
  let cdef_sec_taps = [[2, 1], [2, 1]];
  let pri_taps = cdef_pri_taps[(pri_strength & 1) as usize];
  let sec_taps = cdef_sec_taps[(pri_strength & 1) as usize];

  let pri_tap0 = i16x8_splat(pri_taps[0] as i16);
  let pri_tap1 = i16x8_splat(pri_taps[1] as i16);
  let sec_tap0 = i16x8_splat(sec_taps[0] as i16);
  let sec_tap1 = i16x8_splat(sec_taps[1] as i16);

  let cdef_directions = [
    [-1 * istride + 1, -2 * istride + 2],
    [0 * istride + 1, -1 * istride + 2],
    [0 * istride + 1, 0 * istride + 2],
    [0 * istride + 1, 1 * istride + 2],
    [1 * istride + 1, 2 * istride + 2],
    [1 * istride + 0, 2 * istride + 1],
    [1 * istride + 0, 2 * istride + 0],
    [1 * istride + 0, 2 * istride - 1],
  ];

  let dirs = [
    cdef_directions[dir],
    cdef_directions[(dir + 2) & 7],
    cdef_directions[(dir + 6) & 7],
  ];

  for y in 0..ysize {
    let ptr_in = input.offset((y as isize) * istride);
    let dst_row = &mut dst[y];

    let px_u8 = v128_load64_zero(ptr_in as *const u64);
    let px = u16x8_extend_low_u8x16(px_u8);

    let mut min_px = px;
    let mut max_px = px;
    let mut sum = i16x8_splat(0);

    let process_tap = |offset: isize,
                       tap_vec: v128,
                       thresh_vec: v128,
                       shift: i32,
                       min_px: &mut v128,
                       max_px: &mut v128,
                       sum: &mut v128| {
      let p_ptr = ptr_in.offset(offset);
      let p_u8 = v128_load64_zero(p_ptr as *const u64);
      let p_val = u16x8_extend_low_u8x16(p_u8);

      *max_px = u16x8_max(*max_px, p_val);
      *min_px = u16x8_min(*min_px, p_val);

      let diff = i16x8_sub(p_val, px);
      let constrained =
        v128_constrain(diff, thresh_vec, shift as u32, damping);

      let term = i16x8_mul(tap_vec, constrained);
      *sum = i16x8_add(*sum, term);
    };

    process_tap(
      dirs[0][0],
      pri_tap0,
      pri_thresh_vec,
      pri_damping,
      &mut min_px,
      &mut max_px,
      &mut sum,
    );
    process_tap(
      -dirs[0][0],
      pri_tap0,
      pri_thresh_vec,
      pri_damping,
      &mut min_px,
      &mut max_px,
      &mut sum,
    );
    process_tap(
      dirs[0][1],
      pri_tap1,
      pri_thresh_vec,
      pri_damping,
      &mut min_px,
      &mut max_px,
      &mut sum,
    );
    process_tap(
      -dirs[0][1],
      pri_tap1,
      pri_thresh_vec,
      pri_damping,
      &mut min_px,
      &mut max_px,
      &mut sum,
    );

    for i in 1..3 {
      for j in 0..2 {
        let tap = if j == 0 { sec_tap0 } else { sec_tap1 };
        process_tap(
          dirs[i][j],
          tap,
          sec_thresh_vec,
          sec_damping,
          &mut min_px,
          &mut max_px,
          &mut sum,
        );
        process_tap(
          -dirs[i][j],
          tap,
          sec_thresh_vec,
          sec_damping,
          &mut min_px,
          &mut max_px,
          &mut sum,
        );
      }
    }

    let sum_sign = i16x8_shr(sum, 15);
    let bias = i16x8_add(i16x8_splat(8), sum_sign);
    let rounded = i16x8_add(sum, bias);
    let offset = i16x8_shr(rounded, 4);
    let new_val = i16x8_add(px, offset);
    let clamped = i16x8_max(min_px, i16x8_min(max_px, new_val));

    let res_u8 = u8x16_narrow_i16x8(clamped, clamped);

    if xsize == 8 {
      v128_store64_lane::<0>(res_u8, dst_row.as_mut_ptr() as *mut u64);
    } else {
      v128_store32_lane::<0>(res_u8, dst_row.as_mut_ptr() as *mut u32);
    }
  }
}

pub(crate) fn cdef_find_dir<T: Pixel>(
  img: &PlaneRegion<'_, T>, var: &mut u32, coeff_shift: usize,
  _cpu: CpuFeatureLevel,
) -> i32
where
  i32: util::math::CastFromPrimitive<T>,
{
  match T::type_enum() {
    PixelType::U8 => unsafe {
      cdef_find_dir_u8(
        img.data_ptr() as *const u8,
        T::to_asm_stride(img.plane_cfg.stride),
        var,
        coeff_shift,
      )
    },
    PixelType::U16 => unsafe {
      cdef_find_dir_u16(
        img.data_ptr() as *const u16,
        T::to_asm_stride(img.plane_cfg.stride),
        var,
        coeff_shift,
      )
    },
  }
}

#[inline(always)]
unsafe fn transpose_8x8(lines: &mut [v128; 8]) {
  let a0 = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(lines[0], lines[1]);
  let a1 = i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(lines[0], lines[1]);
  let a2 = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(lines[2], lines[3]);
  let a3 = i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(lines[2], lines[3]);
  let a4 = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(lines[4], lines[5]);
  let a5 = i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(lines[4], lines[5]);
  let a6 = i16x8_shuffle::<0, 8, 1, 9, 2, 10, 3, 11>(lines[6], lines[7]);
  let a7 = i16x8_shuffle::<4, 12, 5, 13, 6, 14, 7, 15>(lines[6], lines[7]);

  let b0 = i16x8_shuffle::<0, 1, 8, 9, 2, 3, 10, 11>(a0, a2);
  let b1 = i16x8_shuffle::<4, 5, 12, 13, 6, 7, 14, 15>(a0, a2);
  let b2 = i16x8_shuffle::<0, 1, 8, 9, 2, 3, 10, 11>(a1, a3);
  let b3 = i16x8_shuffle::<4, 5, 12, 13, 6, 7, 14, 15>(a1, a3);
  let b4 = i16x8_shuffle::<0, 1, 8, 9, 2, 3, 10, 11>(a4, a6);
  let b5 = i16x8_shuffle::<4, 5, 12, 13, 6, 7, 14, 15>(a4, a6);
  let b6 = i16x8_shuffle::<0, 1, 8, 9, 2, 3, 10, 11>(a5, a7);
  let b7 = i16x8_shuffle::<4, 5, 12, 13, 6, 7, 14, 15>(a5, a7);

  lines[0] = i16x8_shuffle::<0, 1, 2, 3, 8, 9, 10, 11>(b0, b4);
  lines[1] = i16x8_shuffle::<4, 5, 6, 7, 12, 13, 14, 15>(b0, b4);
  lines[2] = i16x8_shuffle::<0, 1, 2, 3, 8, 9, 10, 11>(b1, b5);
  lines[3] = i16x8_shuffle::<4, 5, 6, 7, 12, 13, 14, 15>(b1, b5);
  lines[4] = i16x8_shuffle::<0, 1, 2, 3, 8, 9, 10, 11>(b2, b6);
  lines[5] = i16x8_shuffle::<4, 5, 6, 7, 12, 13, 14, 15>(b2, b6);
  lines[6] = i16x8_shuffle::<0, 1, 2, 3, 8, 9, 10, 11>(b3, b7);
  lines[7] = i16x8_shuffle::<4, 5, 6, 7, 12, 13, 14, 15>(b3, b7);
}

#[inline(always)]
unsafe fn sum_sq(v: v128) -> i32 {
  let sq_sums = i32x4_dot_i16x8(v, v);

  let s1 = i32x4_shuffle::<2, 3, 0, 0>(sq_sums, sq_sums);
  let s2 = i32x4_add(sq_sums, s1);
  let s3 = i32x4_shuffle::<1, 0, 0, 0>(s2, s2);
  i32x4_extract_lane::<0>(i32x4_add(s2, s3))
}

#[inline(always)]
unsafe fn cdef_find_dir_u8(
  src: *const u8, stride: isize, var: &mut u32, _coeff_shift: usize,
) -> i32 {
  let mut lines = [i32x4(0, 0, 0, 0); 8];
  let offset = i16x8_splat(128);
  for i in 0..8 {
    let row = v128_load64_zero(src.offset(i as isize * stride) as *const u64);
    lines[i] = i16x8_sub(u16x8_extend_low_u8x16(row), offset);
  }
  cdef_compute_dist(&mut lines, var)
}

#[inline(always)]
unsafe fn cdef_find_dir_u16(
  src: *const u16, stride: isize, var: &mut u32, coeff_shift: usize,
) -> i32 {
  let mut lines = [i32x4(0, 0, 0, 0); 8];
  let offset = i16x8_splat(128);
  let shift = coeff_shift as u32;
  for i in 0..8 {
    let row = v128_load(src.offset(i as isize * stride) as *const v128);
    lines[i] = i16x8_sub(u16x8_shr(row, shift), offset);
  }
  cdef_compute_dist(&mut lines, var)
}

const SHIFT_LEFT_MASKS: [[u8; 16]; 8] = [
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], // 0
  [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 16], // 1
  [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 16, 16, 16], // 2
  [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 16, 16, 16, 16, 16], // 3
  [8, 9, 10, 11, 12, 13, 14, 15, 16, 16, 16, 16, 16, 16, 16, 16], // 4
  [10, 11, 12, 13, 14, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16], // 5
  [12, 13, 14, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16], // 6
  [14, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16], // 7
];

const SHIFT_LEFT_MASKS_INV: [[u8; 16]; 8] = [
  [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16], // 8 (N/A)
  [14, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16], // 7
  [12, 13, 14, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16], // 6
  [10, 11, 12, 13, 14, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16], // 5
  [8, 9, 10, 11, 12, 13, 14, 15, 16, 16, 16, 16, 16, 16, 16, 16],   // 4
  [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 16, 16, 16, 16, 16],     // 3
  [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 16, 16, 16],       // 2
  [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 16],         // 1
];

#[inline(always)]
unsafe fn cdef_compute_dist(lines: &mut [v128; 8], var: &mut u32) -> i32 {
  let mut cost = [0i32; 8];

  let p6 = i16x8_add(
    i16x8_add(i16x8_add(lines[0], lines[1]), i16x8_add(lines[2], lines[3])),
    i16x8_add(i16x8_add(lines[4], lines[5]), i16x8_add(lines[6], lines[7])),
  );
  cost[6] = sum_sq(p6) * CDEF_DIV_TABLE[8];

  let mut t_lines = *lines;
  transpose_8x8(&mut t_lines);
  let p2 = i16x8_add(
    i16x8_add(
      i16x8_add(t_lines[0], t_lines[1]),
      i16x8_add(t_lines[2], t_lines[3]),
    ),
    i16x8_add(
      i16x8_add(t_lines[4], t_lines[5]),
      i16x8_add(t_lines[6], t_lines[7]),
    ),
  );
  cost[2] = sum_sq(p2) * CDEF_DIV_TABLE[8];

  {
    let mut p0_lo = i16x8_splat(0);
    let mut p0_hi = i16x8_splat(0);
    let mut p4_lo = i16x8_splat(0);
    let mut p4_hi = i16x8_splat(0);

    for i in 0..8 {
      let mask_lo = v128_load(SHIFT_LEFT_MASKS[i].as_ptr() as *const v128);
      let mask_hi = v128_load(SHIFT_LEFT_MASKS_INV[i].as_ptr() as *const v128);

      p0_lo = i16x8_add(p0_lo, u8x16_swizzle(lines[i], mask_lo));
      p0_hi = i16x8_add(p0_hi, u8x16_swizzle(lines[i], mask_hi));

      let rev_mask =
        i8x16(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);
      let row_rev = u8x16_swizzle(lines[i], rev_mask);
      p4_lo = i16x8_add(p4_lo, u8x16_swizzle(row_rev, mask_lo));
      p4_hi = i16x8_add(p4_hi, u8x16_swizzle(row_rev, mask_hi));
    }

    let w_lo = i32x4(
      CDEF_DIV_TABLE[1],
      CDEF_DIV_TABLE[2],
      CDEF_DIV_TABLE[3],
      CDEF_DIV_TABLE[4],
    );
    let w_lo2 = i32x4(
      CDEF_DIV_TABLE[5],
      CDEF_DIV_TABLE[6],
      CDEF_DIV_TABLE[7],
      CDEF_DIV_TABLE[8],
    );
    let w_hi = i32x4(
      CDEF_DIV_TABLE[7],
      CDEF_DIV_TABLE[6],
      CDEF_DIV_TABLE[5],
      CDEF_DIV_TABLE[4],
    );
    let w_hi2 =
      i32x4(CDEF_DIV_TABLE[3], CDEF_DIV_TABLE[2], CDEF_DIV_TABLE[1], 0);

    let sum_sq_weighted = |v_lo: v128, v_hi: v128| -> i32 {
      let l_lo = i32x4_extend_low_i16x8(v_lo);
      let l_hi = i32x4_extend_high_i16x8(v_lo);
      let h_lo = i32x4_extend_low_i16x8(v_hi);
      let h_hi = i32x4_extend_high_i16x8(v_hi);

      let s1 = i32x4_mul(i32x4_mul(l_lo, l_lo), w_lo);
      let s2 = i32x4_mul(i32x4_mul(l_hi, l_hi), w_lo2);
      let s3 = i32x4_mul(i32x4_mul(h_lo, h_lo), w_hi);
      let s4 = i32x4_mul(i32x4_mul(h_hi, h_hi), w_hi2);

      let sum = i32x4_add(i32x4_add(s1, s2), i32x4_add(s3, s4));
      let s5 = i32x4_add(sum, i32x4_shuffle::<2, 3, 0, 0>(sum, sum));
      let s6 = i32x4_add(s5, i32x4_shuffle::<1, 0, 0, 0>(s5, s5));
      i32x4_extract_lane::<0>(s6)
    };

    cost[0] = sum_sq_weighted(p0_lo, p0_hi);
    cost[4] = sum_sq_weighted(p4_lo, p4_hi);
  }

  let mut sq_cols = [i32x4(0, 0, 0, 0); 8];
  for i in 0..8 {
    let r = lines[i];
    let a = i16x8_shuffle::<0, 2, 4, 6, 0, 0, 0, 0>(r, r);
    let b = i16x8_shuffle::<1, 3, 5, 7, 0, 0, 0, 0>(r, r);
    sq_cols[i] = i16x8_add(a, b);
  }

  {
    let mut p1_acc = i16x8_splat(0);
    let mut p3_acc = i16x8_splat(0);
    for i in 0..8 {
      let mask = v128_load(SHIFT_LEFT_MASKS[i].as_ptr() as *const v128);
      p1_acc = i16x8_add(p1_acc, u8x16_swizzle(sq_cols[i], mask));

      let rev = i8x16(6, 7, 4, 5, 2, 3, 0, 1, 8, 9, 10, 11, 12, 13, 14, 15);
      let r_rev = u8x16_swizzle(sq_cols[i], rev);
      p3_acc = i16x8_add(p3_acc, u8x16_swizzle(r_rev, mask));
    }

    let w_odd_1 = i32x4(420, 210, 140, 840);
    let w_odd_2 = i32x4(840, 840, 840, 840);
    let w_odd_3 = i32x4(140, 210, 420, 0);

    let mut p1_hi = i16x8_splat(0);
    let mut p3_hi = i16x8_splat(0);
    for i in 0..8 {
      let mask_inv =
        v128_load(SHIFT_LEFT_MASKS_INV[i].as_ptr() as *const v128);
      p1_hi = i16x8_add(p1_hi, u8x16_swizzle(sq_cols[i], mask_inv));

      let rev = i8x16(6, 7, 4, 5, 2, 3, 0, 1, 8, 9, 10, 11, 12, 13, 14, 15);
      let r_rev = u8x16_swizzle(sq_cols[i], rev);
      p3_hi = i16x8_add(p3_hi, u8x16_swizzle(r_rev, mask_inv));
    }

    let sum_sq_odd_weighted = |lo: v128, hi: v128| -> i32 {
      let l_lo = i32x4_extend_low_i16x8(lo);
      let l_hi = i32x4_extend_high_i16x8(lo);
      let h_lo = i32x4_extend_low_i16x8(hi);

      let s1 = i32x4_mul(i32x4_mul(l_lo, l_lo), w_odd_1);
      let s2 = i32x4_mul(i32x4_mul(l_hi, l_hi), w_odd_2);
      let s3 = i32x4_mul(i32x4_mul(h_lo, h_lo), w_odd_3);

      let sum = i32x4_add(i32x4_add(s1, s2), s3);

      let s4 = i32x4_add(sum, i32x4_shuffle::<2, 3, 0, 0>(sum, sum));
      let s5 = i32x4_add(s4, i32x4_shuffle::<1, 0, 0, 0>(s4, s4));
      i32x4_extract_lane::<0>(s5)
    };

    cost[1] = sum_sq_odd_weighted(p1_acc, p1_hi);
    cost[3] = sum_sq_odd_weighted(p3_acc, p3_hi);
  }

  {
    let mut sq_rows = [i32x4(0, 0, 0, 0); 4];
    for i in 0..4 {
      sq_rows[i] = i16x8_add(lines[2 * i], lines[2 * i + 1]);
    }

    let mut p7_acc = i16x8_splat(0);
    let mut p7_hi = i16x8_splat(0);
    let mut p5_acc = i16x8_splat(0);
    let mut p5_hi = i16x8_splat(0);

    for i in 0..4 {
      let mask = v128_load(SHIFT_LEFT_MASKS[i].as_ptr() as *const v128);
      let mask_inv =
        v128_load(SHIFT_LEFT_MASKS_INV[i].as_ptr() as *const v128);

      p7_acc = i16x8_add(p7_acc, u8x16_swizzle(sq_rows[i], mask));
      p7_hi = i16x8_add(p7_hi, u8x16_swizzle(sq_rows[i], mask_inv));

      let r_flip = sq_rows[3 - i];
      p5_acc = i16x8_add(p5_acc, u8x16_swizzle(r_flip, mask));
      p5_hi = i16x8_add(p5_hi, u8x16_swizzle(r_flip, mask_inv));
    }

    let w_odd_1 = i32x4(420, 210, 140, 840);
    let w_odd_2 = i32x4(840, 840, 840, 840);
    let w_odd_3 = i32x4(140, 210, 420, 0);

    let sum_sq_odd_weighted = |lo: v128, hi: v128| -> i32 {
      let l_lo = i32x4_extend_low_i16x8(lo);
      let l_hi = i32x4_extend_high_i16x8(lo);
      let h_lo = i32x4_extend_low_i16x8(hi);

      let s1 = i32x4_mul(i32x4_mul(l_lo, l_lo), w_odd_1);
      let s2 = i32x4_mul(i32x4_mul(l_hi, l_hi), w_odd_2);
      let s3 = i32x4_mul(i32x4_mul(h_lo, h_lo), w_odd_3);

      let sum = i32x4_add(i32x4_add(s1, s2), s3);
      let s4 = i32x4_add(sum, i32x4_shuffle::<2, 3, 0, 0>(sum, sum));
      let s5 = i32x4_add(s4, i32x4_shuffle::<1, 0, 0, 0>(s4, s4));
      i32x4_extract_lane::<0>(s5)
    };

    cost[7] = sum_sq_odd_weighted(p7_acc, p7_hi);
    cost[5] = sum_sq_odd_weighted(p5_acc, p5_hi);
  }

  let mut best_cost = 0;
  let mut best_dir = 0;
  for i in 0..8 {
    if cost[i] > best_cost {
      best_cost = cost[i];
      best_dir = i;
    }
  }

  *var = ((best_cost - cost[(best_dir + 4) & 7]) >> 10) as u32;
  best_dir as i32
}
