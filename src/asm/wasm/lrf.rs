use crate::config::CpuFeatureLevel;
use crate::frame::PlaneSlice;
use crate::lrf::{
  rust, SGRPROJ_MTABLE_BITS, SGRPROJ_RECIP_BITS, SGRPROJ_RST_BITS,
  SGRPROJ_SGR_BITS,
};
use crate::util::Pixel;
use core::arch::wasm32::*;
use std::mem;

static X_BY_XPLUS1: [u32; 256] = [
  1, 128, 171, 192, 205, 213, 219, 224, 228, 230, 233, 235, 236, 238, 239,
  240, 241, 242, 243, 243, 244, 244, 245, 245, 246, 246, 247, 247, 247, 247,
  248, 248, 248, 248, 249, 249, 249, 249, 249, 250, 250, 250, 250, 250, 250,
  250, 251, 251, 251, 251, 251, 251, 251, 251, 251, 251, 252, 252, 252, 252,
  252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 253, 253,
  253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253,
  253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 254, 254, 254,
  254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254,
  254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254,
  254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254,
  254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254,
  254, 254, 254, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
  255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
  255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
  255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
  255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
  255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
  256,
];

#[inline]
pub fn sgrproj_box_ab_r1<const BD: usize>(
  af: &mut [u32], bf: &mut [u32], iimg: &[u32], iimg_sq: &[u32],
  iimg_stride: usize, y: usize, stripe_w: usize, s: u32,
  _cpu: CpuFeatureLevel,
) {
  unsafe {
    sgrproj_box_ab_wasm::<BD>(
      1,
      af,
      bf,
      iimg,
      iimg_sq,
      iimg_stride,
      0,
      y,
      stripe_w,
      s,
    );
  }
}

#[inline]
pub fn sgrproj_box_ab_r2<const BD: usize>(
  af: &mut [u32], bf: &mut [u32], iimg: &[u32], iimg_sq: &[u32],
  iimg_stride: usize, y: usize, stripe_w: usize, s: u32,
  _cpu: CpuFeatureLevel,
) {
  unsafe {
    sgrproj_box_ab_wasm::<BD>(
      2,
      af,
      bf,
      iimg,
      iimg_sq,
      iimg_stride,
      0,
      y,
      stripe_w,
      s,
    );
  }
}

#[inline]
pub fn sgrproj_box_f_r0<T: Pixel>(
  f: &mut [u32], y: usize, w: usize, cdeffed: &PlaneSlice<T>,
  _cpu: CpuFeatureLevel,
) {
  unsafe {
    sgrproj_box_f_r0_wasm(f, y, w, cdeffed);
  }
}

#[inline]
pub fn sgrproj_box_f_r1<T: Pixel>(
  af: &[&[u32]; 3], bf: &[&[u32]; 3], f: &mut [u32], y: usize, w: usize,
  cdeffed: &PlaneSlice<T>, _cpu: CpuFeatureLevel,
) {
  unsafe {
    sgrproj_box_f_r1_wasm(af, bf, f, y, w, cdeffed);
  }
}

#[inline]
pub fn sgrproj_box_f_r2<T: Pixel>(
  af: &[&[u32]; 2], bf: &[&[u32]; 2], f0: &mut [u32], f1: &mut [u32],
  y: usize, w: usize, cdeffed: &PlaneSlice<T>, _cpu: CpuFeatureLevel,
) {
  unsafe {
    sgrproj_box_f_r2_wasm(af, bf, f0, f1, y, w, cdeffed);
  }
}

unsafe fn get_integral_square_wasm(
  iimg: &[u32], stride: usize, x: usize, y: usize, size: usize,
) -> v128 {
  let iimg = iimg.as_ptr().add(y * stride + x);
  let tl = v128_load(iimg as *const v128);
  let tr = v128_load(iimg.add(size) as *const v128);
  let bl = v128_load(iimg.add(size * stride) as *const v128);
  let br = v128_load(iimg.add(size * stride + size) as *const v128);

  i32x4_sub(i32x4_sub(i32x4_add(tl, br), bl), tr)
}

unsafe fn sgrproj_box_ab_wasm<const BD: usize>(
  r: usize, af: &mut [u32], bf: &mut [u32], iimg: &[u32], iimg_sq: &[u32],
  iimg_stride: usize, start_x: usize, y: usize, stripe_w: usize, s: u32,
) {
  let bdm8 = BD - 8;
  let d = r * 2 + 1;
  let n = (d * d) as i32;
  let one_over_n = if r == 1 { 455 } else { 164 };

  let s_vec = i32x4_splat(s as i32);
  let n_vec = i32x4_splat(n);
  let one_over_n_vec = i32x4_splat(one_over_n);
  let mtable_bits_vec = i32x4_splat(1 << SGRPROJ_MTABLE_BITS >> 1);
  let sgr_bits_vec = i32x4_splat(1 << SGRPROJ_SGR_BITS);
  let recip_bits_vec = i32x4_splat(1 << SGRPROJ_RECIP_BITS >> 1);

  for x in (start_x..stripe_w + 2).step_by(4) {
    if x + 4 <= stripe_w + 2 {
      let sum = get_integral_square_wasm(iimg, iimg_stride, x, y, d);
      let ssq = get_integral_square_wasm(iimg_sq, iimg_stride, x, y, d);

      let scaled_sum =
        u32x4_shr(i32x4_add(sum, i32x4_splat(1 << bdm8 >> 1)), bdm8 as u32);
      let scaled_ssq = u32x4_shr(
        i32x4_add(ssq, i32x4_splat(1 << (2 * bdm8) >> 1)),
        (2 * bdm8) as u32,
      );

      let p = i32x4_max(
        i32x4_splat(0),
        i32x4_sub(
          i32x4_mul(scaled_ssq, n_vec),
          i32x4_mul(scaled_sum, scaled_sum),
        ),
      );

      let z = u32x4_shr(
        i32x4_add(i32x4_mul(p, s_vec), mtable_bits_vec),
        SGRPROJ_MTABLE_BITS as u32,
      );

      let mut a_vals = [0u32; 4];
      let z_vals: [u32; 4] = mem::transmute(z);
      for i in 0..4 {
        let idx = z_vals[i].min(255) as usize;
        a_vals[i] = *X_BY_XPLUS1.get_unchecked(idx);
      }
      let a = v128_load(a_vals.as_ptr() as *const v128);

      let b_term1 = i32x4_sub(sgr_bits_vec, a);
      let b = i32x4_mul(i32x4_mul(b_term1, sum), one_over_n_vec);
      let b_shifted =
        u32x4_shr(i32x4_add(b, recip_bits_vec), SGRPROJ_RECIP_BITS as u32);

      v128_store(af.as_mut_ptr().add(x) as *mut v128, a);
      v128_store(bf.as_mut_ptr().add(x) as *mut v128, b_shifted);
    } else {
      rust::sgrproj_box_ab_internal::<BD>(
        r,
        af,
        bf,
        iimg,
        iimg_sq,
        iimg_stride,
        x,
        y,
        stripe_w,
        s,
      );
    }
  }
}

unsafe fn sgrproj_box_f_r0_wasm<T: Pixel>(
  f: &mut [u32], y: usize, w: usize, cdeffed: &PlaneSlice<T>,
) {
  let shift = SGRPROJ_RST_BITS as u32;

  for x in (0..w).step_by(4) {
    if x + 4 <= w {
      let val_i32 = if mem::size_of::<T>() == 1 {
        let ptr = cdeffed.subslice(x, y).as_ptr() as *const u32;
        let v_u8 = v128_load32_zero(ptr);
        let v_u16 = u16x8_extend_low_u8x16(v_u8);
        i32x4_extend_low_u16x8(v_u16)
      } else {
        let ptr = cdeffed.subslice(x, y).as_ptr() as *const u64;
        let v_u16 = v128_load64_zero(ptr);
        i32x4_extend_low_u16x8(v_u16)
      };

      let v_shifted = i32x4_shl(val_i32, shift);
      v128_store(f.as_mut_ptr().add(x) as *mut v128, v_shifted);
    } else {
      rust::sgrproj_box_f_r0_internal(f, x, y, w, cdeffed);
    }
  }
}

unsafe fn sgrproj_box_f_r1_wasm<T: Pixel>(
  af: &[&[u32]; 3], bf: &[&[u32]; 3], f: &mut [u32], y: usize, w: usize,
  cdeffed: &PlaneSlice<T>,
) {
  let three = i32x4_splat(3);
  let four = i32x4_splat(4);
  let shift = (5 + SGRPROJ_SGR_BITS - SGRPROJ_RST_BITS) as u32;
  let rounding = i32x4_splat(1 << shift >> 1);

  let a0_ptr = af[0].as_ptr();
  let a1_ptr = af[1].as_ptr();
  let a2_ptr = af[2].as_ptr();
  let b0_ptr = bf[0].as_ptr();
  let b1_ptr = bf[1].as_ptr();
  let b2_ptr = bf[2].as_ptr();

  for x in (0..w).step_by(4) {
    if x + 4 <= w {
      let load_at = |ptr: *const u32, off: usize| {
        v128_load(ptr.add(x + off) as *const v128)
      };

      let a0_x = load_at(a0_ptr, 0);
      let a0_x1 = load_at(a0_ptr, 1);
      let a0_x2 = load_at(a0_ptr, 2);

      let a1_x = load_at(a1_ptr, 0);
      let a1_x1 = load_at(a1_ptr, 1);
      let a1_x2 = load_at(a1_ptr, 2);

      let a2_x = load_at(a2_ptr, 0);
      let a2_x1 = load_at(a2_ptr, 1);
      let a2_x2 = load_at(a2_ptr, 2);

      let b0_x = load_at(b0_ptr, 0);
      let b0_x1 = load_at(b0_ptr, 1);
      let b0_x2 = load_at(b0_ptr, 2);

      let b1_x = load_at(b1_ptr, 0);
      let b1_x1 = load_at(b1_ptr, 1);
      let b1_x2 = load_at(b1_ptr, 2);

      let b2_x = load_at(b2_ptr, 0);
      let b2_x1 = load_at(b2_ptr, 1);
      let b2_x2 = load_at(b2_ptr, 2);

      let a_sum1 = i32x4_add(i32x4_add(a0_x, a2_x), i32x4_add(a0_x2, a2_x2));
      let a_term1 = i32x4_mul(a_sum1, three);

      let a_sum2 = i32x4_add(
        i32x4_add(a1_x, a1_x2),
        i32x4_add(a0_x1, i32x4_add(a1_x1, a2_x1)),
      );
      let a_term2 = i32x4_mul(a_sum2, four);

      let a = i32x4_add(a_term1, a_term2);

      let b_sum1 = i32x4_add(i32x4_add(b0_x, b2_x), i32x4_add(b0_x2, b2_x2));
      let b_term1 = i32x4_mul(b_sum1, three);

      let b_sum2 = i32x4_add(
        i32x4_add(b1_x, b1_x2),
        i32x4_add(b0_x1, i32x4_add(b1_x1, b2_x1)),
      );
      let b_term2 = i32x4_mul(b_sum2, four);

      let b = i32x4_add(b_term1, b_term2);

      let p_val = if mem::size_of::<T>() == 1 {
        let ptr = cdeffed.subslice(x, y).as_ptr() as *const u32;
        let v_u8 = v128_load32_zero(ptr);
        let v_u16 = u16x8_extend_low_u8x16(v_u8);
        i32x4_extend_low_u16x8(v_u16)
      } else {
        let ptr = cdeffed.subslice(x, y).as_ptr() as *const u64;
        let v_u16 = v128_load64_zero(ptr);
        i32x4_extend_low_u16x8(v_u16)
      };

      let v = i32x4_add(i32x4_mul(a, p_val), b);
      let res = u32x4_shr(i32x4_add(v, rounding), shift);

      v128_store(f.as_mut_ptr().add(x) as *mut v128, res);
    } else {
      rust::sgrproj_box_f_r1_internal(af, bf, f, x, y, w, cdeffed);
    }
  }
}

unsafe fn sgrproj_box_f_r2_wasm<T: Pixel>(
  af: &[&[u32]; 2], bf: &[&[u32]; 2], f0: &mut [u32], f1: &mut [u32],
  y: usize, w: usize, cdeffed: &PlaneSlice<T>,
) {
  let five = i32x4_splat(5);
  let six = i32x4_splat(6);
  let shift = (5 + SGRPROJ_SGR_BITS - SGRPROJ_RST_BITS) as u32;
  let shifto = (4 + SGRPROJ_SGR_BITS - SGRPROJ_RST_BITS) as u32;
  let rounding = i32x4_splat(1 << shift >> 1);
  let roundingo = i32x4_splat(1 << shifto >> 1);

  let a0_ptr = af[0].as_ptr();
  let a1_ptr = af[1].as_ptr();
  let b0_ptr = bf[0].as_ptr();
  let b1_ptr = bf[1].as_ptr();

  for x in (0..w).step_by(4) {
    if x + 4 <= w {
      let load_at = |ptr: *const u32, off: usize| {
        v128_load(ptr.add(x + off) as *const v128)
      };

      let a0_x = load_at(a0_ptr, 0);
      let a0_x1 = load_at(a0_ptr, 1);
      let a0_x2 = load_at(a0_ptr, 2);

      let a1_x = load_at(a1_ptr, 0);
      let a1_x1 = load_at(a1_ptr, 1);
      let a1_x2 = load_at(a1_ptr, 2);

      let b0_x = load_at(b0_ptr, 0);
      let b0_x1 = load_at(b0_ptr, 1);
      let b0_x2 = load_at(b0_ptr, 2);

      let b1_x = load_at(b1_ptr, 0);
      let b1_x1 = load_at(b1_ptr, 1);
      let b1_x2 = load_at(b1_ptr, 2);

      let a = i32x4_add(
        i32x4_mul(i32x4_add(a0_x, a0_x2), five),
        i32x4_mul(a0_x1, six),
      );
      let b = i32x4_add(
        i32x4_mul(i32x4_add(b0_x, b0_x2), five),
        i32x4_mul(b0_x1, six),
      );
      let ao = i32x4_add(
        i32x4_mul(i32x4_add(a1_x, a1_x2), five),
        i32x4_mul(a1_x1, six),
      );
      let bo = i32x4_add(
        i32x4_mul(i32x4_add(b1_x, b1_x2), five),
        i32x4_mul(b1_x1, six),
      );

      let p0_val = if mem::size_of::<T>() == 1 {
        let ptr = cdeffed.subslice(x, y).as_ptr() as *const u32;
        let v_u8 = v128_load32_zero(ptr);
        let v_u16 = u16x8_extend_low_u8x16(v_u8);
        i32x4_extend_low_u16x8(v_u16)
      } else {
        let ptr = cdeffed.subslice(x, y).as_ptr() as *const u64;
        let v_u16 = v128_load64_zero(ptr);
        i32x4_extend_low_u16x8(v_u16)
      };

      let p1_val = if mem::size_of::<T>() == 1 {
        let ptr = cdeffed.subslice(x, y + 1).as_ptr() as *const u32;
        let v_u8 = v128_load32_zero(ptr);
        let v_u16 = u16x8_extend_low_u8x16(v_u8);
        i32x4_extend_low_u16x8(v_u16)
      } else {
        let ptr = cdeffed.subslice(x, y + 1).as_ptr() as *const u64;
        let v_u16 = v128_load64_zero(ptr);
        i32x4_extend_low_u16x8(v_u16)
      };

      let v = i32x4_add(i32x4_mul(i32x4_add(a, ao), p0_val), i32x4_add(b, bo));
      let res0 = u32x4_shr(i32x4_add(v, rounding), shift);
      v128_store(f0.as_mut_ptr().add(x) as *mut v128, res0);

      let vo = i32x4_add(i32x4_mul(ao, p1_val), bo);
      let res1 = u32x4_shr(i32x4_add(vo, roundingo), shifto);
      v128_store(f1.as_mut_ptr().add(x) as *mut v128, res1);
    } else {
      rust::sgrproj_box_f_r2_internal(af, bf, f0, f1, x, y, w, cdeffed);
    }
  }
}
