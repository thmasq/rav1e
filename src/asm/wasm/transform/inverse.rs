use crate::cpu_features::CpuFeatureLevel;
use crate::tiling::PlaneRegionMut;
use crate::transform::{TxSize, TxType};
use crate::util::Pixel;

use std::arch::wasm32::*;

const SQRT2: i32 = 5793;
const INV_COS_BIT: i32 = 12;

static COSPI_INV: [i32; 64] = [
  4096, 4095, 4091, 4085, 4076, 4065, 4052, 4036, 4017, 3996, 3973, 3948,
  3920, 3889, 3857, 3822, 3784, 3745, 3703, 3659, 3612, 3564, 3513, 3461,
  3406, 3349, 3290, 3229, 3166, 3102, 3035, 2967, 2896, 2824, 2751, 2675,
  2598, 2520, 2440, 2359, 2276, 2191, 2106, 2019, 1931, 1842, 1751, 1660,
  1567, 1474, 1380, 1285, 1189, 1092, 995, 897, 799, 700, 601, 501, 401, 301,
  201, 101,
];

static SINPI_INV: [i32; 5] = [0, 1321, 2482, 3344, 3803];

#[derive(Debug, Clone, Copy, PartialEq)]
enum TxType1D {
  DCT,
  ADST,
  FLIPADST,
  IDTX,
  WHT,
}

const fn get_1d_tx_types(tx_type: TxType) -> (TxType1D, TxType1D) {
  match tx_type {
    TxType::DCT_DCT => (TxType1D::DCT, TxType1D::DCT),
    TxType::ADST_DCT => (TxType1D::ADST, TxType1D::DCT),
    TxType::DCT_ADST => (TxType1D::DCT, TxType1D::ADST),
    TxType::ADST_ADST => (TxType1D::ADST, TxType1D::ADST),
    TxType::FLIPADST_DCT => (TxType1D::FLIPADST, TxType1D::DCT),
    TxType::DCT_FLIPADST => (TxType1D::DCT, TxType1D::FLIPADST),
    TxType::FLIPADST_FLIPADST => (TxType1D::FLIPADST, TxType1D::FLIPADST),
    TxType::ADST_FLIPADST => (TxType1D::ADST, TxType1D::FLIPADST),
    TxType::FLIPADST_ADST => (TxType1D::FLIPADST, TxType1D::ADST),
    TxType::IDTX => (TxType1D::IDTX, TxType1D::IDTX),
    TxType::V_DCT => (TxType1D::DCT, TxType1D::IDTX),
    TxType::H_DCT => (TxType1D::IDTX, TxType1D::DCT),
    TxType::V_ADST => (TxType1D::ADST, TxType1D::IDTX),
    TxType::H_ADST => (TxType1D::IDTX, TxType1D::ADST),
    TxType::V_FLIPADST => (TxType1D::FLIPADST, TxType1D::IDTX),
    TxType::H_FLIPADST => (TxType1D::IDTX, TxType1D::FLIPADST),
    TxType::WHT_WHT => (TxType1D::WHT, TxType1D::WHT),
  }
}

#[inline]
unsafe fn round_shift(val: v128, bit: i32) -> v128 {
  let bias = i32x4_splat(1 << (bit - 1));
  let v = i32x4_add(val, bias);
  i32x4_shr(v, bit as u32)
}

#[inline]
unsafe fn half_btf(w0: i32, in0: v128, w1: i32, in1: v128, bit: i32) -> v128 {
  let w0_v = i32x4_splat(w0);
  let w1_v = i32x4_splat(w1);
  let p0 = i32x4_mul(w0_v, in0);
  let p1 = i32x4_mul(w1_v, in1);
  round_shift(i32x4_add(p0, p1), bit)
}

#[inline]
unsafe fn clamp_value(val: v128, range: usize) -> v128 {
  let min = -(1 << (range - 1));
  let max = (1 << (range - 1)) - 1;
  let min_v = i32x4_splat(min);
  let max_v = i32x4_splat(max);
  i32x4_min(max_v, i32x4_max(min_v, val))
}

#[inline]
unsafe fn transpose_4x4(io: &mut [v128; 4]) {
  let t0 = u32x4_shuffle::<0, 4, 1, 5>(io[0], io[1]);
  let t1 = u32x4_shuffle::<2, 6, 3, 7>(io[0], io[1]);
  let t2 = u32x4_shuffle::<0, 4, 1, 5>(io[2], io[3]);
  let t3 = u32x4_shuffle::<2, 6, 3, 7>(io[2], io[3]);

  io[0] = u32x4_shuffle::<0, 1, 4, 5>(t0, t2);
  io[1] = u32x4_shuffle::<2, 3, 6, 7>(t0, t2);
  io[2] = u32x4_shuffle::<0, 1, 4, 5>(t1, t3);
  io[3] = u32x4_shuffle::<2, 3, 6, 7>(t1, t3);
}

unsafe fn idct4(input: &[v128], output: &mut [v128], range: usize) {
  let stg1 = [input[0], input[2], input[1], input[3]];

  let stg2 = [
    half_btf(COSPI_INV[32], stg1[0], COSPI_INV[32], stg1[1], INV_COS_BIT),
    half_btf(COSPI_INV[32], stg1[0], -COSPI_INV[32], stg1[1], INV_COS_BIT),
    half_btf(COSPI_INV[48], stg1[2], -COSPI_INV[16], stg1[3], INV_COS_BIT),
    half_btf(COSPI_INV[16], stg1[2], COSPI_INV[48], stg1[3], INV_COS_BIT),
  ];

  output[0] = clamp_value(i32x4_add(stg2[0], stg2[3]), range);
  output[1] = clamp_value(i32x4_add(stg2[1], stg2[2]), range);
  output[2] = clamp_value(i32x4_sub(stg2[1], stg2[2]), range);
  output[3] = clamp_value(i32x4_sub(stg2[0], stg2[3]), range);
}

unsafe fn iadst4(input: &[v128], output: &mut [v128], _range: usize) {
  let bit = 12;
  let x0 = input[0];
  let x1 = input[1];
  let x2 = input[2];
  let x3 = input[3];

  let sinpi_1 = i32x4_splat(SINPI_INV[1]);
  let sinpi_2 = i32x4_splat(SINPI_INV[2]);
  let sinpi_3 = i32x4_splat(SINPI_INV[3]);
  let sinpi_4 = i32x4_splat(SINPI_INV[4]);

  let s0 = i32x4_mul(sinpi_1, x0);
  let s1 = i32x4_mul(sinpi_2, x0);
  let s2 = i32x4_mul(sinpi_3, x1);
  let s3 = i32x4_mul(sinpi_4, x2);
  let s4 = i32x4_mul(sinpi_1, x2);
  let s5 = i32x4_mul(sinpi_2, x3);
  let s6 = i32x4_mul(sinpi_4, x3);
  let s7 = i32x4_add(i32x4_sub(x0, x2), x3);

  let s0 = i32x4_add(s0, s3);
  let s1 = i32x4_sub(s1, s4);
  let s3 = s2;
  let s2 = i32x4_mul(sinpi_3, s7);

  let s0 = i32x4_add(s0, s5);
  let s1 = i32x4_sub(s1, s6);

  let x0 = i32x4_add(s0, s3);
  let x1 = i32x4_add(s1, s3);
  let x2 = s2;
  let x3 = i32x4_sub(i32x4_add(s0, s1), s3);

  output[0] = round_shift(x0, bit);
  output[1] = round_shift(x1, bit);
  output[2] = round_shift(x2, bit);
  output[3] = round_shift(x3, bit);
}

unsafe fn iidentity4(input: &[v128], output: &mut [v128], _range: usize) {
  let sqrt2 = i32x4_splat(SQRT2);
  for (out, inp) in output.iter_mut().zip(input.iter()) {
    *out = round_shift(i32x4_mul(sqrt2, *inp), 12);
  }
}

unsafe fn iwht4(input: &[v128], output: &mut [v128], _range: usize) {
  let x0 = input[0];
  let x1 = input[1];
  let x2 = input[2];
  let x3 = input[3];
  let s0 = i32x4_add(x0, x1);
  let s2 = i32x4_sub(x2, x3);
  let s4 = i32x4_shr(i32x4_sub(s0, s2), 1);
  let s3 = i32x4_sub(s4, x3);
  let s1 = i32x4_sub(s4, x1);
  output[0] = i32x4_sub(s0, s3);
  output[1] = s3;
  output[2] = s1;
  output[3] = i32x4_add(s2, s1);
}

unsafe fn idct8(input: &[v128], output: &mut [v128], range: usize) {
  let temp_in = [input[0], input[2], input[4], input[6]];
  let mut temp_out: [v128; 4] = [i32x4_splat(0); 4];
  idct4(&temp_in, &mut temp_out, range);

  let stg1 = [input[1], input[5], input[3], input[7]];

  let stg2 = [
    half_btf(COSPI_INV[56], stg1[0], -COSPI_INV[8], stg1[3], INV_COS_BIT),
    half_btf(COSPI_INV[24], stg1[1], -COSPI_INV[40], stg1[2], INV_COS_BIT),
    half_btf(COSPI_INV[40], stg1[1], COSPI_INV[24], stg1[2], INV_COS_BIT),
    half_btf(COSPI_INV[8], stg1[0], COSPI_INV[56], stg1[3], INV_COS_BIT),
  ];

  let stg3 = [
    clamp_value(i32x4_add(stg2[0], stg2[1]), range),
    clamp_value(i32x4_sub(stg2[0], stg2[1]), range),
    clamp_value(i32x4_add(i32x4_neg(stg2[2]), stg2[3]), range),
    clamp_value(i32x4_add(stg2[2], stg2[3]), range),
  ];

  let stg4 = [
    stg3[0],
    half_btf(-COSPI_INV[32], stg3[1], COSPI_INV[32], stg3[2], INV_COS_BIT),
    half_btf(COSPI_INV[32], stg3[1], COSPI_INV[32], stg3[2], INV_COS_BIT),
    stg3[3],
  ];

  output[0] = clamp_value(i32x4_add(temp_out[0], stg4[3]), range);
  output[1] = clamp_value(i32x4_add(temp_out[1], stg4[2]), range);
  output[2] = clamp_value(i32x4_add(temp_out[2], stg4[1]), range);
  output[3] = clamp_value(i32x4_add(temp_out[3], stg4[0]), range);
  output[4] = clamp_value(i32x4_sub(temp_out[3], stg4[0]), range);
  output[5] = clamp_value(i32x4_sub(temp_out[2], stg4[1]), range);
  output[6] = clamp_value(i32x4_sub(temp_out[1], stg4[2]), range);
  output[7] = clamp_value(i32x4_sub(temp_out[0], stg4[3]), range);
}

unsafe fn iadst8(input: &[v128], output: &mut [v128], range: usize) {
  let stg1 = [
    input[7], input[0], input[5], input[2], input[3], input[4], input[1],
    input[6],
  ];

  let stg2 = [
    half_btf(COSPI_INV[4], stg1[0], COSPI_INV[60], stg1[1], INV_COS_BIT),
    half_btf(COSPI_INV[60], stg1[0], -COSPI_INV[4], stg1[1], INV_COS_BIT),
    half_btf(COSPI_INV[20], stg1[2], COSPI_INV[44], stg1[3], INV_COS_BIT),
    half_btf(COSPI_INV[44], stg1[2], -COSPI_INV[20], stg1[3], INV_COS_BIT),
    half_btf(COSPI_INV[36], stg1[4], COSPI_INV[28], stg1[5], INV_COS_BIT),
    half_btf(COSPI_INV[28], stg1[4], -COSPI_INV[36], stg1[5], INV_COS_BIT),
    half_btf(COSPI_INV[52], stg1[6], COSPI_INV[12], stg1[7], INV_COS_BIT),
    half_btf(COSPI_INV[12], stg1[6], -COSPI_INV[52], stg1[7], INV_COS_BIT),
  ];

  let stg3 = [
    clamp_value(i32x4_add(stg2[0], stg2[4]), range),
    clamp_value(i32x4_add(stg2[1], stg2[5]), range),
    clamp_value(i32x4_add(stg2[2], stg2[6]), range),
    clamp_value(i32x4_add(stg2[3], stg2[7]), range),
    clamp_value(i32x4_sub(stg2[0], stg2[4]), range),
    clamp_value(i32x4_sub(stg2[1], stg2[5]), range),
    clamp_value(i32x4_sub(stg2[2], stg2[6]), range),
    clamp_value(i32x4_sub(stg2[3], stg2[7]), range),
  ];

  let stg4 = [
    stg3[0],
    stg3[1],
    stg3[2],
    stg3[3],
    half_btf(COSPI_INV[16], stg3[4], COSPI_INV[48], stg3[5], INV_COS_BIT),
    half_btf(COSPI_INV[48], stg3[4], -COSPI_INV[16], stg3[5], INV_COS_BIT),
    half_btf(-COSPI_INV[48], stg3[6], COSPI_INV[16], stg3[7], INV_COS_BIT),
    half_btf(COSPI_INV[16], stg3[6], COSPI_INV[48], stg3[7], INV_COS_BIT),
  ];

  let stg5 = [
    clamp_value(i32x4_add(stg4[0], stg4[2]), range),
    clamp_value(i32x4_add(stg4[1], stg4[3]), range),
    clamp_value(i32x4_sub(stg4[0], stg4[2]), range),
    clamp_value(i32x4_sub(stg4[1], stg4[3]), range),
    clamp_value(i32x4_add(stg4[4], stg4[6]), range),
    clamp_value(i32x4_add(stg4[5], stg4[7]), range),
    clamp_value(i32x4_sub(stg4[4], stg4[6]), range),
    clamp_value(i32x4_sub(stg4[5], stg4[7]), range),
  ];

  let stg6 = [
    stg5[0],
    stg5[1],
    half_btf(COSPI_INV[32], stg5[2], COSPI_INV[32], stg5[3], INV_COS_BIT),
    half_btf(COSPI_INV[32], stg5[2], -COSPI_INV[32], stg5[3], INV_COS_BIT),
    stg5[4],
    stg5[5],
    half_btf(COSPI_INV[32], stg5[6], COSPI_INV[32], stg5[7], INV_COS_BIT),
    half_btf(COSPI_INV[32], stg5[6], -COSPI_INV[32], stg5[7], INV_COS_BIT),
  ];

  output[0] = stg6[0];
  output[1] = i32x4_neg(stg6[4]);
  output[2] = stg6[6];
  output[3] = i32x4_neg(stg6[2]);
  output[4] = stg6[3];
  output[5] = i32x4_neg(stg6[7]);
  output[6] = stg6[5];
  output[7] = i32x4_neg(stg6[1]);
}

unsafe fn iidentity8(input: &[v128], output: &mut [v128], _range: usize) {
  for (out, inp) in output.iter_mut().zip(input.iter()) {
    *out = i32x4_shl(*inp, 1);
  }
}

unsafe fn idct16(input: &[v128], output: &mut [v128], range: usize) {
  let temp_in = [
    input[0], input[2], input[4], input[6], input[8], input[10], input[12],
    input[14],
  ];
  let mut temp_out: [v128; 8] = [i32x4_splat(0); 8];
  idct8(&temp_in, &mut temp_out, range);

  let stg1 = [
    input[1], input[9], input[5], input[13], input[3], input[11], input[7],
    input[15],
  ];

  let stg2 = [
    half_btf(COSPI_INV[60], stg1[0], -COSPI_INV[4], stg1[7], INV_COS_BIT),
    half_btf(COSPI_INV[28], stg1[1], -COSPI_INV[36], stg1[6], INV_COS_BIT),
    half_btf(COSPI_INV[44], stg1[2], -COSPI_INV[20], stg1[5], INV_COS_BIT),
    half_btf(COSPI_INV[12], stg1[3], -COSPI_INV[52], stg1[4], INV_COS_BIT),
    half_btf(COSPI_INV[52], stg1[3], COSPI_INV[12], stg1[4], INV_COS_BIT),
    half_btf(COSPI_INV[20], stg1[2], COSPI_INV[44], stg1[5], INV_COS_BIT),
    half_btf(COSPI_INV[36], stg1[1], COSPI_INV[28], stg1[6], INV_COS_BIT),
    half_btf(COSPI_INV[4], stg1[0], COSPI_INV[60], stg1[7], INV_COS_BIT),
  ];

  let stg3 = [
    clamp_value(i32x4_add(stg2[0], stg2[1]), range),
    clamp_value(i32x4_sub(stg2[0], stg2[1]), range),
    clamp_value(i32x4_add(i32x4_neg(stg2[2]), stg2[3]), range),
    clamp_value(i32x4_add(stg2[2], stg2[3]), range),
    clamp_value(i32x4_add(stg2[4], stg2[5]), range),
    clamp_value(i32x4_sub(stg2[4], stg2[5]), range),
    clamp_value(i32x4_add(i32x4_neg(stg2[6]), stg2[7]), range),
    clamp_value(i32x4_add(stg2[6], stg2[7]), range),
  ];

  let stg4 = [
    stg3[0],
    half_btf(-COSPI_INV[16], stg3[1], COSPI_INV[48], stg3[6], INV_COS_BIT),
    half_btf(-COSPI_INV[48], stg3[2], -COSPI_INV[16], stg3[5], INV_COS_BIT),
    stg3[3],
    stg3[4],
    half_btf(-COSPI_INV[16], stg3[2], COSPI_INV[48], stg3[5], INV_COS_BIT),
    half_btf(COSPI_INV[48], stg3[1], COSPI_INV[16], stg3[6], INV_COS_BIT),
    stg3[7],
  ];

  let stg5 = [
    clamp_value(i32x4_add(stg4[0], stg4[3]), range),
    clamp_value(i32x4_add(stg4[1], stg4[2]), range),
    clamp_value(i32x4_sub(stg4[1], stg4[2]), range),
    clamp_value(i32x4_sub(stg4[0], stg4[3]), range),
    clamp_value(i32x4_add(i32x4_neg(stg4[4]), stg4[7]), range),
    clamp_value(i32x4_add(i32x4_neg(stg4[5]), stg4[6]), range),
    clamp_value(i32x4_add(stg4[5], stg4[6]), range),
    clamp_value(i32x4_add(stg4[4], stg4[7]), range),
  ];

  let stg6 = [
    stg5[0],
    stg5[1],
    half_btf(-COSPI_INV[32], stg5[2], COSPI_INV[32], stg5[5], INV_COS_BIT),
    half_btf(-COSPI_INV[32], stg5[3], COSPI_INV[32], stg5[4], INV_COS_BIT),
    half_btf(COSPI_INV[32], stg5[3], COSPI_INV[32], stg5[4], INV_COS_BIT),
    half_btf(COSPI_INV[32], stg5[2], COSPI_INV[32], stg5[5], INV_COS_BIT),
    stg5[6],
    stg5[7],
  ];

  output[0] = clamp_value(i32x4_add(temp_out[0], stg6[7]), range);
  output[1] = clamp_value(i32x4_add(temp_out[1], stg6[6]), range);
  output[2] = clamp_value(i32x4_add(temp_out[2], stg6[5]), range);
  output[3] = clamp_value(i32x4_add(temp_out[3], stg6[4]), range);
  output[4] = clamp_value(i32x4_add(temp_out[4], stg6[3]), range);
  output[5] = clamp_value(i32x4_add(temp_out[5], stg6[2]), range);
  output[6] = clamp_value(i32x4_add(temp_out[6], stg6[1]), range);
  output[7] = clamp_value(i32x4_add(temp_out[7], stg6[0]), range);
  output[8] = clamp_value(i32x4_sub(temp_out[7], stg6[0]), range);
  output[9] = clamp_value(i32x4_sub(temp_out[6], stg6[1]), range);
  output[10] = clamp_value(i32x4_sub(temp_out[5], stg6[2]), range);
  output[11] = clamp_value(i32x4_sub(temp_out[4], stg6[3]), range);
  output[12] = clamp_value(i32x4_sub(temp_out[3], stg6[4]), range);
  output[13] = clamp_value(i32x4_sub(temp_out[2], stg6[5]), range);
  output[14] = clamp_value(i32x4_sub(temp_out[1], stg6[6]), range);
  output[15] = clamp_value(i32x4_sub(temp_out[0], stg6[7]), range);
}

unsafe fn iadst16(input: &[v128], output: &mut [v128], range: usize) {
  let stg1 = [
    input[15], input[0], input[13], input[2], input[11], input[4], input[9],
    input[6], input[7], input[8], input[5], input[10], input[3], input[12],
    input[1], input[14],
  ];

  let stg2 = [
    half_btf(COSPI_INV[2], stg1[0], COSPI_INV[62], stg1[1], INV_COS_BIT),
    half_btf(COSPI_INV[62], stg1[0], -COSPI_INV[2], stg1[1], INV_COS_BIT),
    half_btf(COSPI_INV[10], stg1[2], COSPI_INV[54], stg1[3], INV_COS_BIT),
    half_btf(COSPI_INV[54], stg1[2], -COSPI_INV[10], stg1[3], INV_COS_BIT),
    half_btf(COSPI_INV[18], stg1[4], COSPI_INV[46], stg1[5], INV_COS_BIT),
    half_btf(COSPI_INV[46], stg1[4], -COSPI_INV[18], stg1[5], INV_COS_BIT),
    half_btf(COSPI_INV[26], stg1[6], COSPI_INV[38], stg1[7], INV_COS_BIT),
    half_btf(COSPI_INV[38], stg1[6], -COSPI_INV[26], stg1[7], INV_COS_BIT),
    half_btf(COSPI_INV[34], stg1[8], COSPI_INV[30], stg1[9], INV_COS_BIT),
    half_btf(COSPI_INV[30], stg1[8], -COSPI_INV[34], stg1[9], INV_COS_BIT),
    half_btf(COSPI_INV[42], stg1[10], COSPI_INV[22], stg1[11], INV_COS_BIT),
    half_btf(COSPI_INV[22], stg1[10], -COSPI_INV[42], stg1[11], INV_COS_BIT),
    half_btf(COSPI_INV[50], stg1[12], COSPI_INV[14], stg1[13], INV_COS_BIT),
    half_btf(COSPI_INV[14], stg1[12], -COSPI_INV[50], stg1[13], INV_COS_BIT),
    half_btf(COSPI_INV[58], stg1[14], COSPI_INV[6], stg1[15], INV_COS_BIT),
    half_btf(COSPI_INV[6], stg1[14], -COSPI_INV[58], stg1[15], INV_COS_BIT),
  ];

  let stg3 = [
    clamp_value(i32x4_add(stg2[0], stg2[8]), range),
    clamp_value(i32x4_add(stg2[1], stg2[9]), range),
    clamp_value(i32x4_add(stg2[2], stg2[10]), range),
    clamp_value(i32x4_add(stg2[3], stg2[11]), range),
    clamp_value(i32x4_add(stg2[4], stg2[12]), range),
    clamp_value(i32x4_add(stg2[5], stg2[13]), range),
    clamp_value(i32x4_add(stg2[6], stg2[14]), range),
    clamp_value(i32x4_add(stg2[7], stg2[15]), range),
    clamp_value(i32x4_sub(stg2[0], stg2[8]), range),
    clamp_value(i32x4_sub(stg2[1], stg2[9]), range),
    clamp_value(i32x4_sub(stg2[2], stg2[10]), range),
    clamp_value(i32x4_sub(stg2[3], stg2[11]), range),
    clamp_value(i32x4_sub(stg2[4], stg2[12]), range),
    clamp_value(i32x4_sub(stg2[5], stg2[13]), range),
    clamp_value(i32x4_sub(stg2[6], stg2[14]), range),
    clamp_value(i32x4_sub(stg2[7], stg2[15]), range),
  ];

  let stg4 = [
    stg3[0],
    stg3[1],
    stg3[2],
    stg3[3],
    stg3[4],
    stg3[5],
    stg3[6],
    stg3[7],
    half_btf(COSPI_INV[8], stg3[8], COSPI_INV[56], stg3[9], INV_COS_BIT),
    half_btf(COSPI_INV[56], stg3[8], -COSPI_INV[8], stg3[9], INV_COS_BIT),
    half_btf(COSPI_INV[40], stg3[10], COSPI_INV[24], stg3[11], INV_COS_BIT),
    half_btf(COSPI_INV[24], stg3[10], -COSPI_INV[40], stg3[11], INV_COS_BIT),
    half_btf(-COSPI_INV[56], stg3[12], COSPI_INV[8], stg3[13], INV_COS_BIT),
    half_btf(COSPI_INV[8], stg3[12], COSPI_INV[56], stg3[13], INV_COS_BIT),
    half_btf(-COSPI_INV[24], stg3[14], COSPI_INV[40], stg3[15], INV_COS_BIT),
    half_btf(COSPI_INV[40], stg3[14], COSPI_INV[24], stg3[15], INV_COS_BIT),
  ];

  let stg5 = [
    clamp_value(i32x4_add(stg4[0], stg4[4]), range),
    clamp_value(i32x4_add(stg4[1], stg4[5]), range),
    clamp_value(i32x4_add(stg4[2], stg4[6]), range),
    clamp_value(i32x4_add(stg4[3], stg4[7]), range),
    clamp_value(i32x4_sub(stg4[0], stg4[4]), range),
    clamp_value(i32x4_sub(stg4[1], stg4[5]), range),
    clamp_value(i32x4_sub(stg4[2], stg4[6]), range),
    clamp_value(i32x4_sub(stg4[3], stg4[7]), range),
    clamp_value(i32x4_add(stg4[8], stg4[12]), range),
    clamp_value(i32x4_add(stg4[9], stg4[13]), range),
    clamp_value(i32x4_add(stg4[10], stg4[14]), range),
    clamp_value(i32x4_add(stg4[11], stg4[15]), range),
    clamp_value(i32x4_sub(stg4[8], stg4[12]), range),
    clamp_value(i32x4_sub(stg4[9], stg4[13]), range),
    clamp_value(i32x4_sub(stg4[10], stg4[14]), range),
    clamp_value(i32x4_sub(stg4[11], stg4[15]), range),
  ];

  let stg6 = [
    stg5[0],
    stg5[1],
    stg5[2],
    stg5[3],
    half_btf(COSPI_INV[16], stg5[4], COSPI_INV[48], stg5[5], INV_COS_BIT),
    half_btf(COSPI_INV[48], stg5[4], -COSPI_INV[16], stg5[5], INV_COS_BIT),
    half_btf(-COSPI_INV[48], stg5[6], COSPI_INV[16], stg5[7], INV_COS_BIT),
    half_btf(COSPI_INV[16], stg5[6], COSPI_INV[48], stg5[7], INV_COS_BIT),
    stg5[8],
    stg5[9],
    stg5[10],
    stg5[11],
    half_btf(COSPI_INV[16], stg5[12], COSPI_INV[48], stg5[13], INV_COS_BIT),
    half_btf(COSPI_INV[48], stg5[12], -COSPI_INV[16], stg5[13], INV_COS_BIT),
    half_btf(-COSPI_INV[48], stg5[14], COSPI_INV[16], stg5[15], INV_COS_BIT),
    half_btf(COSPI_INV[16], stg5[14], COSPI_INV[48], stg5[15], INV_COS_BIT),
  ];

  let stg7 = [
    clamp_value(i32x4_add(stg6[0], stg6[2]), range),
    clamp_value(i32x4_add(stg6[1], stg6[3]), range),
    clamp_value(i32x4_sub(stg6[0], stg6[2]), range),
    clamp_value(i32x4_sub(stg6[1], stg6[3]), range),
    clamp_value(i32x4_add(stg6[4], stg6[6]), range),
    clamp_value(i32x4_add(stg6[5], stg6[7]), range),
    clamp_value(i32x4_sub(stg6[4], stg6[6]), range),
    clamp_value(i32x4_sub(stg6[5], stg6[7]), range),
    clamp_value(i32x4_add(stg6[8], stg6[10]), range),
    clamp_value(i32x4_add(stg6[9], stg6[11]), range),
    clamp_value(i32x4_sub(stg6[8], stg6[10]), range),
    clamp_value(i32x4_sub(stg6[9], stg6[11]), range),
    clamp_value(i32x4_add(stg6[12], stg6[14]), range),
    clamp_value(i32x4_add(stg6[13], stg6[15]), range),
    clamp_value(i32x4_sub(stg6[12], stg6[14]), range),
    clamp_value(i32x4_sub(stg6[13], stg6[15]), range),
  ];

  let stg8 = [
    stg7[0],
    stg7[1],
    half_btf(COSPI_INV[32], stg7[2], COSPI_INV[32], stg7[3], INV_COS_BIT),
    half_btf(COSPI_INV[32], stg7[2], -COSPI_INV[32], stg7[3], INV_COS_BIT),
    stg7[4],
    stg7[5],
    half_btf(COSPI_INV[32], stg7[6], COSPI_INV[32], stg7[7], INV_COS_BIT),
    half_btf(COSPI_INV[32], stg7[6], -COSPI_INV[32], stg7[7], INV_COS_BIT),
    stg7[8],
    stg7[9],
    half_btf(COSPI_INV[32], stg7[10], COSPI_INV[32], stg7[11], INV_COS_BIT),
    half_btf(COSPI_INV[32], stg7[10], -COSPI_INV[32], stg7[11], INV_COS_BIT),
    stg7[12],
    stg7[13],
    half_btf(COSPI_INV[32], stg7[14], COSPI_INV[32], stg7[15], INV_COS_BIT),
    half_btf(COSPI_INV[32], stg7[14], -COSPI_INV[32], stg7[15], INV_COS_BIT),
  ];

  output[0] = stg8[0];
  output[1] = i32x4_neg(stg8[8]);
  output[2] = stg8[12];
  output[3] = i32x4_neg(stg8[4]);
  output[4] = stg8[6];
  output[5] = i32x4_neg(stg8[14]);
  output[6] = stg8[10];
  output[7] = i32x4_neg(stg8[2]);
  output[8] = stg8[3];
  output[9] = i32x4_neg(stg8[11]);
  output[10] = stg8[15];
  output[11] = i32x4_neg(stg8[7]);
  output[12] = stg8[5];
  output[13] = i32x4_neg(stg8[13]);
  output[14] = stg8[9];
  output[15] = i32x4_neg(stg8[1]);
}

unsafe fn iidentity16(input: &[v128], output: &mut [v128], _range: usize) {
  let sqrt2 = i32x4_splat(SQRT2);
  for (out, inp) in output.iter_mut().zip(input.iter()) {
    *out = round_shift(i32x4_mul(sqrt2, i32x4_shl(*inp, 1)), 12);
  }
}

unsafe fn inv_txfm_add_4x4(
  input: &[i16], output: &mut PlaneRegionMut<'_, u8>, tx_type: TxType,
  bd: usize,
) {
  let (tx_type_col, tx_type_row) = get_1d_tx_types(tx_type);
  let range = bd + 8;

  let mut work: [v128; 4] = [i32x4_splat(0); 4];
  for (i, row) in input.chunks(4).take(4).enumerate() {
    work[i] = i32x4_load_extend_i16x4(row.as_ptr());
  }
  transpose_4x4(&mut work);

  match tx_type_row {
    TxType1D::DCT => idct4(&work.clone(), &mut work, range),
    TxType1D::ADST => iadst4(&work.clone(), &mut work, range),
    TxType1D::IDTX => iidentity4(&work.clone(), &mut work, range),
    TxType1D::WHT => iwht4(&work.clone(), &mut work, range),
    TxType1D::FLIPADST => {
      iadst4(&work.clone(), &mut work, range);
      work.swap(0, 3);
      work.swap(1, 2);
    }
  }

  transpose_4x4(&mut work);

  match tx_type_col {
    TxType1D::DCT => idct4(&work.clone(), &mut work, range),
    TxType1D::ADST => iadst4(&work.clone(), &mut work, range),
    TxType1D::IDTX => iidentity4(&work.clone(), &mut work, range),
    TxType1D::WHT => iwht4(&work.clone(), &mut work, range),
    TxType1D::FLIPADST => {
      iadst4(&work.clone(), &mut work, range);
      work.swap(0, 3);
      work.swap(1, 2);
    }
  }

  for (i, v) in work.iter().enumerate() {
    let dest_ptr = output[i].as_mut_ptr();
    let d_u8 = v128_load32_zero(dest_ptr as *const u32);
    let d_i32 = u32x4_extend_low_u16x8(u16x8_extend_low_u8x16(d_u8));
    let res = i32x4_add(d_i32, *v);
    let res_clamped =
      i32x4_min(i32x4_splat(255), i32x4_max(i32x4_splat(0), res));
    let res_u16 = i16x8_narrow_i32x4(res_clamped, res_clamped);
    let res_u8 = u8x16_narrow_i16x8(res_u16, res_u16);
    v128_store32_lane::<0>(res_u8, dest_ptr as *mut u32);
  }
}

unsafe fn inv_txfm_add_generic(
  input: &[i16], output: &mut PlaneRegionMut<'_, u8>, tx_type: TxType,
  bd: usize, w: usize, h: usize,
) {
  let (tx_type_col, tx_type_row) = get_1d_tx_types(tx_type);
  let range = bd + 8;

  let mut buffer = [i32x4_splat(0); 64];

  for r in (0..h).step_by(4) {
    let mut work = [i32x4_splat(0); 16];

    for c in (0..w).step_by(4) {
      let mut blk = [i32x4_splat(0); 4];
      for i in 0..4 {
        if r + i < h {
          blk[i] = i32x4_load_extend_i16x4(input[(r + i) * w + c..].as_ptr());
        }
      }
      transpose_4x4(&mut blk);
      work[c] = blk[0];
      work[c + 1] = blk[1];
      work[c + 2] = blk[2];
      work[c + 3] = blk[3];
    }

    let mut res = [i32x4_splat(0); 16];
    match w {
      4 => match tx_type_row {
        TxType1D::DCT => idct4(&work[..4], &mut res[..4], range),
        TxType1D::ADST => iadst4(&work[..4], &mut res[..4], range),
        TxType1D::IDTX => iidentity4(&work[..4], &mut res[..4], range),
        TxType1D::WHT => iwht4(&work[..4], &mut res[..4], range),
        TxType1D::FLIPADST => {
          iadst4(&work[..4], &mut res[..4], range);
          res[..4].reverse();
        }
      },
      8 => match tx_type_row {
        TxType1D::DCT => idct8(&work[..8], &mut res[..8], range),
        TxType1D::ADST => iadst8(&work[..8], &mut res[..8], range),
        TxType1D::IDTX => iidentity8(&work[..8], &mut res[..8], range),
        TxType1D::FLIPADST => {
          iadst8(&work[..8], &mut res[..8], range);
          res[..8].reverse();
        }
        _ => unreachable!(),
      },
      16 => match tx_type_row {
        TxType1D::DCT => idct16(&work[..16], &mut res[..16], range),
        TxType1D::ADST => iadst16(&work[..16], &mut res[..16], range),
        TxType1D::IDTX => iidentity16(&work[..16], &mut res[..16], range),
        TxType1D::FLIPADST => {
          iadst16(&work[..16], &mut res[..16], range);
          res[..16].reverse();
        }
        _ => unreachable!(),
      },
      _ => unreachable!(),
    }

    for i in 0..w {
      buffer[i * (h / 4) + (r / 4)] = res[i];
    }
  }

  for c in (0..w).step_by(4) {
    let mut col_input = [i32x4_splat(0); 16];
    for r_chunk in 0..(h / 4) {
      let mut blk = [
        buffer[(c + 0) * (h / 4) + r_chunk],
        buffer[(c + 1) * (h / 4) + r_chunk],
        buffer[(c + 2) * (h / 4) + r_chunk],
        buffer[(c + 3) * (h / 4) + r_chunk],
      ];
      transpose_4x4(&mut blk);
      col_input[r_chunk * 4 + 0] = blk[0];
      col_input[r_chunk * 4 + 1] = blk[1];
      col_input[r_chunk * 4 + 2] = blk[2];
      col_input[r_chunk * 4 + 3] = blk[3];
    }

    let mut res = [i32x4_splat(0); 16];
    match h {
      4 => match tx_type_col {
        TxType1D::DCT => idct4(&col_input[..4], &mut res[..4], range),
        TxType1D::ADST => iadst4(&col_input[..4], &mut res[..4], range),
        TxType1D::IDTX => iidentity4(&col_input[..4], &mut res[..4], range),
        TxType1D::WHT => iwht4(&col_input[..4], &mut res[..4], range),
        TxType1D::FLIPADST => {
          iadst4(&col_input[..4], &mut res[..4], range);
          res[..4].reverse();
        }
      },
      8 => match tx_type_col {
        TxType1D::DCT => idct8(&col_input[..8], &mut res[..8], range),
        TxType1D::ADST => iadst8(&col_input[..8], &mut res[..8], range),
        TxType1D::IDTX => iidentity8(&col_input[..8], &mut res[..8], range),
        TxType1D::FLIPADST => {
          iadst8(&col_input[..8], &mut res[..8], range);
          res[..8].reverse();
        }
        _ => unreachable!(),
      },
      16 => match tx_type_col {
        TxType1D::DCT => idct16(&col_input[..16], &mut res[..16], range),
        TxType1D::ADST => iadst16(&col_input[..16], &mut res[..16], range),
        TxType1D::IDTX => iidentity16(&col_input[..16], &mut res[..16], range),
        TxType1D::FLIPADST => {
          iadst16(&col_input[..16], &mut res[..16], range);
          res[..16].reverse();
        }
        _ => unreachable!(),
      },
      _ => unreachable!(),
    }

    for i in 0..h {
      let dest_ptr = output[i].as_mut_ptr().add(c);
      let d_u8 = v128_load32_zero(dest_ptr as *const u32);
      let d_i32 = u32x4_extend_low_u16x8(u16x8_extend_low_u8x16(d_u8));
      let r_val = i32x4_add(d_i32, res[i]);
      let r_clamped =
        i32x4_min(i32x4_splat(255), i32x4_max(i32x4_splat(0), r_val));
      let r_u16 = i16x8_narrow_i32x4(r_clamped, r_clamped);
      let r_u8 = u8x16_narrow_i16x8(r_u16, r_u16);
      v128_store32_lane::<0>(r_u8, dest_ptr as *mut u32);
    }
  }
}

pub fn inverse_transform_add<T: Pixel>(
  input: &[T::Coeff], output: &mut PlaneRegionMut<'_, T>, eob: u16,
  tx_size: TxSize, tx_type: TxType, bd: usize, cpu: CpuFeatureLevel,
) where
  i32: crate::util::math::CastFromPrimitive<
    <T as crate::util::pixel::Pixel>::Coeff,
  >,
{
  if std::mem::size_of::<T>() != 1 {
    return crate::transform::inverse::rust::inverse_transform_add(
      input, output, eob, tx_size, tx_type, bd, cpu,
    );
  }

  let input_i16 = unsafe {
    std::slice::from_raw_parts(input.as_ptr() as *const i16, input.len())
  };
  let output_u8 = unsafe {
    std::mem::transmute::<
      &mut PlaneRegionMut<'_, T>,
      &mut PlaneRegionMut<'_, u8>,
    >(output)
  };

  match tx_size {
    TxSize::TX_4X4 => unsafe {
      inv_txfm_add_4x4(input_i16, output_u8, tx_type, bd)
    },
    TxSize::TX_8X8 => unsafe {
      inv_txfm_add_generic(input_i16, output_u8, tx_type, bd, 8, 8)
    },
    TxSize::TX_4X8 => unsafe {
      inv_txfm_add_generic(input_i16, output_u8, tx_type, bd, 4, 8)
    },
    TxSize::TX_8X4 => unsafe {
      inv_txfm_add_generic(input_i16, output_u8, tx_type, bd, 8, 4)
    },
    TxSize::TX_16X8 => unsafe {
      inv_txfm_add_generic(input_i16, output_u8, tx_type, bd, 16, 8)
    },
    TxSize::TX_8X16 => unsafe {
      inv_txfm_add_generic(input_i16, output_u8, tx_type, bd, 8, 16)
    },
    TxSize::TX_16X4 => unsafe {
      inv_txfm_add_generic(input_i16, output_u8, tx_type, bd, 16, 4)
    },
    TxSize::TX_4X16 => unsafe {
      inv_txfm_add_generic(input_i16, output_u8, tx_type, bd, 4, 16)
    },
    _ => {
      crate::transform::inverse::rust::inverse_transform_add(
        input, output, eob, tx_size, tx_type, bd, cpu,
      );
    }
  }
}
