use crate::util::CastFromPrimitive;
use num_traits::{AsPrimitive, PrimInt, Signed};
use std::fmt::Debug;
use std::ops::AddAssign;
use v_frame::pixel::Pixel as VFramePixel;

/// Enum to identify bit depth at runtime
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelType {
  U8,
  U16,
}

/// Trait alias for primitives used in the library (shim for v_frame 0.4 removal)
pub trait RegisteredPrimitive: PrimInt + Send + Sync + 'static {}
impl<T: PrimInt + Send + Sync + 'static> RegisteredPrimitive for T {}

/// Extension trait for pixels (u8, u16)
pub trait Pixel:
  VFramePixel
  + AsPrimitive<i16>
  + AsPrimitive<i32>
  + Debug
  + CastFromPrimitive<i32>
{
  type Coeff: Coefficient<Pixel = Self>;

  fn type_enum() -> PixelType;
  fn to_i32(self) -> i32;
  fn to_i16(self) -> i16;
  fn to_u16(self) -> u16;
  fn to_u32(self) -> u32;
  fn to_asm_stride(stride: usize) -> isize;
}

/// Trait for transform coefficients (i16, i32)
pub trait Coefficient:
  RegisteredPrimitive
  + Into<i32>
  + AddAssign
  + Signed
  + Debug
  + 'static
  + CastFromPrimitive<i32>
  + CastFromPrimitive<usize>
{
  type Pixel: Pixel<Coeff = Self>;

  fn to_u32(self) -> u32;
  fn to_i32(self) -> i32;
  fn to_i16(self) -> i16;
  fn as_u8(self) -> u8;
}

impl Pixel for u8 {
  type Coeff = i16;

  #[inline(always)]
  fn type_enum() -> PixelType {
    PixelType::U8
  }

  #[inline(always)]
  fn to_i32(self) -> i32 {
    self as i32
  }

  #[inline(always)]
  fn to_i16(self) -> i16 {
    self as i16
  }

  #[inline(always)]
  fn to_u16(self) -> u16 {
    self as u16
  }

  #[inline(always)]
  fn to_u32(self) -> u32 {
    self as u32
  }

  #[inline(always)]
  fn to_asm_stride(stride: usize) -> isize {
    stride as isize
  }
}

impl Pixel for u16 {
  type Coeff = i32;

  #[inline(always)]
  fn type_enum() -> PixelType {
    PixelType::U16
  }

  #[inline(always)]
  fn to_i32(self) -> i32 {
    self as i32
  }

  #[inline(always)]
  fn to_i16(self) -> i16 {
    self as i16
  }

  #[inline(always)]
  fn to_u16(self) -> u16 {
    self
  }

  #[inline(always)]
  fn to_u32(self) -> u32 {
    self as u32
  }

  #[inline(always)]
  fn to_asm_stride(stride: usize) -> isize {
    stride as isize
  }
}

impl Coefficient for i16 {
  type Pixel = u8;

  #[inline(always)]
  fn to_u32(self) -> u32 {
    self as u32
  }
  #[inline(always)]
  fn to_i32(self) -> i32 {
    self as i32
  }
  #[inline(always)]
  fn to_i16(self) -> i16 {
    self
  }
  #[inline(always)]
  fn as_u8(self) -> u8 {
    self as u8
  }
}

impl Coefficient for i32 {
  type Pixel = u16;

  #[inline(always)]
  fn to_u32(self) -> u32 {
    self as u32
  }
  #[inline(always)]
  fn to_i32(self) -> i32 {
    self
  }
  #[inline(always)]
  fn to_i16(self) -> i16 {
    self as i16
  }
  #[inline(always)]
  fn as_u8(self) -> u8 {
    self as u8
  }
}
