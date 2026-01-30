pub trait Fixed {
  fn floor_log2(&self, n: usize) -> usize;
  fn ceil_log2(&self, n: usize) -> usize;
  fn align_power_of_two(&self, n: usize) -> usize;
  fn align_power_of_two_and_shift(&self, n: usize) -> usize;
}

impl Fixed for usize {
  #[inline]
  fn floor_log2(&self, n: usize) -> usize {
    self & !((1 << n) - 1)
  }

  #[inline]
  fn ceil_log2(&self, n: usize) -> usize {
    (self + (1 << n) - 1).floor_log2(n)
  }

  #[inline]
  fn align_power_of_two(&self, n: usize) -> usize {
    self.ceil_log2(n)
  }

  #[inline]
  fn align_power_of_two_and_shift(&self, n: usize) -> usize {
    (self + (1 << n) - 1) >> n
  }
}

#[inline(always)]
pub fn clamp<T: PartialOrd>(input: T, min: T, max: T) -> T {
  if input < min {
    min
  } else if input > max {
    max
  } else {
    input
  }
}

pub trait ILog {
  fn ilog(self) -> Self;
}

impl ILog for usize {
  fn ilog(self) -> Self {
    usize::BITS as usize - self.leading_zeros() as usize
  }
}

impl ILog for i16 {
  fn ilog(self) -> Self {
    (i16::BITS - self.leading_zeros()) as Self
  }
}

pub trait CastFromPrimitive<T> {
  fn cast_from(v: T) -> Self;
}

impl CastFromPrimitive<u8> for i16 {
  #[inline(always)]
  fn cast_from(v: u8) -> Self {
    v as i16
  }
}

impl CastFromPrimitive<i32> for u16 {
  #[inline(always)]
  fn cast_from(v: i32) -> Self {
    v as u16
  }
}

impl CastFromPrimitive<i32> for u8 {
  #[inline(always)]
  fn cast_from(v: i32) -> Self {
    v as u8
  }
}

impl CastFromPrimitive<u8> for i32 {
  #[inline(always)]
  fn cast_from(v: u8) -> Self {
    v as i32
  }
}

impl CastFromPrimitive<u16> for i32 {
  #[inline(always)]
  fn cast_from(v: u16) -> Self {
    v as i32
  }
}

impl CastFromPrimitive<usize> for i16 {
  #[inline(always)]
  fn cast_from(v: usize) -> Self {
    v as i16
  }
}

impl CastFromPrimitive<i32> for i16 {
  #[inline(always)]
  fn cast_from(v: i32) -> Self {
    v as i16
  }
}

impl CastFromPrimitive<usize> for i32 {
  #[inline(always)]
  fn cast_from(v: usize) -> Self {
    v as i32
  }
}

impl CastFromPrimitive<i32> for i32 {
  #[inline(always)]
  fn cast_from(v: i32) -> Self {
    v
  }
}

impl CastFromPrimitive<i16> for i32 {
  #[inline(always)]
  fn cast_from(v: i16) -> Self {
    v as i32
  }
}

impl CastFromPrimitive<i16> for u32 {
  #[inline(always)]
  fn cast_from(v: i16) -> Self {
    v as u32
  }
}

impl CastFromPrimitive<i32> for u32 {
  #[inline(always)]
  fn cast_from(v: i32) -> Self {
    v as u32
  }
}

impl CastFromPrimitive<u8> for u32 {
  #[inline(always)]
  fn cast_from(v: u8) -> Self {
    v as u32
  }
}

impl CastFromPrimitive<u16> for u32 {
  #[inline(always)]
  fn cast_from(v: u16) -> Self {
    v as u32
  }
}

impl CastFromPrimitive<u16> for i16 {
  #[inline(always)]
  fn cast_from(v: u16) -> Self {
    v as i16
  }
}

#[inline(always)]
pub const fn round_shift(value: i32, bit: usize) -> i32 {
  (value + (1 << bit >> 1)) >> bit
}

#[inline(always)]
pub fn msb(x: i32) -> i32 {
  debug_assert!(x > 0);
  31 ^ (x.leading_zeros() as i32)
}
