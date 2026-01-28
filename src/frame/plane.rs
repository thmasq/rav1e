// Copyright (c) 2017-2022, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::tiling::*;
use crate::util::*;

pub use v_frame::plane::*;

pub trait AsRegion<T: Pixel> {
  fn as_region(&self) -> PlaneRegion<'_, T>;
  fn as_region_mut(&mut self) -> PlaneRegionMut<'_, T>;
  fn region_mut(&mut self, area: Area) -> PlaneRegionMut<'_, T>;
  fn region(&self, area: Area) -> PlaneRegion<'_, T>;
}

impl<T: Pixel> AsRegion<T> for Plane<T> {
  #[inline(always)]
  fn region(&self, area: Area) -> PlaneRegion<'_, T> {
    let geo = self.geometry();
    let rect = area.to_rect(
      geo.subsampling_x.get().trailing_zeros() as usize,
      geo.subsampling_y.get().trailing_zeros() as usize,
      geo.stride.get() - geo.pad_left,
      geo.alloc_height().get() - geo.pad_top,
    );
    PlaneRegion::new(self, rect)
  }

  #[inline(always)]
  fn region_mut(&mut self, area: Area) -> PlaneRegionMut<'_, T> {
    let geo = self.geometry();
    let rect = area.to_rect(
      geo.subsampling_x.get().trailing_zeros() as usize,
      geo.subsampling_y.get().trailing_zeros() as usize,
      geo.stride.get() - geo.pad_left,
      geo.alloc_height().get() - geo.pad_top,
    );
    PlaneRegionMut::new(self, rect)
  }

  #[inline(always)]
  fn as_region(&self) -> PlaneRegion<'_, T> {
    PlaneRegion::new_from_plane(self)
  }

  #[inline(always)]
  fn as_region_mut(&mut self) -> PlaneRegionMut<'_, T> {
    PlaneRegionMut::new_from_plane(self)
  }
}
