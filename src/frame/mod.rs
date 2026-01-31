// Copyright (c) 2018-2022, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::api::{Opaque, T35};
use crate::context::SB_SIZE;
use crate::mc::SUBPEL_FILTER_SIZE;
use crate::tiling::{Tile, TileMut, TileRect};
use crate::util::Pixel;
use num_derive::FromPrimitive;
use std::num::{NonZeroU8, NonZeroUsize};
use std::{fmt, iter};
use v_frame::chroma::ChromaSubsampling;

mod plane;
pub use plane::{AsRegion, Plane, PlaneConfig, PlaneGeometry};

const FRAME_MARGIN: usize = 16 + SUBPEL_FILTER_SIZE;
const LUMA_PADDING: usize = SB_SIZE + FRAME_MARGIN;

/// Extension trait to iterate over planes in a Frame
pub trait FrameIter<T: Pixel> {
  fn planes(&self) -> impl Iterator<Item = &Plane<T>>;
  fn planes_mut(&mut self) -> impl Iterator<Item = &mut Plane<T>>;
}

impl<T: Pixel> FrameIter<T> for Frame<T> {
  fn planes(&self) -> impl Iterator<Item = &Plane<T>> {
    iter::once(&self.y_plane)
      .chain(self.u_plane.as_ref())
      .chain(self.v_plane.as_ref())
  }

  fn planes_mut(&mut self) -> impl Iterator<Item = &mut Plane<T>> {
    iter::once(&mut self.y_plane)
      .chain(self.u_plane.as_mut())
      .chain(self.v_plane.as_mut())
  }
}

/// A 2D offset in a plane
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PlaneOffset {
  pub x: isize,
  pub y: isize,
}

/// Override the frame type decision
///
/// Only certain frame types can be selected.
#[derive(Debug, PartialEq, Eq, Clone, Copy, FromPrimitive, Default)]
#[repr(C)]
pub enum FrameTypeOverride {
  /// Do not force any decision.
  #[default]
  No,
  /// Force the frame to be a Keyframe.
  Key,
}

/// Optional per-frame encoder parameters
#[derive(Debug, Default)]
pub struct FrameParameters {
  /// Force emitted frame to be of the type selected
  pub frame_type_override: FrameTypeOverride,
  /// Output the provided data in the matching encoded Packet
  pub opaque: Option<Opaque>,
  /// List of t35 metadata associated with this frame
  pub t35_metadata: Box<[T35]>,
}

pub use v_frame::frame::Frame;

pub trait PlanePad {
  fn pad(&mut self, w: usize, h: usize);
  fn probe_padding(&self, w: usize, h: usize) -> bool;
}

impl<T: Pixel> PlanePad for Plane<T> {
  fn pad(&mut self, w: usize, h: usize) {
    let geo: PlaneGeometry = self.geometry();
    let pad_l = geo.pad_left;
    let pad_r = geo.pad_right;
    let pad_t = geo.pad_top;
    let pad_b = geo.pad_bottom;
    let stride = geo.stride.get();
    let data_origin = self.data_origin();
    let data = self.data_mut();
    let len = data.len();

    if w == 0 || h == 0 {
      return;
    }

    // 1. Pad Left/Right for visible rows
    for y in 0..h {
      let row_start = data_origin + y * stride;

      // Safety check: ensure row_start is within bounds
      if row_start >= len {
        break;
      }

      // Pad left
      let val_left = data[row_start];
      for x in 1..=pad_l {
        if row_start >= x {
          // Prevent underflow
          data[row_start - x] = val_left;
        }
      }

      // Pad right
      // Ensure we don't read or write past the end of the buffer
      if row_start + w > 0 {
        let val_right_idx = row_start + w - 1;
        if val_right_idx < len {
          let val_right = data[val_right_idx];
          for x in 0..pad_r {
            let idx = row_start + w + x;
            if idx < len {
              data[idx] = val_right;
            }
          }
        }
      }
    }

    // 2. Pad Top (copying the first padded row)
    // Ensure we don't underflow data_origin
    if data_origin >= pad_l {
      let first_row_start = data_origin - pad_l;
      let row_len = w + pad_l + pad_r;
      for y in 1..=pad_t {
        if y * stride > first_row_start {
          break;
        } // Prevent underflow
        let dest = first_row_start - y * stride;

        // Bounds-safe copy
        for x in 0..row_len {
          let src_idx = first_row_start + x;
          let dst_idx = dest + x;
          if src_idx < len && dst_idx < len {
            data[dst_idx] = data[src_idx];
          }
        }
      }
    }

    // 3. Pad Bottom (copying the last padded row)
    // Safety check for last_row calculation
    let last_row_idx = h.saturating_sub(1);
    let last_row_start_base = data_origin + last_row_idx * stride;

    if last_row_start_base >= pad_l {
      let last_row_start = last_row_start_base - pad_l;
      let row_len = w + pad_l + pad_r;

      for y in 1..=pad_b {
        let dest = last_row_start + y * stride;
        for x in 0..row_len {
          let src_idx = last_row_start + x;
          let dst_idx = dest + x;
          if src_idx < len && dst_idx < len {
            data[dst_idx] = data[src_idx];
          }
        }
      }
    }
  }

  fn probe_padding(&self, w: usize, h: usize) -> bool {
    let geo = self.geometry();

    let stride = geo.stride.get();
    let alloc_height = geo.alloc_height().get();
    let xorigin = geo.pad_left;
    let yorigin = geo.pad_top;

    let xdec = geo.subsampling_x.get().trailing_zeros() as usize;
    let ydec = geo.subsampling_y.get().trailing_zeros() as usize;

    let width = (w + xdec) >> xdec;
    let height = (h + ydec) >> ydec;

    let corner = (yorigin + height - 1) * stride + xorigin + width - 1;
    let corner_value = self.data()[corner];

    self.data()[(yorigin + height) * stride - 1] == corner_value
      && self.data()[(alloc_height - 1) * stride + xorigin + width - 1]
        == corner_value
      && self.data()[alloc_height * stride - 1] == corner_value
  }
}

pub(crate) trait FrameAlloc {
  fn new(
    width: usize, height: usize, chroma_sampling: ChromaSubsampling,
  ) -> Self;
}

impl<T: Pixel> FrameAlloc for Frame<T> {
  fn new(
    width: usize, height: usize, chroma_sampling: ChromaSubsampling,
  ) -> Self {
    use v_frame::frame::FrameBuilder;

    // Default to 8-bit for allocation if u8, else 10-bit for u16 (covers 10/12b)
    let bit_depth =
      NonZeroU8::new(if std::mem::size_of::<T>() == 1 { 8 } else { 10 })
        .unwrap();

    FrameBuilder::new(
      NonZeroUsize::new(width).expect("Frame width must be > 0"),
      NonZeroUsize::new(height).expect("Frame height must be > 0"),
      chroma_sampling,
      bit_depth,
    )
    .luma_padding_left(LUMA_PADDING)
    .luma_padding_right(LUMA_PADDING)
    .luma_padding_top(LUMA_PADDING)
    .luma_padding_bottom(LUMA_PADDING)
    .build()
    .expect("Failed to build frame")
  }
}

pub(crate) trait FramePad {
  fn pad(&mut self, w: usize, h: usize, planes: usize);
}

impl<T: Pixel> FramePad for Frame<T> {
  fn pad(&mut self, w: usize, h: usize, _planes: usize) {
    self.y_plane.pad(w, h);

    let subsampling = self.subsampling;
    if let Some(u) = &mut self.u_plane {
      if let Some((cw, ch)) = subsampling.chroma_dimensions(w, h) {
        u.pad(cw, ch);
      }
    }
    if let Some(v) = &mut self.v_plane {
      if let Some((cw, ch)) = subsampling.chroma_dimensions(w, h) {
        v.pad(cw, ch);
      }
    }
  }
}

/// Public Trait for new Tile of a frame
pub(crate) trait AsTile<T: Pixel> {
  fn as_tile(&self) -> Tile<'_, T>;
  fn as_tile_mut(&mut self) -> TileMut<'_, T>;
}

impl<T: Pixel> AsTile<T> for Frame<T> {
  #[inline(always)]
  fn as_tile(&self) -> Tile<'_, T> {
    let width = self.y_plane.width().get();
    let height = self.y_plane.height().get();
    Tile::new(self, TileRect { x: 0, y: 0, width, height })
  }
  #[inline(always)]
  fn as_tile_mut(&mut self) -> TileMut<'_, T> {
    let width = self.y_plane.width().get();
    let height = self.y_plane.height().get();
    TileMut::new(self, TileRect { x: 0, y: 0, width, height })
  }
}

#[derive(Clone, Copy)]
pub struct PlaneSlice<'a, T: Pixel> {
  pub plane: &'a Plane<T>,
  pub x: isize,
  pub y: isize,
}

// Manually implement Debug because Plane<T> does not implement it
impl<'a, T: Pixel> fmt::Debug for PlaneSlice<'a, T> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.debug_struct("PlaneSlice")
      .field("x", &self.x)
      .field("y", &self.y)
      .finish()
  }
}

#[allow(dead_code)]
impl<'a, T: Pixel> PlaneSlice<'a, T> {
  pub fn new(plane: &'a Plane<T>, x: isize, y: isize) -> Self {
    Self { plane, x, y }
  }

  #[allow(unused)]
  pub fn as_ptr(&self) -> *const T {
    let geo = self.plane.geometry();
    // Cast to isize for offsets
    let y_abs = self.y + geo.pad_top as isize;
    let x_abs = self.x + geo.pad_left as isize;
    let idx = y_abs as usize * geo.stride.get() + x_abs as usize;
    unsafe { self.plane.data().as_ptr().add(idx) }
  }

  pub fn clamp(&self) -> PlaneSlice<'a, T> {
    let geo = self.plane.geometry();
    PlaneSlice {
      plane: self.plane,
      x: self.x.clamp(-(geo.pad_left as isize), geo.width.get() as isize),
      y: self.y.clamp(-(geo.pad_top as isize), geo.height.get() as isize),
    }
  }

  pub fn subslice(&self, xo: usize, yo: usize) -> PlaneSlice<'a, T> {
    PlaneSlice {
      plane: self.plane,
      x: self.x + xo as isize,
      y: self.y + yo as isize,
    }
  }

  pub fn reslice(&self, xo: isize, yo: isize) -> PlaneSlice<'a, T> {
    PlaneSlice { plane: self.plane, x: self.x + xo, y: self.y + yo }
  }

  pub fn go_up(&self, i: usize) -> PlaneSlice<'a, T> {
    PlaneSlice { plane: self.plane, x: self.x, y: self.y - i as isize }
  }

  pub fn go_left(&self, i: usize) -> PlaneSlice<'a, T> {
    PlaneSlice { plane: self.plane, x: self.x - i as isize, y: self.y }
  }

  pub fn p(&self, add_x: usize, add_y: usize) -> T {
    let geo = self.plane.geometry();
    let new_y = (self.y + add_y as isize + geo.pad_top as isize) as usize;
    let new_x = (self.x + add_x as isize + geo.pad_left as isize) as usize;

    self.plane.data()[new_y * geo.stride.get() + new_x]
  }

  pub fn accessible(&self, add_x: usize, add_y: usize) -> bool {
    let geo = self.plane.geometry();
    let y = self.y + add_y as isize + geo.pad_top as isize;
    let x = self.x + add_x as isize + geo.pad_left as isize;
    y >= 0
      && (y as usize) < geo.alloc_height().get()
      && x >= 0
      && (x as usize) < geo.stride.get()
  }

  pub fn accessible_neg(&self, sub_x: usize, sub_y: usize) -> bool {
    let geo = self.plane.geometry();
    let y = self.y - sub_y as isize + geo.pad_top as isize;
    let x = self.x - sub_x as isize + geo.pad_left as isize;
    y >= 0 && x >= 0
  }

  pub fn row_cropped(&self, y: usize) -> &[T] {
    let geo = self.plane.geometry();
    let y_idx = (self.y + y as isize + geo.pad_top as isize) as usize;
    let x_idx = (self.x + geo.pad_left as isize) as usize;
    let start = y_idx * geo.stride.get() + x_idx;
    let width = (geo.width.get() as isize - self.x).max(0) as usize;

    return &self.plane.data()[start..start + width];
  }

  pub fn row(&self, y: usize) -> &[T] {
    let geo = self.plane.geometry();
    let y_idx = (self.y + y as isize + geo.pad_top as isize) as usize;
    let x_idx = (self.x + geo.pad_left as isize) as usize;
    let start = y_idx * geo.stride.get() + x_idx;
    let width = geo.stride.get().saturating_sub(x_idx);
    return &self.plane.data()[start..start + width];
  }
}
