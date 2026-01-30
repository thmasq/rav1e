// Copyright (c) 2001-2016, Alliance for Open Media. All rights reserved
// Copyright (c) 2017-2022, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use std::io::Read;

use crate::decoder::{DecodeError, Decoder, FrameBuilder, VideoDetails};
use rav1e::prelude::*;

trait PlaneMutExt {
  fn copy_from_raw_u8(
    &mut self, source: &[u8], source_stride: usize, bytes_per_sample: usize,
  );
}

impl<T: Pixel> PlaneMutExt for Plane<T> {
  fn copy_from_raw_u8(
    &mut self, source: &[u8], source_stride: usize, bytes_per_sample: usize,
  ) {
    let width = self.width().get();
    for (i, row) in self.rows_mut().enumerate() {
      let src_start = i * source_stride;
      let src_end = src_start + width * bytes_per_sample;

      if src_end > source.len() {
        break;
      }

      let src = &source[src_start..src_end];

      // SAFETY: We cast the destination slice (T) to bytes (u8) and copy.
      // This assumes T (u8 or u16) matches the input byte layout (e.g. Little Endian).
      unsafe {
        let dst_ptr = row.as_mut_ptr() as *mut u8;
        std::ptr::copy_nonoverlapping(src.as_ptr(), dst_ptr, src.len());
      }
    }
  }
}

impl<T: Pixel> PlaneMutExt for Option<Plane<T>> {
  fn copy_from_raw_u8(
    &mut self, source: &[u8], source_stride: usize, bytes_per_sample: usize,
  ) {
    if let Some(plane) = self {
      plane.copy_from_raw_u8(source, source_stride, bytes_per_sample);
    }
  }
}

impl Decoder for y4m::Decoder<Box<dyn Read + Send>> {
  fn get_video_details(&self) -> VideoDetails {
    let width = self.get_width();
    let height = self.get_height();
    let aspect_ratio = self.get_pixel_aspect();
    let color_space = self.get_colorspace();
    let bit_depth = color_space.get_bit_depth();
    let (chroma_sampling, chroma_sample_position) =
      map_y4m_color_space(color_space);
    let framerate = self.get_framerate();
    let time_base = Rational::new(framerate.den as u64, framerate.num as u64);

    VideoDetails {
      width,
      height,
      sample_aspect_ratio: if aspect_ratio.num == 0 && aspect_ratio.den == 0 {
        Rational::new(1, 1)
      } else {
        Rational::new(aspect_ratio.num as u64, aspect_ratio.den as u64)
      },
      bit_depth,
      chroma_sampling,
      chroma_sample_position,
      time_base,
    }
  }

  fn read_frame<T: Pixel, F: FrameBuilder<T>>(
    &mut self, ctx: &F, cfg: &VideoDetails,
  ) -> Result<Frame<T>, DecodeError> {
    let bytes = self.get_bytes_per_sample();
    self
      .read_frame()
      .map(|frame| {
        let mut f = ctx.new_frame();

        let (chroma_width, _) = cfg
          .chroma_sampling
          .chroma_dimensions(cfg.width, cfg.height)
          .unwrap();

        f.y_plane.copy_from_raw_u8(
          frame.get_y_plane(),
          cfg.width * bytes,
          bytes,
        );
        if cfg.chroma_sampling != ChromaSubsampling::Monochrome {
          f.u_plane.copy_from_raw_u8(
            frame.get_u_plane(),
            chroma_width * bytes,
            bytes,
          );
          f.v_plane.copy_from_raw_u8(
            frame.get_v_plane(),
            chroma_width * bytes,
            bytes,
          );
        }
        f
      })
      .map_err(Into::into)
  }
}

impl From<y4m::Error> for DecodeError {
  fn from(e: y4m::Error) -> DecodeError {
    match e {
      y4m::Error::EOF => DecodeError::EOF,
      y4m::Error::BadInput => DecodeError::BadInput,
      y4m::Error::UnknownColorspace => DecodeError::UnknownColorspace,
      y4m::Error::ParseError(_) => DecodeError::ParseError,
      y4m::Error::IoError(_) => DecodeError::IoError,
      // Note that this error code has nothing to do with the system running out of memory,
      // it means the y4m decoder has exceeded its memory allocation limit.
      y4m::Error::OutOfMemory => DecodeError::MemoryLimitExceeded,
    }
  }
}

pub const fn map_y4m_color_space(
  color_space: y4m::Colorspace,
) -> (ChromaSubsampling, ChromaSamplePosition) {
  use crate::ChromaSamplePosition::*;
  use crate::ChromaSubsampling::*;
  use y4m::Colorspace::*;
  match color_space {
    Cmono | Cmono12 => (Monochrome, Unknown),
    C420jpeg | C420paldv => (Yuv420, Unknown),
    C420mpeg2 => (Yuv420, Vertical),
    C420 | C420p10 | C420p12 => (Yuv420, Colocated),
    C422 | C422p10 | C422p12 => (Yuv422, Colocated),
    C444 | C444p10 | C444p12 => (Yuv444, Colocated),
    _ => unimplemented!(),
  }
}
