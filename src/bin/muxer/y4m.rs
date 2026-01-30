// Copyright (c) 2017-2022, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.

use crate::decoder::VideoDetails;
use rav1e::prelude::*;
use std::io::Write;

#[profiling::function]
pub fn write_y4m_frame<T: Pixel>(
  y4m_enc: &mut y4m::Encoder<Box<dyn Write + Send>>, rec: &Frame<T>,
  y4m_details: VideoDetails,
) {
  let planes = if y4m_details.chroma_sampling == ChromaSubsampling::Monochrome
  {
    1
  } else {
    3
  };
  let bytes_per_sample = if y4m_details.bit_depth > 8 { 2 } else { 1 };
  let (chroma_width, chroma_height) = y4m_details
    .chroma_sampling
    .chroma_dimensions(y4m_details.width, y4m_details.height)
    .unwrap();
  let pitch_y = y4m_details.width * bytes_per_sample;
  let pitch_uv = chroma_width * bytes_per_sample;

  let (mut rec_y, mut rec_u, mut rec_v) = (
    vec![128u8; pitch_y * y4m_details.height],
    vec![128u8; pitch_uv * chroma_height],
    vec![128u8; pitch_uv * chroma_height],
  );

  let copy_plane =
    |plane: &Plane<T>, dest: &mut [u8], w: usize, h: usize, pitch: usize| {
      let stride = plane.geometry().stride.get();

      let origin =
        plane.geometry().pad_top * stride + plane.geometry().pad_left;
      let data = plane.data();

      for y in 0..h {
        let src_start = origin + y * stride;
        let src_end = src_start + w;

        if src_end > data.len() {
          break;
        }

        let src_row = &data[src_start..src_end];
        let dest_start = y * pitch;
        let dest_row =
          &mut dest[dest_start..dest_start + (w * bytes_per_sample)];

        if y4m_details.bit_depth > 8 {
          // 10/12-bit: Copy raw bytes (assuming generic T layout matches system endianness for u16)
          // SAFETY: Casting T slice to u8 slice for copy.
          unsafe {
            let src_ptr = src_row.as_ptr() as *const u8;
            let dest_ptr = dest_row.as_mut_ptr();
            std::ptr::copy_nonoverlapping(src_ptr, dest_ptr, w * 2);
          }
        } else {
          for (out, &val) in dest_row.iter_mut().zip(src_row.iter()) {
            *out = val.to_u16() as u8;
          }
        }
      }
    };

  // Process Y Plane
  copy_plane(
    &rec.y_plane,
    &mut rec_y,
    y4m_details.width,
    y4m_details.height,
    pitch_y,
  );

  // Process U and V Planes if they exist
  if planes > 1 {
    if let Some(u_plane) = &rec.u_plane {
      copy_plane(u_plane, &mut rec_u, chroma_width, chroma_height, pitch_uv);
    }
    if let Some(v_plane) = &rec.v_plane {
      copy_plane(v_plane, &mut rec_v, chroma_width, chroma_height, pitch_uv);
    }
  }

  let rec_frame = y4m::Frame::new([&rec_y, &rec_u, &rec_v], None);
  y4m_enc.write_frame(&rec_frame).unwrap();
}
