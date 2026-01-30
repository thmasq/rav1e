// Copyright (c) 2018-2022, The rav1e contributors. All rights reserved
//
// This source code is subject to the terms of the BSD 2 Clause License and
// the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
// was not distributed with this source code in the LICENSE file, you can
// obtain it at www.aomedia.org/license/software. If the Alliance for Open
// Media Patent License 1.0 was not distributed with this source code in the
// PATENTS file, you can obtain it at www.aomedia.org/license/patent.
#![deny(missing_docs)]

use std::fmt;
use std::io;
use std::sync::Arc;

use bitstream_io::{BigEndian, BitWrite, BitWriter};
use v_frame::chroma::ChromaSubsampling;

use crate::api::config::*;
use crate::api::internal::*;
use crate::api::util::*;
use crate::encoder::*;
use crate::frame::*;
use crate::util;
use crate::util::Pixel;

/// The encoder context.
///
/// Contains the encoding state.
pub struct Context<T: Pixel> {
  pub(crate) inner: ContextInner<T>,
  pub(crate) config: EncoderConfig,
  pub(crate) pool: Option<Arc<rayon::ThreadPool>>,
  pub(crate) is_flushing: bool,
}

impl<T: Pixel> Context<T>
where
  u32: crate::util::math::CastFromPrimitive<T::Coeff>,
  i32: crate::util::math::CastFromPrimitive<T::Coeff>,
  <T as util::pixel::Pixel>::Coeff: num_traits::AsPrimitive<u8>,
  i32: crate::util::math::CastFromPrimitive<T>,
  u32: crate::util::math::CastFromPrimitive<T>,
  i16: crate::util::math::CastFromPrimitive<T>,
  i16: crate::util::math::CastFromPrimitive<T::Coeff>,
{
  /// Allocates and returns a new frame.
  #[inline]
  pub fn new_frame(&self) -> Frame<T> {
    Frame::new(
      self.config.width,
      self.config.height,
      self.config.chroma_sampling,
    )
  }

  /// Sends the frame for encoding.
  #[inline]
  pub fn send_frame<F>(&mut self, frame: F) -> Result<(), EncoderStatus>
  where
    F: IntoFrame<T>,
  {
    let (frame, params) = frame.into();

    if frame.is_none() {
      if self.is_flushing {
        return Ok(());
      }
      self.inner.limit = Some(self.inner.frame_count);
      self.is_flushing = true;
    } else if self.is_flushing
      || (self.inner.config.still_picture && self.inner.frame_count > 0)
    {
      return Err(EncoderStatus::EnoughData);
    // The rate control can process at most i32::MAX frames
    } else if self.inner.frame_count == i32::MAX as u64 - 1 {
      self.inner.limit = Some(self.inner.frame_count);
      self.is_flushing = true;
    }

    let inner = &mut self.inner;
    let run = move || inner.send_frame(frame, params);

    match &self.pool {
      Some(pool) => pool.install(run),
      None => run(),
    }
  }

  /// Returns the first-pass data of a two-pass encode for the frame that was
  /// just encoded.
  #[inline]
  pub fn twopass_out(&mut self) -> Option<&[u8]> {
    self.inner.rc_state.twopass_out(self.inner.done_processing())
  }

  /// Returns the number of bytes of the stats file needed before the next
  /// frame of the second pass in a two-pass encode can be encoded.
  #[inline]
  pub fn twopass_bytes_needed(&mut self) -> usize {
    self.inner.rc_state.twopass_in(None).unwrap_or(0)
  }

  /// Provides the stats data produced in the first pass of a two-pass encode
  /// to the second pass.
  #[inline]
  pub fn twopass_in(&mut self, buf: &[u8]) -> Result<usize, EncoderStatus> {
    self.inner.rc_state.twopass_in(Some(buf)).or(Err(EncoderStatus::Failure))
  }

  /// Encodes the next frame and returns the encoded data.
  #[inline]
  pub fn receive_packet(&mut self) -> Result<Packet<T>, EncoderStatus> {
    let inner = &mut self.inner;
    let mut run = move || inner.receive_packet();

    match &self.pool {
      Some(pool) => pool.install(run),
      None => run(),
    }
  }

  /// Flushes the encoder.
  #[inline]
  pub fn flush(&mut self) {
    self.send_frame(None).unwrap();
  }

  /// Produces a sequence header matching the current encoding context.
  #[inline]
  pub fn container_sequence_header(&self) -> Vec<u8> {
    fn sequence_header_inner(seq: &Sequence) -> io::Result<Vec<u8>> {
      let mut buf = Vec::new();

      {
        let mut bw = BitWriter::endian(&mut buf, BigEndian);
        bw.write_bit(true)?; // marker
        bw.write::<7, u8>(1)?; // version
        bw.write::<3, u8>(seq.profile)?;
        bw.write::<5, u8>(31)?; // level
        bw.write_bit(false)?; // tier
        bw.write_bit(seq.bit_depth > 8)?; // high_bitdepth
        bw.write_bit(seq.bit_depth == 12)?; // twelve_bit
        bw.write_bit(seq.chroma_sampling == ChromaSubsampling::Monochrome)?; // monochrome
        bw.write_bit(seq.chroma_sampling != ChromaSubsampling::Yuv444)?; // chroma_subsampling_x
        bw.write_bit(seq.chroma_sampling == ChromaSubsampling::Yuv420)?; // chroma_subsampling_y
        bw.write::<2, u8>(0)?; // chroma_sample_position
        bw.write::<3, u8>(0)?; // reserved
        bw.write_bit(false)?; // initial_presentation_delay_present

        bw.write::<4, u8>(0)?; // reserved
      }

      Ok(buf)
    }

    let seq = Sequence::new(&self.config);

    sequence_header_inner(&seq).unwrap()
  }
}

/// Rate Control Data
pub enum RcData {
  /// A Rate Control Summary Packet
  Summary(Box<[u8]>),
  /// A Rate Control Frame-specific Packet
  Frame(Box<[u8]>),
}

impl<T: Pixel> Context<T>
where
  u32: crate::util::math::CastFromPrimitive<T::Coeff>,
  i32: crate::util::math::CastFromPrimitive<T::Coeff>,
  <T as util::pixel::Pixel>::Coeff: num_traits::AsPrimitive<u8>,
  i32: crate::util::math::CastFromPrimitive<T>,
  u32: crate::util::math::CastFromPrimitive<T>,
  i16: crate::util::math::CastFromPrimitive<T>,
  i16: crate::util::math::CastFromPrimitive<T::Coeff>,
{
  /// Return the Rate Control Summary Packet size
  pub fn rc_summary_size(&self) -> usize {
    crate::rate::TWOPASS_HEADER_SZ
  }

  /// Return the first pass data
  pub fn rc_receive_pass_data(&mut self) -> Option<RcData> {
    if self.inner.done_processing() && self.inner.rc_state.pass1_data_retrieved
    {
      let data = self.inner.rc_state.emit_summary();
      Some(RcData::Summary(data.to_vec().into_boxed_slice()))
    } else if self.inner.rc_state.pass1_data_retrieved {
      None
    } else if let Some(data) = self.inner.rc_state.emit_frame_data() {
      Some(RcData::Frame(data.to_vec().into_boxed_slice()))
    } else {
      unreachable!(
        "The encoder received more frames than its internal limit allows"
      )
    }
  }

  /// Lower bound number of pass data packets required to progress the
  /// encoding process.
  pub fn rc_second_pass_data_required(&self) -> usize {
    if self.inner.done_processing() {
      0
    } else {
      self.inner.rc_state.twopass_in_frames_needed() as usize
    }
  }

  /// Feed the first pass Rate Control data to the encoder,
  /// Frame-specific Packets only.
  pub fn rc_send_pass_data(
    &mut self, data: &[u8],
  ) -> Result<(), EncoderStatus> {
    self
      .inner
      .rc_state
      .parse_frame_data_packet(data)
      .map_err(|_| EncoderStatus::Failure)
  }
}

impl<T: Pixel> fmt::Debug for Context<T> {
  fn fmt(
    &self, f: &mut fmt::Formatter<'_>,
  ) -> std::result::Result<(), fmt::Error> {
    write!(
      f,
      "{{ \
        config: {:?}, \
        is_flushing: {}, \
      }}",
      self.config, self.is_flushing,
    )
  }
}
