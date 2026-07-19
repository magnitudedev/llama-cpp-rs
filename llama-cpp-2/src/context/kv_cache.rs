//! utilities for working with the kv cache

use crate::context::LlamaContext;
use std::ffi::c_int;
use std::num::{NonZeroU8, TryFromIntError};
use std::ptr::NonNull;

/// A context-scoped immutable KV page retained by llama.cpp's unified cache.
///
/// The native identifier is intentionally private and carries its originating context pointer so
/// safe callers cannot accidentally attach a target-context page to a draft context.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct LlamaKvPageId {
    context: NonNull<llama_cpp_sys_2::llama_context>,
    id: u64,
}

/// Current logical cell usage for the native page cache.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LlamaKvPageStats {
    /// Number of retained page records.
    pub pages: u32,
    /// Cells owned by requests, page pins, or both.
    pub used_cells: u32,
    /// Cells immediately available to native decode.
    pub free_cells: u32,
    /// Unique cells retained by at least one page pin.
    pub pinned_cells: u32,
}

/// Failure from a safe radix-page operation.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum KvPageError {
    /// The active memory is not a unified, independently pageable attention cache.
    #[error("the active model memory does not support immutable KV pages")]
    Unsupported,
    /// The requested page range is empty, inverted, or exceeds llama.cpp's position type.
    #[error("invalid KV page position range")]
    InvalidRange,
    /// A page belongs to a different native context.
    #[error("KV page belongs to a different context")]
    WrongContext,
    /// Native metadata did not contain one ordinary token cell for every requested position.
    #[error("native memory rejected the KV page operation")]
    NativeRejected,
    /// Native tensor serialization or validation of an imported page failed.
    #[error("native memory rejected the KV page blob")]
    InvalidBlob,
}

/// Errors that can occur when attempting to prepare values for the kv cache
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
#[allow(clippy::module_name_repetitions)]
pub enum KvCacheConversionError {
    /// Sequence id conversion to i32 failed
    #[error("Provided sequence id is too large for a i32")]
    SeqIdTooLarge(#[source] TryFromIntError),
    /// Position 0 conversion to i32 failed
    #[error("Provided start position is too large for a i32")]
    P0TooLarge(#[source] TryFromIntError),
    /// Position 1 conversion to i32 failed
    #[error("Provided end position is too large for a i32")]
    P1TooLarge(#[source] TryFromIntError),
}

impl LlamaContext<'_> {
    fn page_memory(&self) -> llama_cpp_sys_2::llama_memory_t {
        unsafe { llama_cpp_sys_2::llama_get_memory(self.context.as_ptr()) }
    }

    /// Whether this context supports immutable, composable KV pages.
    ///
    /// The first implementation deliberately requires unified ordinary-attention memory. Hybrid,
    /// recurrent, multidimensional-position, and per-sequence-stream layouts return `false`.
    #[must_use]
    pub fn kv_pages_supported(&self) -> bool {
        unsafe { llama_cpp_sys_2::llama_memory_page_supports(self.page_memory()) }
    }

    /// Retain the exact sequence positions `[start, end)` as an immutable native KV page.
    pub fn pin_kv_page(
        &mut self,
        sequence_id: i32,
        start: u32,
        end: u32,
    ) -> Result<LlamaKvPageId, KvPageError> {
        if !self.kv_pages_supported() {
            return Err(KvPageError::Unsupported);
        }
        let start = i32::try_from(start).map_err(|_| KvPageError::InvalidRange)?;
        let end = i32::try_from(end).map_err(|_| KvPageError::InvalidRange)?;
        if end <= start {
            return Err(KvPageError::InvalidRange);
        }
        let id = unsafe {
            llama_cpp_sys_2::llama_memory_page_pin(self.page_memory(), sequence_id, start, end)
        };
        if id == 0 {
            return Err(KvPageError::NativeRejected);
        }
        Ok(LlamaKvPageId {
            context: self.context,
            id,
        })
    }

    /// Attach a retained page to a cleared destination sequence without copying its K/V data.
    pub fn attach_kv_page(
        &mut self,
        page: LlamaKvPageId,
        destination_sequence_id: i32,
    ) -> Result<(), KvPageError> {
        if page.context != self.context {
            return Err(KvPageError::WrongContext);
        }
        let attached = unsafe {
            llama_cpp_sys_2::llama_memory_page_attach(
                self.page_memory(),
                page.id,
                destination_sequence_id,
            )
        };
        attached.then_some(()).ok_or(KvPageError::NativeRejected)
    }

    /// Release one cache pin. Request sequences already attached to the page remain valid.
    pub fn release_kv_page(&mut self, page: LlamaKvPageId) -> Result<(), KvPageError> {
        if page.context != self.context {
            return Err(KvPageError::WrongContext);
        }
        let released =
            unsafe { llama_cpp_sys_2::llama_memory_page_release(self.page_memory(), page.id) };
        released.then_some(()).ok_or(KvPageError::NativeRejected)
    }

    /// Copy one retained page into a self-describing opaque host blob.
    pub fn export_kv_page(&self, page: LlamaKvPageId) -> Result<Vec<u8>, KvPageError> {
        if page.context != self.context {
            return Err(KvPageError::WrongContext);
        }
        let memory = self.page_memory();
        let size = unsafe {
            llama_cpp_sys_2::llama_memory_page_export(memory, page.id, std::ptr::null_mut(), 0)
        };
        if size == 0 {
            return Err(KvPageError::InvalidBlob);
        }
        let mut blob = vec![0_u8; size];
        let written = unsafe {
            llama_cpp_sys_2::llama_memory_page_export(
                memory,
                page.id,
                blob.as_mut_ptr().cast(),
                blob.len(),
            )
        };
        if written != size {
            return Err(KvPageError::InvalidBlob);
        }
        Ok(blob)
    }

    /// Restore an exported page into free native cells and retain it with a new context-scoped ID.
    pub fn import_kv_page(&mut self, blob: &[u8]) -> Result<LlamaKvPageId, KvPageError> {
        if !self.kv_pages_supported() || blob.is_empty() {
            return Err(KvPageError::InvalidBlob);
        }
        let id = unsafe {
            llama_cpp_sys_2::llama_memory_page_import(
                self.page_memory(),
                blob.as_ptr().cast(),
                blob.len(),
            )
        };
        if id == 0 {
            return Err(KvPageError::InvalidBlob);
        }
        Ok(LlamaKvPageId {
            context: self.context,
            id,
        })
    }

    /// Number of token cells retained by one page, or zero if it is stale.
    #[must_use]
    pub fn kv_page_token_count(&self, page: LlamaKvPageId) -> u32 {
        if page.context != self.context {
            return 0;
        }
        unsafe { llama_cpp_sys_2::llama_memory_page_token_count(self.page_memory(), page.id) }
    }

    /// Return exact native page/cell accounting.
    #[must_use]
    pub fn kv_page_stats(&self) -> LlamaKvPageStats {
        let memory = self.page_memory();
        LlamaKvPageStats {
            pages: unsafe { llama_cpp_sys_2::llama_memory_page_count(memory) },
            used_cells: unsafe { llama_cpp_sys_2::llama_memory_page_used_cells(memory) },
            free_cells: unsafe { llama_cpp_sys_2::llama_memory_page_free_cells(memory) },
            pinned_cells: unsafe { llama_cpp_sys_2::llama_memory_page_pinned_cells(memory) },
        }
    }

    /// Whether this model's active memory implementation supports sequence position shifting.
    #[must_use]
    pub fn memory_can_shift(&self) -> bool {
        let mem = unsafe { llama_cpp_sys_2::llama_get_memory(self.context.as_ptr()) };
        unsafe { llama_cpp_sys_2::llama_memory_can_shift(mem) }
    }

    /// Return the smallest position currently stored for a sequence, or a negative value when the
    /// sequence has no resident state.
    #[must_use]
    pub fn memory_seq_pos_min(&self, seq_id: i32) -> i32 {
        let mem = unsafe { llama_cpp_sys_2::llama_get_memory(self.context.as_ptr()) };
        unsafe { llama_cpp_sys_2::llama_memory_seq_pos_min(mem, seq_id) }
    }

    /// Copy the cache from one sequence to another.
    ///
    /// # Parameters
    ///
    /// * `src` - The sequence id to copy the cache from.
    /// * `dest` - The sequence id to copy the cache to.
    /// * `size` - The size of the cache to copy.
    pub fn copy_cache(&mut self, src: i32, dest: i32, size: i32) {
        let mem = unsafe { llama_cpp_sys_2::llama_get_memory(self.context.as_ptr()) };
        unsafe { llama_cpp_sys_2::llama_memory_seq_cp(mem, src, dest, 0, size) }
    }

    /// Copy the cache from one sequence to another.
    ///
    /// # Returns
    /// A `Result` indicating whether the operation was successful.
    ///
    /// # Parameters
    /// * `src` - The sequence id to copy the cache from.
    /// * `dest` - The sequence id to copy the cache to.
    /// * `p0` - The start position of the cache to clear. If `None`, the entire cache is copied up to `p1`.
    /// * `p1` - The end position of the cache to clear. If `None`, the entire cache is copied starting from `p0`.
    ///
    /// # Errors
    /// If either position exceeds [`i32::MAX`].
    pub fn copy_kv_cache_seq(
        &mut self,
        src: i32,
        dest: i32,
        p0: Option<u32>,
        p1: Option<u32>,
    ) -> Result<(), KvCacheConversionError> {
        let p0 = p0
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheConversionError::P0TooLarge)?;
        let p1 = p1
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheConversionError::P1TooLarge)?;
        let mem = unsafe { llama_cpp_sys_2::llama_get_memory(self.context.as_ptr()) };
        unsafe { llama_cpp_sys_2::llama_memory_seq_cp(mem, src, dest, p0, p1) };
        Ok(())
    }

    /// Clear the kv cache for the given sequence within the specified range `[p0, p1)`
    /// Returns `false` only when partial sequence removals fail. Full sequence removals always succeed.
    ///
    /// # Returns
    /// A `Result` indicating whether the operation was successful. If the sequence id or
    /// either position exceeds the maximum i32 value, no removal is attempted and an `Err` is returned.
    ///
    /// # Parameters
    /// * `src` - The sequence id to clear the cache for. If `None`, matches all sequences
    /// * `p0` - The start position of the cache to clear. If `None`, the entire cache is cleared up to `p1`.
    /// * `p1` - The end position of the cache to clear. If `None`, the entire cache is cleared from `p0`.
    ///
    /// # Errors
    /// If the sequence id or either position exceeds [`i32::MAX`].
    pub fn clear_kv_cache_seq(
        &mut self,
        src: Option<u32>,
        p0: Option<u32>,
        p1: Option<u32>,
    ) -> Result<bool, KvCacheConversionError> {
        let src = src
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheConversionError::SeqIdTooLarge)?;
        let p0 = p0
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheConversionError::P0TooLarge)?;
        let p1 = p1
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheConversionError::P1TooLarge)?;
        let mem = unsafe { llama_cpp_sys_2::llama_get_memory(self.context.as_ptr()) };
        Ok(unsafe { llama_cpp_sys_2::llama_memory_seq_rm(mem, src, p0, p1) })
    }

    /// Clear all context memory metadata and optionally zero its backing data buffers.
    ///
    /// Pass `false` for llama.cpp's fast logical reset used between `llama-bench` repetitions.
    /// Pass `true` when cached data must also be overwritten.
    pub fn clear_memory(&mut self, data: bool) {
        let mem = unsafe { llama_cpp_sys_2::llama_get_memory(self.context.as_ptr()) };
        unsafe { llama_cpp_sys_2::llama_memory_clear(mem, data) }
    }

    /// Clear the KV cache, including its backing data buffers.
    ///
    /// This compatibility alias preserves the method's historical zeroing behavior. Use
    /// [`Self::clear_memory`] when the distinction is important.
    pub fn clear_kv_cache(&mut self) {
        self.clear_memory(true);
    }

    /// Removes all tokens that do not belong to the specified sequence
    ///
    /// # Parameters
    ///
    /// * `seq_id` - The sequence id to keep
    pub fn llama_kv_cache_seq_keep(&mut self, seq_id: i32) {
        let mem = unsafe { llama_cpp_sys_2::llama_get_memory(self.context.as_ptr()) };
        unsafe { llama_cpp_sys_2::llama_memory_seq_keep(mem, seq_id) }
    }

    #[allow(clippy::doc_markdown)]
    /// Adds relative position "delta" to all tokens that belong to the specified sequence and have positions in `[p0, p1)`
    /// If the KV cache is RoPEd, the KV data is updated accordingly:
    ///   - lazily on next [`LlamaContext::decode`]
    ///   - explicitly with [`Self::kv_cache_update`]
    ///
    /// # Returns
    /// A `Result` indicating whether the operation was successful.
    ///
    /// # Parameters
    ///
    /// * `seq_id` - The sequence id to update
    /// * `p0` - The start position of the cache to update. If `None`, the entire cache is updated up to `p1`.
    /// * `p1` - The end position of the cache to update. If `None`, the entire cache is updated starting from `p0`.
    /// * `delta` - The relative position to add to the tokens
    ///
    /// # Errors
    /// If either position exceeds [`i32::MAX`].
    pub fn kv_cache_seq_add(
        &mut self,
        seq_id: i32,
        p0: Option<u32>,
        p1: Option<u32>,
        delta: i32,
    ) -> Result<(), KvCacheConversionError> {
        let p0 = p0
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheConversionError::P0TooLarge)?;
        let p1 = p1
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheConversionError::P1TooLarge)?;
        let mem = unsafe { llama_cpp_sys_2::llama_get_memory(self.context.as_ptr()) };
        unsafe { llama_cpp_sys_2::llama_memory_seq_add(mem, seq_id, p0, p1, delta) };
        Ok(())
    }

    /// Integer division of the positions by factor of `d > 1`
    /// If the KV cache is `RoPEd`, the KV data is updated accordingly:
    ///   - lazily on next [`LlamaContext::decode`]
    ///   - explicitly with [`Self::kv_cache_update`]
    ///
    /// # Returns
    /// A `Result` indicating whether the operation was successful.
    ///
    /// # Parameters
    ///
    /// * `seq_id` - The sequence id to update
    /// * `p0` - The start position of the cache to update. If `None`, the entire cache is updated up to `p1`.
    /// * `p1` - The end position of the cache to update. If `None`, the entire cache is updated starting from `p0`.
    /// * `d` - The factor to divide the positions by
    ///
    /// # Errors
    /// If either position exceeds [`i32::MAX`].
    pub fn kv_cache_seq_div(
        &mut self,
        seq_id: i32,
        p0: Option<u32>,
        p1: Option<u32>,
        d: NonZeroU8,
    ) -> Result<(), KvCacheConversionError> {
        let p0 = p0
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheConversionError::P0TooLarge)?;
        let p1 = p1
            .map_or(Ok(-1), i32::try_from)
            .map_err(KvCacheConversionError::P1TooLarge)?;
        let d = c_int::from(d.get());
        let mem = unsafe { llama_cpp_sys_2::llama_get_memory(self.context.as_ptr()) };
        unsafe { llama_cpp_sys_2::llama_memory_seq_div(mem, seq_id, p0, p1, d) }
        Ok(())
    }

    /// Returns the largest position present in the KV cache for the specified sequence
    ///
    /// # Parameters
    ///
    /// * `seq_id` - The sequence id to get the max position for
    #[must_use]
    pub fn kv_cache_seq_pos_max(&self, seq_id: i32) -> i32 {
        let mem = unsafe { llama_cpp_sys_2::llama_get_memory(self.context.as_ptr()) };
        unsafe { llama_cpp_sys_2::llama_memory_seq_pos_max(mem, seq_id) }
    }
}
