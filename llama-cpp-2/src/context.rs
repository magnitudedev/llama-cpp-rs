//! Safe wrapper around `llama_context`.

#[cfg(feature = "common")]
use std::ffi::CStr;
use std::fmt::{Debug, Formatter};
use std::num::NonZeroI32;
use std::ops::{Deref, DerefMut};
use std::os::raw::c_void;
use std::ptr::NonNull;
use std::slice;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::llama_backend::LlamaThreadPool;
use crate::llama_batch::LlamaBatch;
use crate::model::{LlamaLoraAdapter, LlamaModel};
use crate::timing::LlamaTimings;
use crate::token::data::LlamaTokenData;
use crate::token::data_array::LlamaTokenDataArray;
use crate::token::LlamaToken;
use crate::{
    DecodeError, EmbeddingsError, EncodeError, LlamaLoraAdapterRemoveError,
    LlamaLoraAdapterSetError,
};

pub mod kv_cache;
pub mod params;
pub mod session;

/// Safe wrapper around `llama_context`.
#[allow(clippy::module_name_repetitions)]
pub struct LlamaContext<'a> {
    pub(crate) context: NonNull<llama_cpp_sys_2::llama_context>,
    /// a reference to the contexts model.
    pub model: &'a LlamaModel,
    initialized_logits: Vec<i32>,
    embeddings_enabled: bool,
    abort_callback: Option<Arc<AtomicBool>>,
}

/// Physical memory location reported by llama.cpp for a resident allocation.
#[cfg(feature = "common")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LlamaMemoryLocation {
    /// Host memory.
    Host,
    /// Memory owned by one registered backend device.
    Device {
        /// Normalized backend name.
        backend: String,
        /// Backend-provided physical device identity, when available.
        physical_id: Option<String>,
        /// Index in llama.cpp's registered backend-device list.
        native_index: usize,
    },
}

/// Resident bytes attributed by llama.cpp to one physical memory location.
#[cfg(feature = "common")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LlamaMemoryBreakdown {
    /// Physical allocation location.
    pub location: LlamaMemoryLocation,
    /// Model allocation bytes.
    pub model_bytes: u64,
    /// Context and KV allocation bytes.
    pub context_bytes: u64,
    /// Compute workspace allocation bytes.
    pub compute_bytes: u64,
}

/// Failure to obtain a typed resident-memory report from llama.cpp.
#[cfg(feature = "common")]
#[derive(Debug, thiserror::Error)]
pub enum LlamaMemoryBreakdownError {
    /// The native bridge rejected the capture operation.
    #[error("llama.cpp resident-memory bridge failed with status {status}: {message}")]
    Native {
        /// Raw `llama_rs_status` value.
        status: i32,
        /// Native diagnostic.
        message: String,
    },
    /// A successful native call returned no report.
    #[error("llama.cpp returned a null resident-memory report")]
    NullReport,
    /// Rust could not reserve space for the native report entries.
    #[error("could not reserve {requested} resident-memory entries")]
    RustAllocation {
        /// Native-reported entry count.
        requested: usize,
    },
    /// A report entry could not be read.
    #[error("llama.cpp returned an invalid resident-memory entry at index {0}")]
    InvalidEntry(usize),
    /// The report contained a location kind unknown to this binding.
    #[error("llama.cpp returned an unknown resident-memory location {0}")]
    UnknownLocation(i64),
}

/// A thread-safe cancellation handle for an installed llama.cpp abort callback.
///
/// The callback itself is owned by the associated [`LlamaContext`]. Dropping this handle does not
/// remove the callback; it only gives up one way to signal it.
#[derive(Clone, Debug)]
pub struct LlamaAbortHandle {
    cancelled: Arc<AtomicBool>,
}

/// A context with a persistent native GGML thread pool attached.
///
/// The guard exclusively borrows both resources, so neither the context nor the pool can be moved
/// or dropped while llama.cpp retains the pool pointer. Dropping (or explicitly detaching) the guard
/// first detaches the pool from the native context.
#[derive(Debug)]
pub struct LlamaThreadPoolAttachment<'context, 'pool, 'model, 'backend> {
    context: &'context mut LlamaContext<'model>,
    _threadpool: &'pool mut LlamaThreadPool<'backend>,
}

/// Lifetime-safe attachment of distinct generation and prompt-processing pools.
#[derive(Debug)]
pub struct LlamaThreadPoolsAttachment<'context, 'main, 'batch, 'model, 'backend> {
    context: &'context mut LlamaContext<'model>,
    _main_threadpool: &'main mut LlamaThreadPool<'backend>,
    _batch_threadpool: &'batch mut LlamaThreadPool<'backend>,
}

impl LlamaThreadPoolsAttachment<'_, '_, '_, '_, '_> {
    /// Detach both pools before the guard would otherwise leave scope.
    pub fn detach(self) {}
}

impl<'model> Deref for LlamaThreadPoolsAttachment<'_, '_, '_, 'model, '_> {
    type Target = LlamaContext<'model>;

    fn deref(&self) -> &Self::Target {
        self.context
    }
}

impl DerefMut for LlamaThreadPoolsAttachment<'_, '_, '_, '_, '_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.context
    }
}

impl Drop for LlamaThreadPoolsAttachment<'_, '_, '_, '_, '_> {
    fn drop(&mut self) {
        unsafe { llama_cpp_sys_2::llama_detach_threadpool(self.context.context.as_ptr()) }
    }
}

impl LlamaThreadPoolAttachment<'_, '_, '_, '_> {
    /// Detach the thread pool before the guard would otherwise leave scope.
    pub fn detach(self) {}
}

impl<'model> Deref for LlamaThreadPoolAttachment<'_, '_, 'model, '_> {
    type Target = LlamaContext<'model>;

    fn deref(&self) -> &Self::Target {
        self.context
    }
}

impl DerefMut for LlamaThreadPoolAttachment<'_, '_, '_, '_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.context
    }
}

impl Drop for LlamaThreadPoolAttachment<'_, '_, '_, '_> {
    fn drop(&mut self) {
        unsafe { llama_cpp_sys_2::llama_detach_threadpool(self.context.context.as_ptr()) }
    }
}

impl LlamaAbortHandle {
    /// Request cancellation of the current native evaluation.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    /// Clear a previous cancellation request before starting another evaluation.
    pub fn reset(&self) {
        self.cancelled.store(false, Ordering::Release);
    }

    /// Whether cancellation has been requested.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

unsafe extern "C" fn abort_if_cancelled(data: *mut c_void) -> bool {
    if data.is_null() {
        return false;
    }
    // SAFETY: `install_abort_callback` stores this allocation on the context and unregisters the
    // callback before replacing or dropping it.
    let cancelled = unsafe { &*data.cast::<AtomicBool>() };
    cancelled.load(Ordering::Acquire)
}

impl Debug for LlamaContext<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaContext")
            .field("context", &self.context)
            .finish()
    }
}

impl<'model> LlamaContext<'model> {
    #[cfg(feature = "common")]
    pub(crate) fn has_initialized_logits(&self, index: i32) -> bool {
        logits_index_is_initialized(&self.initialized_logits, index)
    }

    pub(crate) fn new(
        llama_model: &'model LlamaModel,
        llama_context: NonNull<llama_cpp_sys_2::llama_context>,
        embeddings_enabled: bool,
    ) -> Self {
        Self {
            context: llama_context,
            model: llama_model,
            initialized_logits: Vec::new(),
            embeddings_enabled,
            abort_callback: None,
        }
    }

    /// Gets the max number of logical tokens that can be submitted to decode. Must be greater than or equal to [`Self::n_ubatch`].
    #[must_use]
    pub fn n_batch(&self) -> u32 {
        unsafe { llama_cpp_sys_2::llama_n_batch(self.context.as_ptr()) }
    }

    /// Gets the max number of physical tokens (hardware level) to decode in batch. Must be less than or equal to [`Self::n_batch`].
    #[must_use]
    pub fn n_ubatch(&self) -> u32 {
        unsafe { llama_cpp_sys_2::llama_n_ubatch(self.context.as_ptr()) }
    }

    /// Gets the size of the context.
    #[must_use]
    pub fn n_ctx(&self) -> u32 {
        unsafe { llama_cpp_sys_2::llama_n_ctx(self.context.as_ptr()) }
    }

    /// Gets the effective context capacity available to each sequence.
    #[must_use]
    pub fn n_ctx_seq(&self) -> u32 {
        unsafe { llama_cpp_sys_2::llama_n_ctx_seq(self.context.as_ptr()) }
    }

    /// Gets the recurrent-state sequence capacity.
    #[must_use]
    pub fn n_rs_seq(&self) -> u32 {
        unsafe { llama_cpp_sys_2::llama_n_rs_seq(self.context.as_ptr()) }
    }

    /// Gets the effective maximum number of sequences supported by this context.
    #[must_use]
    pub fn n_seq_max(&self) -> u32 {
        unsafe { llama_cpp_sys_2::llama_n_seq_max(self.context.as_ptr()) }
    }

    /// Gets the current thread count used for single-token generation.
    #[must_use]
    pub fn n_threads(&self) -> i32 {
        unsafe { llama_cpp_sys_2::llama_n_threads(self.context.as_ptr()) }
    }

    /// Gets the current thread count used for batched prompt processing.
    #[must_use]
    pub fn n_threads_batch(&self) -> i32 {
        unsafe { llama_cpp_sys_2::llama_n_threads_batch(self.context.as_ptr()) }
    }

    /// Update the generation and prompt-processing thread counts.
    pub fn set_n_threads(&mut self, n_threads: i32, n_threads_batch: i32) {
        unsafe {
            llama_cpp_sys_2::llama_set_n_threads(self.context.as_ptr(), n_threads, n_threads_batch);
        }
    }

    /// Attach one persistent GGML thread pool for both generation and batched prompt processing.
    ///
    /// This matches `llama_attach_threadpool(ctx, threadpool, nullptr)`, the configuration used by
    /// `llama-bench`. The returned guard provides context access and detaches automatically.
    pub fn attach_threadpool<'context, 'pool, 'backend>(
        &'context mut self,
        threadpool: &'pool mut LlamaThreadPool<'backend>,
    ) -> LlamaThreadPoolAttachment<'context, 'pool, 'model, 'backend> {
        unsafe {
            llama_cpp_sys_2::llama_attach_threadpool(
                self.context.as_ptr(),
                threadpool.as_ptr(),
                std::ptr::null_mut(),
            );
        }
        LlamaThreadPoolAttachment {
            context: self,
            _threadpool: threadpool,
        }
    }

    /// Attach distinct persistent pools for generation and batched prompt processing.
    pub fn attach_threadpools<'context, 'main, 'batch, 'backend>(
        &'context mut self,
        main_threadpool: &'main mut LlamaThreadPool<'backend>,
        batch_threadpool: &'batch mut LlamaThreadPool<'backend>,
    ) -> LlamaThreadPoolsAttachment<'context, 'main, 'batch, 'model, 'backend> {
        unsafe {
            llama_cpp_sys_2::llama_attach_threadpool(
                self.context.as_ptr(),
                main_threadpool.as_ptr(),
                batch_threadpool.as_ptr(),
            );
        }
        LlamaThreadPoolsAttachment {
            context: self,
            _main_threadpool: main_threadpool,
            _batch_threadpool: batch_threadpool,
        }
    }

    /// Block until all asynchronous backend work for this context has completed.
    pub fn synchronize(&mut self) {
        unsafe { llama_cpp_sys_2::llama_synchronize(self.context.as_ptr()) }
    }

    /// Install a lifetime-safe native abort callback and return its cancellation handle.
    ///
    /// llama.cpp currently checks this callback only on backends that support native aborts. Callers
    /// must still check the handle between decode calls for portable cancellation.
    pub fn install_abort_callback(&mut self) -> LlamaAbortHandle {
        self.install_abort_callback_with_flag(Arc::new(AtomicBool::new(false)))
    }

    /// Install a native abort callback backed by a caller-owned cancellation flag.
    pub fn install_abort_callback_with_flag(
        &mut self,
        cancelled: Arc<AtomicBool>,
    ) -> LlamaAbortHandle {
        self.clear_abort_callback();
        let callback_state = Arc::clone(&cancelled);
        let callback_data = Arc::as_ptr(&callback_state).cast_mut().cast::<c_void>();
        unsafe {
            llama_cpp_sys_2::llama_set_abort_callback(
                self.context.as_ptr(),
                Some(abort_if_cancelled),
                callback_data,
            );
        }
        self.abort_callback = Some(callback_state);
        LlamaAbortHandle { cancelled }
    }

    /// Remove the currently installed native abort callback, if any.
    pub fn clear_abort_callback(&mut self) {
        if self.abort_callback.is_some() {
            unsafe {
                llama_cpp_sys_2::llama_set_abort_callback(
                    self.context.as_ptr(),
                    None,
                    std::ptr::null_mut(),
                );
            }
            self.abort_callback = None;
        }
    }

    /// Decodes the batch.
    ///
    /// # Errors
    ///
    /// - `DecodeError` if the decoding failed.
    ///
    /// # Panics
    ///
    /// - the returned [`std::ffi::c_int`] from llama-cpp does not fit into a i32 (this should never happen on most systems)
    pub fn decode(&mut self, batch: &mut LlamaBatch) -> Result<(), DecodeError> {
        let result = unsafe { llama_cpp_sys_2::llama_decode(self.context.as_ptr(), batch.raw) };

        match NonZeroI32::new(result) {
            None => {
                self.initialized_logits
                    .clone_from(&batch.initialized_logits);
                Ok(())
            }
            Some(error) => Err(DecodeError::from(error)),
        }
    }

    /// Encodes the batch.
    ///
    /// # Errors
    ///
    /// - `EncodeError` if the decoding failed.
    ///
    /// # Panics
    ///
    /// - the returned [`std::ffi::c_int`] from llama-cpp does not fit into a i32 (this should never happen on most systems)
    pub fn encode(&mut self, batch: &mut LlamaBatch) -> Result<(), EncodeError> {
        let result = unsafe { llama_cpp_sys_2::llama_encode(self.context.as_ptr(), batch.raw) };

        match NonZeroI32::new(result) {
            None => {
                self.initialized_logits
                    .clone_from(&batch.initialized_logits);
                Ok(())
            }
            Some(error) => Err(EncodeError::from(error)),
        }
    }

    /// Get the embeddings for the `i`th sequence in the current context.
    ///
    /// # Returns
    ///
    /// A slice containing the embeddings for the last decoded batch.
    /// The size corresponds to the `n_embd` parameter of the context's model.
    ///
    /// # Errors
    ///
    /// - When the current context was constructed without enabling embeddings.
    /// - If the current model had a pooling type of [`llama_cpp_sys_2::LLAMA_POOLING_TYPE_NONE`]
    /// - If the given sequence index exceeds the max sequence id.
    ///
    /// # Panics
    ///
    /// * `n_embd` does not fit into a usize
    pub fn embeddings_seq_ith(&self, i: i32) -> Result<&[f32], EmbeddingsError> {
        if !self.embeddings_enabled {
            return Err(EmbeddingsError::NotEnabled);
        }

        let n_embd =
            usize::try_from(self.model.n_embd()).expect("n_embd does not fit into a usize");

        unsafe {
            let embedding = llama_cpp_sys_2::llama_get_embeddings_seq(self.context.as_ptr(), i);

            // Technically also possible whenever `i >= max(batch.n_seq)`, but can't check that here.
            if embedding.is_null() {
                Err(EmbeddingsError::NonePoolType)
            } else {
                Ok(slice::from_raw_parts(embedding, n_embd))
            }
        }
    }

    /// Get the embeddings for the `i`th token in the current context.
    ///
    /// # Returns
    ///
    /// A slice containing the embeddings for the last decoded batch of the given token.
    /// The size corresponds to the `n_embd` parameter of the context's model.
    ///
    /// # Errors
    ///
    /// - When the current context was constructed without enabling embeddings.
    /// - When the given token didn't have logits enabled when it was passed.
    /// - If the given token index exceeds the max token id.
    ///
    /// # Panics
    ///
    /// * `n_embd` does not fit into a usize
    pub fn embeddings_ith(&self, i: i32) -> Result<&[f32], EmbeddingsError> {
        if !self.embeddings_enabled {
            return Err(EmbeddingsError::NotEnabled);
        }

        let n_embd =
            usize::try_from(self.model.n_embd()).expect("n_embd does not fit into a usize");

        unsafe {
            let embedding = llama_cpp_sys_2::llama_get_embeddings_ith(self.context.as_ptr(), i);
            // Technically also possible whenever `i >= batch.n_tokens`, but no good way of checking `n_tokens` here.
            if embedding.is_null() {
                Err(EmbeddingsError::LogitsNotEnabled)
            } else {
                Ok(slice::from_raw_parts(embedding, n_embd))
            }
        }
    }

    /// Get the logits for the last token in the context.
    ///
    /// # Returns
    /// An iterator over unsorted `LlamaTokenData` containing the
    /// logits for the last token in the context.
    ///
    /// # Panics
    ///
    /// - underlying logits data is null
    pub fn candidates(&self) -> impl Iterator<Item = LlamaTokenData> + '_ {
        (0_i32..).zip(self.get_logits()).map(|(i, logit)| {
            let token = LlamaToken::new(i);
            LlamaTokenData::new(token, *logit, 0_f32)
        })
    }

    /// Get the token data array for the last token in the context.
    ///
    /// This is a convience method that implements:
    /// ```ignore
    /// LlamaTokenDataArray::from_iter(ctx.candidates(), false)
    /// ```
    ///
    /// # Panics
    ///
    /// - underlying logits data is null
    #[must_use]
    pub fn token_data_array(&self) -> LlamaTokenDataArray {
        LlamaTokenDataArray::from_iter(self.candidates(), false)
    }

    /// Token logits obtained from the last call to `decode()`.
    /// The logits for which `batch.logits[i] != 0` are stored contiguously
    /// in the order they have appeared in the batch.
    /// Rows: number of tokens for which `batch.logits[i] != 0`
    /// Cols: `n_vocab`
    ///
    /// # Returns
    ///
    /// A slice containing the logits for the last decoded token.
    /// The size corresponds to the `n_vocab` parameter of the context's model.
    ///
    /// # Panics
    ///
    /// - `n_vocab` does not fit into a usize
    /// - token data returned is null
    #[must_use]
    pub fn get_logits(&self) -> &[f32] {
        let data = unsafe { llama_cpp_sys_2::llama_get_logits(self.context.as_ptr()) };
        assert!(!data.is_null(), "logits data for last token is null");
        let len = usize::try_from(self.model.n_vocab()).expect("n_vocab does not fit into a usize");

        unsafe { slice::from_raw_parts(data, len) }
    }

    /// Get the logits for the ith token in the context.
    ///
    /// # Panics
    ///
    /// - logit `i` is not initialized.
    pub fn candidates_ith(&self, i: i32) -> impl Iterator<Item = LlamaTokenData> + '_ {
        (0_i32..).zip(self.get_logits_ith(i)).map(|(i, logit)| {
            let token = LlamaToken::new(i);
            LlamaTokenData::new(token, *logit, 0_f32)
        })
    }

    /// Get the token data array for the ith token in the context.
    ///
    /// This is a convience method that implements:
    /// ```ignore
    /// LlamaTokenDataArray::from_iter(ctx.candidates_ith(i), false)
    /// ```
    ///
    /// # Panics
    ///
    /// - logit `i` is not initialized.
    #[must_use]
    pub fn token_data_array_ith(&self, i: i32) -> LlamaTokenDataArray {
        LlamaTokenDataArray::from_iter(self.candidates_ith(i), false)
    }

    /// Get the logits for the ith token in the context.
    ///
    /// # Panics
    ///
    /// - `i` is greater than `n_ctx`
    /// - `n_vocab` does not fit into a usize
    /// - logit `i` is not initialized.
    #[must_use]
    pub fn get_logits_ith(&self, i: i32) -> &[f32] {
        assert!(
            self.initialized_logits.contains(&i),
            "logit {i} is not initialized. only {:?} is",
            self.initialized_logits
        );
        assert!(
            self.n_ctx() > u32::try_from(i).expect("i does not fit into a u32"),
            "n_ctx ({}) must be greater than i ({})",
            self.n_ctx(),
            i
        );

        let data = unsafe { llama_cpp_sys_2::llama_get_logits_ith(self.context.as_ptr(), i) };
        let len = usize::try_from(self.model.n_vocab()).expect("n_vocab does not fit into a usize");

        unsafe { slice::from_raw_parts(data, len) }
    }

    /// Reset the timings for the context.
    pub fn reset_timings(&mut self) {
        unsafe { llama_cpp_sys_2::llama_perf_context_reset(self.context.as_ptr()) }
    }

    /// Returns the timings for the context.
    pub fn timings(&mut self) -> LlamaTimings {
        let timings = unsafe { llama_cpp_sys_2::llama_perf_context(self.context.as_ptr()) };
        LlamaTimings { timings }
    }

    /// Sets a lora adapter.
    ///
    /// # Errors
    ///
    /// See [`LlamaLoraAdapterSetError`] for more information.
    pub fn lora_adapter_set(
        &self,
        adapter: &mut LlamaLoraAdapter,
        scale: f32,
    ) -> Result<(), LlamaLoraAdapterSetError> {
        let mut adapters = [adapter.lora_adapter.as_ptr()];
        let mut scales = [scale];
        let err_code = unsafe {
            llama_cpp_sys_2::llama_set_adapters_lora(
                self.context.as_ptr(),
                adapters.as_mut_ptr(),
                1,
                scales.as_mut_ptr(),
            )
        };
        if err_code != 0 {
            return Err(LlamaLoraAdapterSetError::ErrorResult(err_code));
        }

        tracing::debug!("Set lora adapter");
        Ok(())
    }

    /// Remove all lora adapters.
    ///
    /// Note: The upstream API now replaces all adapters at once via
    /// `llama_set_adapters_lora`. This clears all adapters from the context.
    ///
    /// # Errors
    ///
    /// See [`LlamaLoraAdapterRemoveError`] for more information.
    pub fn lora_adapter_remove(
        &self,
        _adapter: &mut LlamaLoraAdapter,
    ) -> Result<(), LlamaLoraAdapterRemoveError> {
        let err_code = unsafe {
            llama_cpp_sys_2::llama_set_adapters_lora(
                self.context.as_ptr(),
                std::ptr::null_mut(),
                0,
                std::ptr::null_mut(),
            )
        };
        if err_code != 0 {
            return Err(LlamaLoraAdapterRemoveError::ErrorResult(err_code));
        }

        tracing::debug!("Remove lora adapter");
        Ok(())
    }

    /// Print a breakdown of per-device memory use to the default logger.
    #[cfg(feature = "common")]
    pub fn print_memory_breakdown(&self) {
        unsafe { llama_cpp_sys_2::llama_rs_memory_breakdown_print(self.context.as_ptr()) }
    }

    /// Return llama.cpp's resident model, context, and compute allocations by physical location.
    ///
    /// # Errors
    ///
    /// Returns an error if llama.cpp cannot construct or decode the typed report.
    #[cfg(feature = "common")]
    pub fn memory_breakdown(&self) -> Result<Vec<LlamaMemoryBreakdown>, LlamaMemoryBreakdownError> {
        struct Report(NonNull<llama_cpp_sys_2::llama_rs_memory_breakdown_report>);
        impl Drop for Report {
            fn drop(&mut self) {
                unsafe { llama_cpp_sys_2::llama_rs_memory_breakdown_free(self.0.as_ptr()) }
            }
        }

        fn text(value: *const std::os::raw::c_char) -> String {
            if value.is_null() {
                return String::new();
            }
            unsafe { CStr::from_ptr(value) }
                .to_string_lossy()
                .into_owned()
        }

        let mut raw_report = std::ptr::null_mut();
        let mut raw_error = std::ptr::null_mut();
        let status = unsafe {
            llama_cpp_sys_2::llama_rs_memory_breakdown_create(
                self.context.as_ptr(),
                &raw mut raw_report,
                &raw mut raw_error,
            )
        };
        if status != llama_cpp_sys_2::LLAMA_RS_STATUS_OK {
            let message = if raw_error.is_null() {
                String::new()
            } else {
                let message = unsafe { CStr::from_ptr(raw_error) }
                    .to_string_lossy()
                    .into_owned();
                unsafe { llama_cpp_sys_2::llama_rs_string_free(raw_error) };
                message
            };
            return Err(LlamaMemoryBreakdownError::Native { status, message });
        }
        if !raw_error.is_null() {
            unsafe { llama_cpp_sys_2::llama_rs_string_free(raw_error) };
        }
        let report = Report(NonNull::new(raw_report).ok_or(LlamaMemoryBreakdownError::NullReport)?);
        let count = unsafe { llama_cpp_sys_2::llama_rs_memory_breakdown_count(report.0.as_ptr()) };
        let mut result = Vec::new();
        result
            .try_reserve_exact(count)
            .map_err(|_| LlamaMemoryBreakdownError::RustAllocation { requested: count })?;
        for index in 0..count {
            let mut entry =
                std::mem::MaybeUninit::<llama_cpp_sys_2::llama_rs_memory_breakdown_entry>::uninit();
            if !unsafe {
                llama_cpp_sys_2::llama_rs_memory_breakdown_get(
                    report.0.as_ptr(),
                    index,
                    entry.as_mut_ptr(),
                )
            } {
                return Err(LlamaMemoryBreakdownError::InvalidEntry(index));
            }
            let entry = unsafe { entry.assume_init() };
            let location = match entry.location {
                llama_cpp_sys_2::LLAMA_RS_MEMORY_LOCATION_HOST => LlamaMemoryLocation::Host,
                llama_cpp_sys_2::LLAMA_RS_MEMORY_LOCATION_DEVICE => LlamaMemoryLocation::Device {
                    backend: text(entry.backend),
                    physical_id: (!entry.device_id.is_null()).then(|| text(entry.device_id)),
                    native_index: entry.native_index,
                },
                other => {
                    return Err(LlamaMemoryBreakdownError::UnknownLocation(i64::from(other)));
                }
            };
            result.push(LlamaMemoryBreakdown {
                location,
                model_bytes: entry.model_bytes,
                context_bytes: entry.context_bytes,
                compute_bytes: entry.compute_bytes,
            });
        }
        Ok(result)
    }
}

#[cfg(any(feature = "common", test))]
fn logits_index_is_initialized(initialized: &[i32], index: i32) -> bool {
    if index >= 0 {
        return initialized.contains(&index);
    }

    // llama.cpp interprets negative indices as rows counted backwards from the
    // compact output buffer. Avoid entering its aborting error path when no
    // output exists or the negative index is out of range.
    let output_count = i64::try_from(initialized.len()).unwrap_or(i64::MAX);
    !initialized.is_empty() && i64::from(index) >= -output_count
}

impl Drop for LlamaContext<'_> {
    fn drop(&mut self) {
        self.clear_abort_callback();
        unsafe { llama_cpp_sys_2::llama_free(self.context.as_ptr()) }
    }
}

#[cfg(test)]
mod output_index_tests {
    use super::logits_index_is_initialized;

    #[test]
    fn validates_batch_and_negative_output_indices() {
        assert!(!logits_index_is_initialized(&[], 0));
        assert!(!logits_index_is_initialized(&[], -1));

        let outputs = [2, 7];
        assert!(logits_index_is_initialized(&outputs, 2));
        assert!(logits_index_is_initialized(&outputs, 7));
        assert!(!logits_index_is_initialized(&outputs, 0));
        assert!(logits_index_is_initialized(&outputs, -1));
        assert!(logits_index_is_initialized(&outputs, -2));
        assert!(!logits_index_is_initialized(&outputs, -3));
    }
}
