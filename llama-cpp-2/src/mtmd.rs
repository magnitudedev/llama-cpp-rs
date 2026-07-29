//! Safe wrapper around multimodal (MTMD) functionality in llama.cpp.
//!
//! This module provides Rust bindings for llama.cpp's multimodal support,
//! allowing processing of text, image, and audio inputs through a unified interface.
//!
//! # Warning
//! This API is experimental and subject to breaking changes.
use std::ffi::{c_char, CStr, CString};
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::ptr;
use std::ptr::NonNull;
use std::slice;
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::context::params::FlashAttentionPolicy;
use crate::context::LlamaContext;
use crate::model::LlamaModel;
use crate::token::LlamaToken;

// Upstream `mtmd_get_memory_usage` temporarily replaces mtmd's process-global callback/userdata
// pair with ordinary, non-atomic stores. A write guard makes that operation exclusive while every
// other safe mtmd native entry (including owner destructors and borrowed-view getters) is covered
// by a read guard. Memory-report projection and destruction remain under the originating write
// guard. This lets operations on independent contexts remain concurrent without allowing a logger
// swap to race any native log emission. Workspace consumers are forbidden from bypassing this
// module and calling the sys crate directly.
static MTMD_LOG_STATE: RwLock<()> = RwLock::new(());

fn mtmd_native_read() -> RwLockReadGuard<'static, ()> {
    MTMD_LOG_STATE
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn mtmd_native_write() -> RwLockWriteGuard<'static, ()> {
    MTMD_LOG_STATE
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Input chunk types for multimodal data
///
/// # Examples
///
/// ```
/// use llama_cpp_2::mtmd::MtmdInputChunkType;
///
/// let text_chunk = MtmdInputChunkType::Text;
/// let image_chunk = MtmdInputChunkType::Image;
/// let audio_chunk = MtmdInputChunkType::Audio;
///
/// assert_eq!(text_chunk, MtmdInputChunkType::Text);
/// assert_ne!(text_chunk, image_chunk);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MtmdInputChunkType {
    /// Text input chunk
    Text,
    /// Image input chunk
    Image,
    /// Audio input chunk
    Audio,
    /// A chunk type introduced by a newer llama.cpp revision.
    Unknown(i64),
}

impl MtmdInputChunkType {
    fn from_raw(chunk_type: llama_cpp_sys_2::mtmd_input_chunk_type) -> Self {
        match chunk_type {
            llama_cpp_sys_2::MTMD_INPUT_CHUNK_TYPE_TEXT => Self::Text,
            llama_cpp_sys_2::MTMD_INPUT_CHUNK_TYPE_IMAGE => Self::Image,
            llama_cpp_sys_2::MTMD_INPUT_CHUNK_TYPE_AUDIO => Self::Audio,
            raw => Self::Unknown(i64::from(raw)),
        }
    }
}

/// Configuration parameters for MTMD context
///
/// # Examples
///
/// ```
/// use llama_cpp_2::context::params::FlashAttentionPolicy;
/// use llama_cpp_2::mtmd::{MtmdContextParams, mtmd_default_marker};
/// use std::ffi::CString;
///
/// let params = MtmdContextParams {
///     use_gpu: false,
///     print_timings: true,
///     n_threads: 4,
///     media_marker: CString::new(mtmd_default_marker()).unwrap(),
///     flash_attention: FlashAttentionPolicy::Auto,
///     warmup: true,
///     image_min_tokens: None,
///     image_max_tokens: None,
/// };
/// ```
#[derive(Debug, Clone)]
pub struct MtmdContextParams {
    /// Whether to use GPU acceleration
    pub use_gpu: bool,
    /// Whether to print timing information
    pub print_timings: bool,
    /// Number of threads to use for processing
    pub n_threads: i32,
    /// Media marker string used to identify media positions in text
    pub media_marker: CString,
    /// Whether the projector may use Flash Attention.
    pub flash_attention: FlashAttentionPolicy,
    /// Whether to run the native projector warmup during initialization.
    pub warmup: bool,
    /// Minimum number of tokens used to represent an image.
    /// Controls the visual token budget lower bound. `None` uses the model default.
    /// Gemma 4 supported budgets: 70, 140, 280, 560, 1120.
    pub image_min_tokens: Option<NonZeroU32>,
    /// Maximum number of tokens used to represent an image.
    /// Controls the visual token budget upper bound. `None` uses the model default.
    /// Lower values reduce memory and compute at the cost of visual detail.
    /// Gemma 4 supported budgets: 70, 140, 280, 560, 1120.
    pub image_max_tokens: Option<NonZeroU32>,
}

impl Default for MtmdContextParams {
    fn default() -> Self {
        let params = {
            let _native_guard = mtmd_native_read();
            unsafe { llama_cpp_sys_2::mtmd_context_params_default() }
        };
        let media_marker = match CString::new(mtmd_default_marker()) {
            Ok(marker) => marker,
            // `mtmd_default_marker` comes from a NUL-terminated C string and cannot contain an
            // interior NUL. Keep a non-panicking fallback in case a future native revision breaks
            // that contract.
            Err(_) => unsafe { CString::from_vec_unchecked(b"<__media__>".to_vec()) },
        };
        Self {
            use_gpu: params.use_gpu,
            print_timings: params.print_timings,
            n_threads: params.n_threads,
            media_marker,
            flash_attention: params.flash_attn_type.into(),
            warmup: params.warmup,
            image_min_tokens: None,
            image_max_tokens: None,
        }
    }
}

impl MtmdContextParams {
    fn to_raw(&self) -> Result<llama_cpp_sys_2::mtmd_context_params, MtmdContextParamsError> {
        fn token_budget(
            field: &'static str,
            value: Option<NonZeroU32>,
        ) -> Result<i32, MtmdContextParamsError> {
            value.map_or(Ok(-1), |value| {
                i32::try_from(value.get()).map_err(|_| {
                    MtmdContextParamsError::ImageTokenBudgetOverflow {
                        field,
                        value: value.get(),
                    }
                })
            })
        }

        let _native_guard = mtmd_native_read();
        let mut context = unsafe { llama_cpp_sys_2::mtmd_context_params_default() };
        let Self {
            use_gpu,
            print_timings,
            n_threads,
            media_marker,
            flash_attention,
            warmup,
            image_min_tokens,
            image_max_tokens,
        } = self;

        context.use_gpu = *use_gpu;
        context.print_timings = *print_timings;
        context.n_threads = *n_threads;
        context.media_marker = media_marker.as_ptr();
        context.flash_attn_type = (*flash_attention).into();
        context.warmup = *warmup;
        context.image_min_tokens = token_budget("image_min_tokens", *image_min_tokens)?;
        context.image_max_tokens = token_budget("image_max_tokens", *image_max_tokens)?;
        // These callbacks instrument graph evaluation. They are intentionally not exposed as a
        // general cancellation API and must never inherit an ambient native callback.
        context.cb_eval = None;
        context.cb_eval_user_data = ptr::null_mut();

        Ok(context)
    }
}

/// Text input configuration
///
/// # Examples
///
/// ```
/// use llama_cpp_2::mtmd::MtmdInputText;
///
/// let input = MtmdInputText {
///     text: "Describe this image.".to_string(),
///     add_special: true,
///     parse_special: true,
/// };
/// ```
#[derive(Debug, Clone)]
pub struct MtmdInputText {
    /// The input text string
    pub text: String,
    /// Whether to add special tokens
    pub add_special: bool,
    /// Whether to parse special tokens
    pub parse_special: bool,
}

/// Safe wrapper around `mtmd_context`.
///
/// This represents an initialized multimodal context that can process
/// text, images, and audio through llama.cpp's multimodal interface.
#[derive(Debug)]
pub struct MtmdContext<'model> {
    context: NonNull<llama_cpp_sys_2::mtmd_context>,
    model: &'model LlamaModel,
}

// SAFETY: the context is uniquely owned and its retained model reference is `Send + Sync`.
// Projector operations are synchronous; moving the unique owner between threads is therefore
// sound. Some native preprocessing is reachable through `&self`, so concurrent access is not
// established here and we deliberately do not implement `Sync`.
unsafe impl Send for MtmdContext<'_> {}

impl<'model> MtmdContext<'model> {
    /// Initialize MTMD context from a multimodal projection file.
    ///
    /// # Arguments
    ///
    /// * `mmproj_path` - Path to the multimodal projection file
    /// * `text_model` - Reference to the text model
    /// * `params` - Configuration parameters for the MTMD context
    ///
    /// # Returns
    ///
    /// Returns `Ok(MtmdContext)` on success, or `Err(MtmdInitError)` on failure.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The path cannot be converted to a C string
    /// - The underlying C function returns null (indicating initialization failure)
    pub fn init_from_file(
        mmproj_path: impl AsRef<Path>,
        text_model: &'model LlamaModel,
        params: &MtmdContextParams,
    ) -> Result<Self, MtmdInitError> {
        let path_cstr = path_to_c_string(mmproj_path.as_ref())?;
        let ctx_params = params.to_raw()?;
        let _native_guard = mtmd_native_read();

        let mut context = ptr::null_mut();
        let mut error = ptr::null_mut();
        let status = unsafe {
            llama_cpp_sys_2::llama_rs_mtmd_context_init_from_file(
                path_cstr.as_ptr(),
                text_model.model.as_ptr(),
                &raw const ctx_params,
                &raw mut context,
                &raw mut error,
            )
        };
        check_mtmd_native_status(status, error)?;

        let context = NonNull::new(context).ok_or(MtmdInitError::NullResult)?;
        Ok(Self {
            context,
            model: text_model,
        })
    }

    /// Check whether non-causal attention mask is needed before `llama_decode`.
    #[must_use]
    pub fn decode_use_non_causal(&self) -> bool {
        let _native_guard = mtmd_native_read();
        unsafe {
            llama_cpp_sys_2::mtmd_decode_use_non_causal(self.context.as_ptr(), std::ptr::null())
        }
    }

    /// Check whether the current model uses M-RoPE for `llama_decode`.
    ///
    /// M-RoPE (Multimodal Rotary Position Embedding) affects how positions
    /// are calculated for multimodal inputs.
    #[must_use]
    pub fn decode_use_mrope(&self) -> bool {
        let _native_guard = mtmd_native_read();
        unsafe { llama_cpp_sys_2::mtmd_decode_use_mrope(self.context.as_ptr()) }
    }

    /// Check whether the current model supports vision input.
    #[must_use]
    pub fn support_vision(&self) -> bool {
        let _native_guard = mtmd_native_read();
        unsafe { llama_cpp_sys_2::mtmd_support_vision(self.context.as_ptr()) }
    }

    /// Check whether the current model supports audio input.
    #[must_use]
    pub fn support_audio(&self) -> bool {
        let _native_guard = mtmd_native_read();
        unsafe { llama_cpp_sys_2::mtmd_support_audio(self.context.as_ptr()) }
    }

    /// Get audio sample rate in Hz (e.g., 16000 for Whisper).
    /// Returns None if audio is not supported.
    #[must_use]
    pub fn get_audio_sample_rate(&self) -> Option<u32> {
        let _native_guard = mtmd_native_read();
        let rate = unsafe { llama_cpp_sys_2::mtmd_get_audio_sample_rate(self.context.as_ptr()) };
        (rate > 0).then_some(rate.unsigned_abs())
    }

    /// Backward-compatible alias for the audio sample rate getter.
    #[must_use]
    pub fn get_audio_bitrate(&self) -> Option<u32> {
        self.get_audio_sample_rate()
    }

    /// Tokenize input text and bitmaps into chunks.
    ///
    /// The input text must contain media markers (default: `<__media__>`) that will be
    /// replaced with the corresponding bitmap data from the `bitmaps` array.
    /// The number of bitmaps must equal the number of markers in the text.
    ///
    /// # Arguments
    ///
    /// * `text` - Text input configuration containing the text and tokenization options
    /// * `bitmaps` - Array of bitmaps (images/audio) to replace markers with
    ///
    /// # Returns
    ///
    /// Returns `Ok(MtmdInputChunks)` containing the tokenized chunks on success.
    ///
    /// # Errors
    ///
    /// * `BitmapCountMismatch` - Number of bitmaps doesn't match number of markers
    /// * `ImagePreprocessingError` - Error occurred during image preprocessing
    /// * `UnknownError` - Other tokenization error occurred
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use llama_cpp_2::mtmd::*;
    /// # fn example(ctx: &MtmdContext<'_>, bitmap: &MtmdBitmap) -> Result<(), Box<dyn std::error::Error>> {
    /// let text = MtmdInputText {
    ///     text: "Here is an image: <__media__>\nDescribe it.".to_string(),
    ///     add_special: true,
    ///     parse_special: true,
    /// };
    /// let chunks = ctx.tokenize(text, &[bitmap])?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn tokenize(
        &self,
        text: MtmdInputText,
        bitmaps: &[&MtmdBitmap],
    ) -> Result<MtmdInputChunks, MtmdTokenizeError> {
        let chunks = MtmdInputChunks::new()?;
        let text_cstring = CString::new(text.text)?;
        let input_text = llama_cpp_sys_2::mtmd_input_text {
            text: text_cstring.as_ptr(),
            text_len: text_cstring.as_bytes().len(),
            add_special: text.add_special,
            parse_special: text.parse_special,
        };

        // Create bitmap pointers
        let bitmap_ptrs: Vec<*const llama_cpp_sys_2::mtmd_bitmap> = bitmaps
            .iter()
            .map(|b| b.bitmap.as_ptr().cast_const())
            .collect();

        let _native_guard = mtmd_native_read();
        let mut result = 0;
        let mut error = ptr::null_mut();
        let empty_bitmap = ptr::null();
        let bitmap_array = if bitmap_ptrs.is_empty() {
            &raw const empty_bitmap
        } else {
            bitmap_ptrs.as_ptr()
        };
        let status = unsafe {
            llama_cpp_sys_2::llama_rs_mtmd_tokenize(
                self.context.as_ptr(),
                chunks.chunks.as_ptr(),
                &raw const input_text,
                bitmap_array.cast_mut(),
                bitmaps.len(),
                &raw mut result,
                &raw mut error,
            )
        };
        check_mtmd_native_status(status, error)?;

        match result {
            0 => Ok(chunks),
            1 => Err(MtmdTokenizeError::BitmapCountMismatch),
            2 => Err(MtmdTokenizeError::ImagePreprocessingError),
            _ => Err(MtmdTokenizeError::UnknownError(result)),
        }
    }

    /// Encode a chunk for image/audio processing.
    ///
    /// This function processes image or audio chunks by encoding them into
    /// embeddings that can be used by the language model. The embeddings
    /// can be retrieved using `get_output_embeddings()`.
    ///
    /// # Arguments
    ///
    /// * `chunk` - The input chunk to encode (should be image or audio type)
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success.
    ///
    /// # Errors
    ///
    /// Returns `MtmdEncodeError::EncodeFailure` if encoding fails.
    pub fn encode_chunk(&mut self, chunk: &MtmdInputChunk<'_>) -> Result<(), MtmdEncodeError> {
        if let MtmdInputChunkType::Unknown(raw) = chunk.chunk_type() {
            return Err(MtmdEncodeError::UnsupportedType { raw });
        }
        let _native_guard = mtmd_native_read();
        let mut result = 0;
        let mut error = ptr::null_mut();
        let status = unsafe {
            llama_cpp_sys_2::llama_rs_mtmd_encode_chunk(
                self.context.as_ptr(),
                chunk.chunk.as_ptr(),
                &raw mut result,
                &raw mut error,
            )
        };
        check_mtmd_native_status(status, error)?;

        if result == 0 {
            Ok(())
        } else {
            Err(MtmdEncodeError::EncodeFailure(result))
        }
    }
}

impl Drop for MtmdContext<'_> {
    fn drop(&mut self) {
        let _native_guard = mtmd_native_read();
        unsafe { llama_cpp_sys_2::mtmd_free(self.context.as_ptr()) }
    }
}

/// Safe wrapper around `mtmd_bitmap`.
///
/// Represents bitmap data for images or audio that can be processed
/// by the multimodal system. For images, data is stored in RGB format.
/// For audio, data is stored as PCM F32 samples.
#[derive(Debug)]
pub struct MtmdBitmap {
    bitmap: NonNull<llama_cpp_sys_2::mtmd_bitmap>,
}

// Ownership may move to an executor thread. The native ID setter mutates the bitmap, so
// deliberately do not implement Sync.
unsafe impl Send for MtmdBitmap {}

impl MtmdBitmap {
    /// Create a bitmap from image data in RGB format.
    ///
    /// # Arguments
    ///
    /// * `nx` - Width of the image in pixels
    /// * `ny` - Height of the image in pixels
    /// * `data` - Image data in RGBRGBRGB... format (must be exactly `nx * ny * 3` bytes)
    ///
    /// # Returns
    ///
    /// Returns `Ok(MtmdBitmap)` on success.
    ///
    /// # Errors
    ///
    /// * `InvalidDataSize` - Data length doesn't match `nx * ny * 3`
    /// * `NullResult` - Underlying C function returned null
    ///
    /// # Examples
    ///
    /// ```
    /// use llama_cpp_2::mtmd::MtmdBitmap;
    ///
    /// // Create a 2x2 red image
    /// let red_pixel = [255, 0, 0]; // RGB values for red
    /// let image_data = red_pixel.repeat(4); // 2x2 = 4 pixels
    ///
    /// let bitmap = MtmdBitmap::from_image_data(2, 2, &image_data);
    /// assert!(bitmap.is_ok());
    /// ```
    pub fn from_image_data(nx: u32, ny: u32, data: &[u8]) -> Result<Self, MtmdBitmapError> {
        let expected = (nx as usize)
            .checked_mul(ny as usize)
            .and_then(|pixels| pixels.checked_mul(3))
            .ok_or(MtmdBitmapError::DimensionsOverflow { nx, ny })?;
        if data.len() != expected {
            return Err(MtmdBitmapError::InvalidDataSize);
        }

        let _native_guard = mtmd_native_read();
        let mut bitmap = ptr::null_mut();
        let mut error = ptr::null_mut();
        let status = unsafe {
            llama_cpp_sys_2::llama_rs_mtmd_bitmap_init_image(
                nx,
                ny,
                data.as_ptr(),
                &raw mut bitmap,
                &raw mut error,
            )
        };
        check_mtmd_native_status(status, error)?;

        let bitmap = NonNull::new(bitmap).ok_or(MtmdBitmapError::NullResult)?;
        Ok(Self { bitmap })
    }

    /// Create a bitmap from audio data in PCM F32 format.
    ///
    /// # Arguments
    ///
    /// * `data` - Audio samples as 32-bit floating point values
    ///
    /// # Returns
    ///
    /// Returns `Ok(MtmdBitmap)` on success.
    ///
    /// # Errors
    ///
    /// * `NullResult` - Underlying C function returned null
    ///
    /// # Examples
    ///
    /// ```
    /// use llama_cpp_2::mtmd::MtmdBitmap;
    ///
    /// // Create a simple sine wave audio sample
    /// let audio_data: Vec<f32> = (0..100)
    ///     .map(|i| (i as f32 * 0.1).sin())
    ///     .collect();
    ///
    /// let bitmap = MtmdBitmap::from_audio_data(&audio_data);
    /// // Note: This will likely fail without proper MTMD context setup
    /// ```
    pub fn from_audio_data(data: &[f32]) -> Result<Self, MtmdBitmapError> {
        if data.len() > u32::MAX as usize {
            return Err(MtmdBitmapError::SampleCountOverflow { length: data.len() });
        }
        let _native_guard = mtmd_native_read();
        let mut bitmap = ptr::null_mut();
        let mut error = ptr::null_mut();
        let status = unsafe {
            llama_cpp_sys_2::llama_rs_mtmd_bitmap_init_audio(
                data.len(),
                data.as_ptr(),
                &raw mut bitmap,
                &raw mut error,
            )
        };
        check_mtmd_native_status(status, error)?;

        let bitmap = NonNull::new(bitmap).ok_or(MtmdBitmapError::NullResult)?;
        Ok(Self { bitmap })
    }

    /// Create a bitmap from a file.
    ///
    /// Supported formats:
    /// - Images: formats supported by `stb_image` (jpg, png, bmp, gif, etc.)
    /// - Audio: formats supported by miniaudio (wav, mp3, flac)
    ///
    /// Audio files are auto-detected based on magic bytes.
    ///
    /// # Arguments
    ///
    /// * `ctx` - MTMD context for processing
    /// * `path` - Path to the image or audio file
    /// * `placeholder` - If `true`, build a data-less bitmap (dimensions/length only, with no
    ///   decoded pixels or audio samples) — useful for counting tokens without loading the media.
    ///   If `false`, decode and load the actual data.
    ///
    /// # Returns
    ///
    /// Returns `Ok(MtmdBitmap)` on success.
    ///
    /// # Errors
    ///
    /// * `CStringError` - Path contains null bytes
    /// * `NullResult` - File could not be loaded or processed
    ///
    /// This function is thread-safe.
    pub fn from_file(
        ctx: &MtmdContext<'_>,
        path: impl AsRef<Path>,
        placeholder: bool,
    ) -> Result<Self, MtmdBitmapError> {
        let path_cstr = path_to_c_string(path.as_ref())?;
        let _native_guard = mtmd_native_read();
        let mut bitmap = ptr::null_mut();
        let mut error = ptr::null_mut();
        let status = unsafe {
            llama_cpp_sys_2::llama_rs_mtmd_bitmap_init_from_file(
                ctx.context.as_ptr(),
                path_cstr.as_ptr(),
                placeholder,
                &raw mut bitmap,
                &raw mut error,
            )
        };
        check_mtmd_native_status(status, error)?;

        let bitmap = NonNull::new(bitmap).ok_or(MtmdBitmapError::NullResult)?;
        Ok(Self { bitmap })
    }

    /// Create a bitmap from a buffer containing file data.
    ///
    /// Supported formats:
    /// - Images: formats supported by `stb_image` (jpg, png, bmp, gif, etc.)
    /// - Audio: formats supported by miniaudio (wav, mp3, flac)
    ///
    /// Audio files are auto-detected based on magic bytes.
    ///
    /// # Arguments
    ///
    /// * `ctx` - MTMD context for processing
    /// * `data` - Buffer containing the file data
    /// * `placeholder` - If `true`, build a data-less bitmap (dimensions/length only, with no
    ///   decoded pixels or audio samples) — useful for counting tokens without loading the media.
    ///   If `false`, decode and load the actual data.
    ///
    /// # Returns
    ///
    /// Returns `Ok(MtmdBitmap)` on success.
    ///
    /// # Errors
    ///
    /// * `NullResult` - Buffer could not be processed
    ///
    /// This function is thread-safe.
    pub fn from_buffer(
        ctx: &MtmdContext<'_>,
        data: &[u8],
        placeholder: bool,
    ) -> Result<Self, MtmdBitmapError> {
        let _native_guard = mtmd_native_read();
        let mut bitmap = ptr::null_mut();
        let mut error = ptr::null_mut();
        let status = unsafe {
            llama_cpp_sys_2::llama_rs_mtmd_bitmap_init_from_buffer(
                ctx.context.as_ptr(),
                data.as_ptr(),
                data.len(),
                placeholder,
                &raw mut bitmap,
                &raw mut error,
            )
        };
        check_mtmd_native_status(status, error)?;

        let bitmap = NonNull::new(bitmap).ok_or(MtmdBitmapError::NullResult)?;
        Ok(Self { bitmap })
    }

    /// Get bitmap width in pixels.
    #[must_use]
    pub fn nx(&self) -> u32 {
        let _native_guard = mtmd_native_read();
        unsafe { llama_cpp_sys_2::mtmd_bitmap_get_nx(self.bitmap.as_ptr()) }
    }

    /// Get bitmap height in pixels.
    #[must_use]
    pub fn ny(&self) -> u32 {
        let _native_guard = mtmd_native_read();
        unsafe { llama_cpp_sys_2::mtmd_bitmap_get_ny(self.bitmap.as_ptr()) }
    }

    /// Get bitmap data as a byte slice.
    ///
    /// For images: RGB format with length `nx * ny * 3`
    /// For audio: PCM F32 format with length `n_samples * 4`
    ///
    /// # Errors
    ///
    /// Returns an error if native bitmap data is null for a nonempty bitmap.
    pub fn data(&self) -> Result<&[u8], MtmdBitmapDataError> {
        let _native_guard = mtmd_native_read();
        let ptr = unsafe { llama_cpp_sys_2::mtmd_bitmap_get_data(self.bitmap.as_ptr()) };
        let len = unsafe { llama_cpp_sys_2::mtmd_bitmap_get_n_bytes(self.bitmap.as_ptr()) };
        // SAFETY: mtmd owns `ptr` for at least the lifetime of this bitmap. The helper handles
        // the native empty/null convention before constructing a slice.
        unsafe { bitmap_data_from_raw(ptr, len) }
    }

    /// Check if this bitmap contains audio data (vs image data).
    #[must_use]
    pub fn is_audio(&self) -> bool {
        let _native_guard = mtmd_native_read();
        unsafe { llama_cpp_sys_2::mtmd_bitmap_is_audio(self.bitmap.as_ptr()) }
    }

    /// Get the bitmap's optional ID string.
    ///
    /// Bitmap ID is useful for KV cache tracking and can e.g. be calculated
    /// based on a hash of the bitmap data.
    ///
    /// # Errors
    ///
    /// Returns an error if the native ID is not valid UTF-8.
    pub fn id(&self) -> Result<Option<&str>, MtmdBitmapIdError> {
        let _native_guard = mtmd_native_read();
        let ptr = unsafe { llama_cpp_sys_2::mtmd_bitmap_get_id(self.bitmap.as_ptr()) };
        if ptr.is_null() {
            return Ok(None);
        }
        let id = unsafe { CStr::from_ptr(ptr) };
        if id.is_empty() {
            return Ok(None);
        }
        id.to_str()
            .map(Some)
            .map_err(MtmdBitmapIdError::InvalidUtf8)
    }

    /// Set the bitmap's ID string.
    ///
    /// Bitmap ID is useful for KV cache tracking and can e.g. be calculated
    /// based on a hash of the bitmap data.
    ///
    /// # Arguments
    ///
    /// * `id` - The ID string to set
    ///
    /// # Errors
    ///
    /// Returns an error if the ID string contains null bytes.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # use llama_cpp_2::mtmd::MtmdBitmap;
    /// # fn example(bitmap: &mut MtmdBitmap) -> Result<(), Box<dyn std::error::Error>> {
    /// bitmap.set_id("image_001")?;
    /// assert_eq!(bitmap.id()?, Some("image_001"));
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_id(&mut self, id: &str) -> Result<(), MtmdBitmapIdError> {
        let id_cstr = CString::new(id)?;
        let _native_guard = mtmd_native_read();
        let mut error = ptr::null_mut();
        let status = unsafe {
            llama_cpp_sys_2::llama_rs_mtmd_bitmap_set_id(
                self.bitmap.as_ptr(),
                id_cstr.as_ptr(),
                &raw mut error,
            )
        };
        check_mtmd_native_status(status, error)?;
        Ok(())
    }
}

impl Drop for MtmdBitmap {
    fn drop(&mut self) {
        let _native_guard = mtmd_native_read();
        unsafe { llama_cpp_sys_2::mtmd_bitmap_free(self.bitmap.as_ptr()) }
    }
}

/// Safe wrapper around `mtmd_input_chunks`.
///
/// This is a collection of input chunks created from tokenizing text and media.
/// The chunks represent the tokenized input that can be processed by the model,
/// with text chunks containing tokens and media chunks containing embeddings.
#[derive(Debug)]
pub struct MtmdInputChunks {
    chunks: NonNull<llama_cpp_sys_2::mtmd_input_chunks>,
}

impl MtmdInputChunks {
    /// Create a new empty input chunks collection
    ///
    /// # Errors
    /// Returns [`MtmdInputChunksError::NullResult`] if native allocation fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use llama_cpp_2::mtmd::MtmdInputChunks;
    ///
    /// let chunks = MtmdInputChunks::new().unwrap();
    /// assert_eq!(chunks.len(), 0);
    /// assert!(chunks.is_empty());
    /// ```
    pub fn new() -> Result<Self, MtmdInputChunksError> {
        let _native_guard = mtmd_native_read();
        let mut chunks = ptr::null_mut();
        let mut error = ptr::null_mut();
        let status = unsafe {
            llama_cpp_sys_2::llama_rs_mtmd_input_chunks_init(&raw mut chunks, &raw mut error)
        };
        check_mtmd_native_status(status, error)?;
        Self::from_raw(chunks)
    }

    fn from_raw(
        chunks: *mut llama_cpp_sys_2::mtmd_input_chunks,
    ) -> Result<Self, MtmdInputChunksError> {
        let chunks = NonNull::new(chunks).ok_or(MtmdInputChunksError::NullResult)?;
        Ok(Self { chunks })
    }

    /// Get the number of chunks
    #[must_use]
    pub fn len(&self) -> usize {
        let _native_guard = mtmd_native_read();
        unsafe { llama_cpp_sys_2::mtmd_input_chunks_size(self.chunks.as_ptr()) }
    }

    /// Check if chunks collection is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get a chunk by index
    #[must_use]
    pub fn get(&self, index: usize) -> Option<MtmdInputChunk<'_>> {
        // Keep the size check and element lookup under one native-read guard. Acquiring a second
        // read lock while a writer is queued is not guaranteed to be reentrant on every platform.
        let _native_guard = mtmd_native_read();
        let length = unsafe { llama_cpp_sys_2::mtmd_input_chunks_size(self.chunks.as_ptr()) };
        if index >= length {
            return None;
        }

        let chunk_ptr =
            unsafe { llama_cpp_sys_2::mtmd_input_chunks_get(self.chunks.as_ptr(), index) };

        // Note: We don't own this chunk, it's owned by the chunks collection
        NonNull::new(chunk_ptr.cast_mut()).map(|ptr| MtmdInputChunk {
            chunk: ptr,
            borrowed_from: PhantomData,
        })
    }

    /// Copy the text tokens from every text chunk, omitting media embedding placeholders.
    ///
    /// # Errors
    ///
    /// Returns an error for a missing chunk, unsupported native type, or invalid token view.
    pub fn text_tokens(&self) -> Result<Vec<LlamaToken>, MtmdInputChunkError> {
        let mut tokens = Vec::new();
        for index in 0..self.len() {
            let chunk = self
                .get(index)
                .ok_or(MtmdInputChunkError::MissingChunk { index })?;
            if let Some(text) = chunk.text_tokens()? {
                tokens.extend_from_slice(text);
            }
        }
        Ok(tokens)
    }

    /// Return the type of the final chunk, when present.
    #[must_use]
    pub fn last_chunk_type(&self) -> Option<MtmdInputChunkType> {
        self.len()
            .checked_sub(1)
            .and_then(|index| self.get(index))
            .map(|chunk| chunk.chunk_type())
    }

    /// Get total number of tokens across all chunks.
    ///
    /// This is useful for keeping track of KV cache size.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid chunks or when the total token count overflows `usize`.
    pub fn total_tokens(&self) -> Result<usize, MtmdInputChunkError> {
        let mut total = 0usize;
        for index in 0..self.len() {
            let chunk = self
                .get(index)
                .ok_or(MtmdInputChunkError::MissingChunk { index })?;
            total = total
                .checked_add(chunk.n_tokens()?)
                .ok_or(MtmdInputChunkError::TokenCountOverflow)?;
        }
        Ok(total)
    }

    /// Get total position count across all chunks.
    ///
    /// This is useful to keep track of `n_past`. Normally `n_pos` equals `n_tokens`,
    /// but for M-RoPE it is different.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid chunks, negative positions, or total-count overflow.
    pub fn total_positions(&self) -> Result<i32, MtmdInputChunkError> {
        let mut total = 0i32;
        for index in 0..self.len() {
            let chunk = self
                .get(index)
                .ok_or(MtmdInputChunkError::MissingChunk { index })?;
            total = total
                .checked_add(chunk.n_positions()?)
                .ok_or(MtmdInputChunkError::PositionCountOverflow)?;
        }
        Ok(total)
    }

    /// Evaluate chunks using the multimodal context and LLAMA context.
    ///
    /// This helper function automatically:
    /// 1. Runs `llama_decode()` on text chunks
    /// 2. Runs `mtmd_encode()` on image chunks, then `mtmd_get_output_embd()` and then `llama_decode()`
    ///
    /// If any of the `mtmd_encode()` or `llama_decode()` calls return non-zero, the function
    /// stops and forwards the error.
    ///
    /// # Arguments
    ///
    /// * `mtmd_ctx` - The multimodal context
    /// * `llama_ctx` - The LLAMA context
    /// * `n_past` - Current position in the sequence
    /// * `seq_id` - Sequence ID for the batch
    /// * `n_batch` - Batch size for processing
    /// * `logits_last` - Whether to compute logits for the last token only
    ///
    /// # Returns
    ///
    /// Returns the new `n_past` value on success.
    ///
    /// # Errors
    ///
    /// Returns `MtmdEvalError::EvalFailure` if any encoding or decoding operation fails.
    ///
    /// This function is NOT thread-safe.
    pub fn eval_chunks(
        &self,
        mtmd_ctx: &mut MtmdContext<'_>,
        llama_ctx: &mut LlamaContext,
        n_past: i32,
        seq_id: i32,
        n_batch: i32,
        logits_last: bool,
    ) -> Result<i32, MtmdEvalError> {
        if n_batch <= 0 {
            return Err(MtmdEvalError::InvalidBatchSize(n_batch));
        }
        ensure_matching_model(
            std::ptr::from_ref(mtmd_ctx.model),
            std::ptr::from_ref(llama_ctx.model),
        )?;
        for index in 0..self.len() {
            let chunk = self
                .get(index)
                .ok_or(MtmdEvalError::MissingChunk { index })?;
            if let MtmdInputChunkType::Unknown(raw) = chunk.chunk_type() {
                return Err(MtmdEvalError::UnsupportedType { raw });
            }
        }
        let mut new_n_past = 0i32;

        let _native_guard = mtmd_native_read();
        let mut result = 0;
        let mut error = ptr::null_mut();
        let status = unsafe {
            llama_cpp_sys_2::llama_rs_mtmd_eval_chunks(
                mtmd_ctx.context.as_ptr(),
                llama_ctx.context.as_ptr(),
                self.chunks.as_ptr(),
                n_past,
                seq_id,
                n_batch,
                logits_last,
                &raw mut new_n_past,
                &raw mut result,
                &raw mut error,
            )
        };
        check_mtmd_native_status(status, error)?;

        if result == 0 {
            Ok(new_n_past)
        } else {
            Err(MtmdEvalError::EvalFailure(result))
        }
    }
}

fn ensure_matching_model(
    expected: *const LlamaModel,
    actual: *const LlamaModel,
) -> Result<(), MtmdEvalError> {
    if std::ptr::eq(expected, actual) {
        Ok(())
    } else {
        Err(MtmdEvalError::MismatchedModel)
    }
}

impl Drop for MtmdInputChunks {
    fn drop(&mut self) {
        let _native_guard = mtmd_native_read();
        unsafe { llama_cpp_sys_2::mtmd_input_chunks_free(self.chunks.as_ptr()) }
    }
}

/// Safe wrapper around `mtmd_input_chunk`.
///
/// Represents a single chunk of input data, which can be either text tokens,
/// image tokens, or audio tokens. The chunk type determines what kind of
/// data and operations are available.
#[derive(Debug)]
pub struct MtmdInputChunk<'chunks> {
    chunk: NonNull<llama_cpp_sys_2::mtmd_input_chunk>,
    borrowed_from: PhantomData<&'chunks MtmdInputChunks>,
}

impl MtmdInputChunk<'_> {
    /// Get the type of this chunk
    #[must_use]
    pub fn chunk_type(&self) -> MtmdInputChunkType {
        let _native_guard = mtmd_native_read();
        let chunk_type = unsafe { llama_cpp_sys_2::mtmd_input_chunk_get_type(self.chunk.as_ptr()) };
        MtmdInputChunkType::from_raw(chunk_type)
    }

    /// Get text tokens from this chunk.
    ///
    /// Only valid for text chunks. Returns `None` for image or audio chunks.
    ///
    /// # Returns
    ///
    /// Returns `Some(&[LlamaToken])` for text chunks, `None` otherwise.
    ///
    /// # Errors
    ///
    /// Returns an error for an unsupported native chunk type or invalid token view.
    pub fn text_tokens(&self) -> Result<Option<&[LlamaToken]>, MtmdInputChunkError> {
        match self.chunk_type() {
            MtmdInputChunkType::Text => {}
            MtmdInputChunkType::Image | MtmdInputChunkType::Audio => return Ok(None),
            MtmdInputChunkType::Unknown(raw) => {
                return Err(MtmdInputChunkError::UnsupportedType { raw });
            }
        }

        let _native_guard = mtmd_native_read();
        let mut n_tokens = 0usize;
        let tokens_ptr = unsafe {
            llama_cpp_sys_2::mtmd_input_chunk_get_tokens_text(
                self.chunk.as_ptr(),
                &raw mut n_tokens,
            )
        };

        // SAFETY: the token storage is owned by the parent chunks collection and remains stable
        // for this chunk borrow. The helper handles the empty/null convention before forming a
        // slice.
        unsafe { text_tokens_from_raw(tokens_ptr.cast::<LlamaToken>(), n_tokens).map(Some) }
    }

    /// Get the number of tokens in this chunk
    ///
    /// # Errors
    ///
    /// Returns an error when llama.cpp reports an unsupported chunk type.
    pub fn n_tokens(&self) -> Result<usize, MtmdInputChunkError> {
        if let MtmdInputChunkType::Unknown(raw) = self.chunk_type() {
            return Err(MtmdInputChunkError::UnsupportedType { raw });
        }
        let _native_guard = mtmd_native_read();
        Ok(unsafe { llama_cpp_sys_2::mtmd_input_chunk_get_n_tokens(self.chunk.as_ptr()) })
    }

    /// Get the number of positions in this chunk.
    ///
    /// Returns the number of temporal positions (always 1 for M-RoPE, `n_tokens` otherwise).
    ///
    /// # Errors
    ///
    /// Returns an error for an unsupported chunk type or negative native position count.
    pub fn n_positions(&self) -> Result<i32, MtmdInputChunkError> {
        if let MtmdInputChunkType::Unknown(raw) = self.chunk_type() {
            return Err(MtmdInputChunkError::UnsupportedType { raw });
        }
        let _native_guard = mtmd_native_read();
        let positions = unsafe { llama_cpp_sys_2::mtmd_input_chunk_get_n_pos(self.chunk.as_ptr()) };
        if positions < 0 {
            return Err(MtmdInputChunkError::InvalidPositionCount { value: positions });
        }
        Ok(positions)
    }

    /// Get chunk ID if available.
    ///
    /// Returns `None` for text chunks, may return an ID for image/audio chunks.
    ///
    /// # Errors
    ///
    /// Returns an error for an unsupported chunk type or an ID that is not valid UTF-8.
    pub fn id(&self) -> Result<Option<&str>, MtmdInputChunkError> {
        match self.chunk_type() {
            MtmdInputChunkType::Text => return Ok(None),
            MtmdInputChunkType::Image | MtmdInputChunkType::Audio => {}
            MtmdInputChunkType::Unknown(raw) => {
                return Err(MtmdInputChunkError::UnsupportedType { raw });
            }
        }
        let _native_guard = mtmd_native_read();
        let ptr = unsafe { llama_cpp_sys_2::mtmd_input_chunk_get_id(self.chunk.as_ptr()) };
        if ptr.is_null() {
            return Ok(None);
        }
        let id = unsafe { CStr::from_ptr(ptr) };
        if id.is_empty() {
            return Ok(None);
        }
        id.to_str()
            .map(Some)
            .map_err(MtmdInputChunkError::InvalidUtf8)
    }
}

/// Get the default media marker string.
///
/// Returns the default marker used to identify media positions in text
/// (typically `"<__media__>"`). This marker should be used in your input text
/// to indicate where media content should be inserted.
///
/// # Returns
///
/// Returns the default media marker as a string slice.
///
/// # Examples
///
/// ```
/// use llama_cpp_2::mtmd::mtmd_default_marker;
///
/// let marker = mtmd_default_marker();
/// assert!(!marker.is_empty());
///
/// let text = format!("Describe this image: {}", marker);
/// assert!(text.contains(marker));
/// ```
#[must_use]
pub fn mtmd_default_marker() -> &'static str {
    let _native_guard = mtmd_native_read();
    unsafe {
        let c_str = llama_cpp_sys_2::mtmd_default_marker();
        if c_str.is_null() {
            return "<__media__>";
        }
        CStr::from_ptr(c_str).to_str().unwrap_or("<__media__>")
    }
}

/// Projector modalities discoverable without initializing a full MTMD context.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MtmdFileCapabilities {
    /// The projector accepts vision input.
    pub vision: bool,
    /// The projector accepts audio input.
    pub audio: bool,
}

/// Worst-case native allocation estimate for one projector backend device.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MtmdDeviceMemoryEstimate {
    /// Index in llama.cpp's registered backend-device list, when the native device is registered.
    pub device_index: Option<usize>,
    /// Raw `ggml_backend_dev_type` value from the pinned llama.cpp revision.
    pub backend_type: i32,
    /// Backend device name.
    pub device_name: String,
    /// Backend device description.
    pub device_description: String,
    /// Worst-case projector allocation in bytes.
    pub bytes: u64,
}

/// Inspect projector modalities without loading the full projector.
///
/// Unlike llama.cpp's public `mtmd_get_cap_from_file`, this operation preserves native load and
/// parse errors rather than collapsing them into a pair of `false` values.
///
/// # Errors
///
/// Returns an error for an invalid path or when the projector cannot be loaded and inspected.
pub fn mtmd_capabilities_from_file(
    path: impl AsRef<Path>,
) -> Result<MtmdFileCapabilities, MtmdPreflightError> {
    let path = path_to_c_string(path.as_ref())?;
    let _native_guard = mtmd_native_read();
    let mut raw = MaybeUninit::<llama_cpp_sys_2::llama_rs_mtmd_capabilities>::uninit();
    let mut error = ptr::null_mut();
    let status = unsafe {
        llama_cpp_sys_2::llama_rs_mtmd_capabilities_from_file(
            path.as_ptr(),
            raw.as_mut_ptr(),
            &raw mut error,
        )
    };
    check_mtmd_status(status, error)?;
    let raw = unsafe { raw.assume_init() };
    Ok(MtmdFileCapabilities {
        vision: raw.vision,
        audio: raw.audio,
    })
}

/// Estimate the projector's worst-case native allocation per backend device.
///
/// This estimate does not describe current free memory and does not prove that allocation will
/// succeed. The pinned upstream function collapses its internal failures into an empty map, which
/// this wrapper reports as [`MtmdPreflightError::EmptyEstimate`].
///
/// # Errors
///
/// Returns an error for invalid inputs, native estimation failures, or an empty estimate.
pub fn mtmd_memory_usage(
    path: impl AsRef<Path>,
    params: &MtmdContextParams,
) -> Result<Vec<MtmdDeviceMemoryEstimate>, MtmdPreflightError> {
    let path = path_to_c_string(path.as_ref())?;
    let raw_params = params.to_raw()?;
    let native_guard = mtmd_native_write();
    let mut raw_report = ptr::null_mut();
    let mut error = ptr::null_mut();
    let status = unsafe {
        llama_cpp_sys_2::llama_rs_mtmd_memory_report_create(
            path.as_ptr(),
            &raw const raw_params,
            &raw mut raw_report,
            &raw mut error,
        )
    };
    check_mtmd_status(status, error)?;
    decode_mtmd_memory_report(raw_report, &native_guard)
}

fn decode_mtmd_memory_report(
    raw_report: *mut llama_cpp_sys_2::llama_rs_mtmd_memory_report,
    _native_guard: &RwLockWriteGuard<'static, ()>,
) -> Result<Vec<MtmdDeviceMemoryEstimate>, MtmdPreflightError> {
    let report = NativeMtmdMemoryReport(
        NonNull::new(raw_report).ok_or(MtmdPreflightError::NullOutput("memory report"))?,
    );
    let count = unsafe { llama_cpp_sys_2::llama_rs_mtmd_memory_report_count(report.0.as_ptr()) };
    if count == 0 {
        return Err(MtmdPreflightError::EmptyEstimate);
    }

    let mut result = Vec::new();
    result
        .try_reserve_exact(count)
        .map_err(MtmdPreflightError::RustAllocation)?;
    for index in 0..count {
        let mut raw_device = MaybeUninit::<llama_cpp_sys_2::llama_rs_mtmd_device_memory>::uninit();
        let mut error = ptr::null_mut();
        let status = unsafe {
            llama_cpp_sys_2::llama_rs_mtmd_memory_report_get(
                report.0.as_ptr(),
                index,
                raw_device.as_mut_ptr(),
                &raw mut error,
            )
        };
        check_mtmd_status(status, error)?;
        let raw_device = unsafe { raw_device.assume_init() };
        result.push(MtmdDeviceMemoryEstimate {
            device_index: raw_device
                .has_device_index
                .then_some(raw_device.device_index),
            backend_type: raw_device.backend_type,
            device_name: decode_mtmd_view(raw_device.device_name, "device name")?,
            device_description: decode_mtmd_view(
                raw_device.device_description,
                "device description",
            )?,
            bytes: raw_device.bytes,
        });
    }
    Ok(result)
}

#[derive(Debug)]
struct NativeMtmdMemoryReport(NonNull<llama_cpp_sys_2::llama_rs_mtmd_memory_report>);

impl Drop for NativeMtmdMemoryReport {
    fn drop(&mut self) {
        unsafe { llama_cpp_sys_2::llama_rs_mtmd_memory_report_free(self.0.as_ptr()) }
    }
}

fn path_to_c_string(path: &Path) -> Result<CString, MtmdPathError> {
    let utf8 = path
        .to_str()
        .ok_or_else(|| MtmdPathError::NonUtf8(path.to_path_buf()))?;
    CString::new(utf8).map_err(MtmdPathError::ContainsNul)
}

fn check_mtmd_native_status(
    status: llama_cpp_sys_2::llama_rs_status,
    error: *mut c_char,
) -> Result<(), MtmdNativeError> {
    if status == llama_cpp_sys_2::LLAMA_RS_STATUS_OK {
        if !error.is_null() {
            unsafe { llama_cpp_sys_2::llama_rs_string_free(error) };
        }
        return Ok(());
    }

    let message = take_mtmd_error(error);
    match status {
        llama_cpp_sys_2::LLAMA_RS_STATUS_INVALID_ARGUMENT => {
            Err(MtmdNativeError::InvalidArgument { message })
        }
        llama_cpp_sys_2::LLAMA_RS_STATUS_ALLOCATION_FAILED => {
            Err(MtmdNativeError::AllocationFailed { message })
        }
        llama_cpp_sys_2::LLAMA_RS_STATUS_EXCEPTION => {
            Err(MtmdNativeError::NativeException { message })
        }
        llama_cpp_sys_2::LLAMA_RS_STATUS_INVALID_UTF8 => {
            Err(MtmdNativeError::InvalidUtf8 { message })
        }
        llama_cpp_sys_2::LLAMA_RS_STATUS_INVALID_STATE => {
            Err(MtmdNativeError::InvalidState { message })
        }
        raw => Err(MtmdNativeError::UnexpectedStatus {
            status: raw,
            message,
        }),
    }
}

fn check_mtmd_status(
    status: llama_cpp_sys_2::llama_rs_status,
    error: *mut c_char,
) -> Result<(), MtmdPreflightError> {
    if status == llama_cpp_sys_2::LLAMA_RS_STATUS_OK {
        if !error.is_null() {
            unsafe { llama_cpp_sys_2::llama_rs_string_free(error) };
        }
        return Ok(());
    }

    let message = take_mtmd_error(error);
    match status {
        llama_cpp_sys_2::LLAMA_RS_STATUS_INVALID_ARGUMENT => {
            Err(MtmdPreflightError::InvalidArgument { message })
        }
        llama_cpp_sys_2::LLAMA_RS_STATUS_ALLOCATION_FAILED => {
            Err(MtmdPreflightError::AllocationFailed { message })
        }
        llama_cpp_sys_2::LLAMA_RS_STATUS_EXCEPTION => {
            Err(MtmdPreflightError::NativeException { message })
        }
        raw => Err(MtmdPreflightError::UnexpectedStatus {
            status: raw,
            message,
        }),
    }
}

fn take_mtmd_error(error: *mut c_char) -> String {
    if error.is_null() {
        return "native bridge returned no diagnostic".to_owned();
    }
    let message = unsafe { CStr::from_ptr(error).to_string_lossy().into_owned() };
    unsafe { llama_cpp_sys_2::llama_rs_string_free(error) };
    message
}

fn decode_mtmd_view(
    view: llama_cpp_sys_2::llama_rs_bytes_view,
    field: &'static str,
) -> Result<String, MtmdPreflightError> {
    let bytes = if view.len == 0 {
        &[]
    } else {
        if view.data.is_null() {
            return Err(MtmdPreflightError::NullByteView {
                field,
                length: view.len,
            });
        }
        unsafe { slice::from_raw_parts(view.data, view.len) }
    };
    std::str::from_utf8(bytes)
        .map(str::to_owned)
        .map_err(|source| MtmdPreflightError::InvalidUtf8 { field, source })
}

unsafe fn bitmap_data_from_raw<'bitmap>(
    data: *const u8,
    length: usize,
) -> Result<&'bitmap [u8], MtmdBitmapDataError> {
    if length == 0 {
        return Ok(&[]);
    }
    if data.is_null() {
        return Err(MtmdBitmapDataError::NullData { length });
    }
    Ok(unsafe { slice::from_raw_parts(data, length) })
}

unsafe fn text_tokens_from_raw<'chunk>(
    data: *const LlamaToken,
    length: usize,
) -> Result<&'chunk [LlamaToken], MtmdInputChunkError> {
    if length == 0 {
        return Ok(&[]);
    }
    if data.is_null() {
        return Err(MtmdInputChunkError::NullTokenData { length });
    }
    Ok(unsafe { slice::from_raw_parts(data, length) })
}

// Error types
/// An MTMD context option cannot be represented by the pinned native API.
#[derive(thiserror::Error, Debug, Eq, PartialEq)]
pub enum MtmdContextParamsError {
    /// The image token budget exceeds native `int`.
    #[error("{field} value {value} exceeds MTMD's i32 representation")]
    ImageTokenBudgetOverflow {
        /// Parameter name.
        field: &'static str,
        /// Requested token budget.
        value: u32,
    },
}

/// A C++ exception or bridge-contract error from an exception-safe mtmd operation.
#[derive(thiserror::Error, Debug)]
pub enum MtmdNativeError {
    /// The bridge rejected an invalid pointer, count, or representability constraint.
    #[error("invalid MTMD native argument: {message}")]
    InvalidArgument {
        /// Native diagnostic.
        message: String,
    },
    /// Native allocation failed before ownership could be transferred to Rust.
    #[error("MTMD native allocation failed: {message}")]
    AllocationFailed {
        /// Native diagnostic.
        message: String,
    },
    /// Native C++ code threw an exception.
    #[error("MTMD native operation failed: {message}")]
    NativeException {
        /// Native diagnostic.
        message: String,
    },
    /// Native output violated a UTF-8 contract.
    #[error("MTMD native operation returned invalid UTF-8: {message}")]
    InvalidUtf8 {
        /// Native diagnostic.
        message: String,
    },
    /// The native operation was rejected in its current state.
    #[error("MTMD native operation is invalid in the current state: {message}")]
    InvalidState {
        /// Native diagnostic.
        message: String,
    },
    /// The bridge returned a status not defined by the pinned contract.
    #[error("MTMD bridge returned unexpected status {status}: {message}")]
    UnexpectedStatus {
        /// Raw bridge status.
        status: i32,
        /// Native diagnostic.
        message: String,
    },
}

/// A filesystem path cannot be passed to mtmd's UTF-8, NUL-terminated path API.
#[derive(thiserror::Error, Debug)]
pub enum MtmdPathError {
    /// The path is not valid UTF-8 on this platform.
    #[error("MTMD path is not valid UTF-8: {}", .0.display())]
    NonUtf8(PathBuf),
    /// The path contains an interior NUL byte.
    #[error("MTMD path contains an interior NUL byte: {0}")]
    ContainsNul(#[source] std::ffi::NulError),
}

/// Errors that can occur when initializing MTMD context
#[derive(thiserror::Error, Debug)]
pub enum MtmdInitError {
    /// Context parameters cannot be represented by the native API.
    #[error(transparent)]
    Params(#[from] MtmdContextParamsError),
    /// A filesystem path could not be represented by the native API.
    #[error(transparent)]
    Path(#[from] MtmdPathError),
    /// The exception-safe native bridge rejected or failed the operation.
    #[error(transparent)]
    Native(#[from] MtmdNativeError),
    /// MTMD context initialization returned null
    #[error("MTMD context initialization returned null")]
    NullResult,
}

/// Errors returned when borrowing native bitmap bytes.
#[derive(thiserror::Error, Debug, Eq, PartialEq)]
pub enum MtmdBitmapDataError {
    /// Native code reported data bytes but returned no data pointer.
    #[error("MTMD bitmap returned a null data pointer for {length} bytes")]
    NullData {
        /// Native byte length.
        length: usize,
    },
}

/// Errors returned by projector capability and memory preflight.
#[derive(thiserror::Error, Debug)]
pub enum MtmdPreflightError {
    /// Context parameters cannot be represented by the native API.
    #[error(transparent)]
    Params(#[from] MtmdContextParamsError),
    /// A filesystem path could not be represented by the native API.
    #[error(transparent)]
    Path(#[from] MtmdPathError),
    /// The native bridge rejected its arguments.
    #[error("invalid MTMD preflight argument: {message}")]
    InvalidArgument {
        /// Native diagnostic.
        message: String,
    },
    /// Native allocation failed.
    #[error("MTMD preflight allocation failed: {message}")]
    AllocationFailed {
        /// Native diagnostic.
        message: String,
    },
    /// Native C++ code threw while inspecting the projector.
    #[error("MTMD preflight failed: {message}")]
    NativeException {
        /// Native diagnostic.
        message: String,
    },
    /// The bridge returned a status not defined by the pinned contract.
    #[error("MTMD bridge returned unexpected status {status}: {message}")]
    UnexpectedStatus {
        /// Raw bridge status.
        status: i32,
        /// Native diagnostic.
        message: String,
    },
    /// A successful native call did not return its required owner.
    #[error("MTMD bridge returned no {0}")]
    NullOutput(&'static str),
    /// Native code returned a null pointer for a nonempty byte view.
    #[error("MTMD bridge returned a null {field} pointer for {length} bytes")]
    NullByteView {
        /// Projected field.
        field: &'static str,
        /// Native byte length.
        length: usize,
    },
    /// A native textual result was not valid UTF-8.
    #[error("MTMD bridge returned invalid UTF-8 in {field}: {source}")]
    InvalidUtf8 {
        /// Projected field.
        field: &'static str,
        /// UTF-8 decoder failure.
        #[source]
        source: std::str::Utf8Error,
    },
    /// Rust could not reserve the owned result collection.
    #[error("failed to allocate the MTMD memory result: {0}")]
    RustAllocation(#[source] std::collections::TryReserveError),
    /// Upstream memory estimation returned no entries and suppressed its internal error.
    #[error("MTMD projector memory estimation returned no device entries")]
    EmptyEstimate,
}

/// Errors that can occur when working with MTMD bitmaps
#[derive(thiserror::Error, Debug)]
pub enum MtmdBitmapError {
    /// A filesystem path could not be represented by the native API.
    #[error(transparent)]
    Path(#[from] MtmdPathError),
    /// Invalid data size for bitmap
    #[error("Invalid data size for bitmap")]
    InvalidDataSize,
    /// Image dimensions cannot be represented as an RGB byte length.
    #[error("bitmap dimensions {nx}x{ny} overflow the addressable RGB byte length")]
    DimensionsOverflow {
        /// Image width.
        nx: u32,
        /// Image height.
        ny: u32,
    },
    /// The audio sample count cannot be represented by upstream mtmd's internal `uint32_t` field.
    #[error("audio sample count {length} exceeds MTMD's uint32 representation")]
    SampleCountOverflow {
        /// Number of provided samples.
        length: usize,
    },
    /// The exception-safe native bridge rejected or failed the operation.
    #[error(transparent)]
    Native(#[from] MtmdNativeError),
    /// Bitmap creation returned null
    #[error("Bitmap creation returned null")]
    NullResult,
}

/// Errors returned while updating a bitmap's optional cache identity.
#[derive(thiserror::Error, Debug)]
pub enum MtmdBitmapIdError {
    /// The ID contains an interior NUL and cannot be passed to the native C API.
    #[error("MTMD bitmap ID contains an interior NUL: {0}")]
    ContainsNul(#[from] std::ffi::NulError),
    /// A native bitmap ID was not valid UTF-8.
    #[error("MTMD bitmap ID is not valid UTF-8: {0}")]
    InvalidUtf8(#[source] std::str::Utf8Error),
    /// The exception-safe native bridge rejected or failed the operation.
    #[error(transparent)]
    Native(#[from] MtmdNativeError),
}

/// Errors that can occur when working with MTMD input chunks collections
#[derive(thiserror::Error, Debug)]
pub enum MtmdInputChunksError {
    /// The exception-safe native bridge rejected or failed the operation.
    #[error(transparent)]
    Native(#[from] MtmdNativeError),
    /// Input chunks creation returned null
    #[error("Input chunks creation returned null")]
    NullResult,
}

/// Errors that can occur when working with individual MTMD input chunks
#[derive(thiserror::Error, Debug)]
pub enum MtmdInputChunkError {
    /// The exception-safe native bridge rejected or failed the operation.
    #[error(transparent)]
    Native(#[from] MtmdNativeError),
    /// Input chunk operation returned null
    #[error("Input chunk operation returned null")]
    NullResult,
    /// The installed llama.cpp revision returned a chunk kind this wrapper does not understand.
    #[error("unsupported MTMD input chunk type: {raw}")]
    UnsupportedType {
        /// Raw native enum value.
        raw: i64,
    },
    /// A native chunk ID was not valid UTF-8.
    #[error("MTMD input chunk ID is not valid UTF-8: {0}")]
    InvalidUtf8(#[source] std::str::Utf8Error),
    /// Native collection metadata named an index but returned no chunk pointer.
    #[error("MTMD input chunk collection returned no chunk at in-range index {index}")]
    MissingChunk {
        /// In-range native index.
        index: usize,
    },
    /// Native code reported text tokens but returned no token pointer.
    #[error("MTMD text chunk returned a null token pointer for {length} tokens")]
    NullTokenData {
        /// Native token count.
        length: usize,
    },
    /// Native code returned a negative temporal-position count.
    #[error("MTMD input chunk returned an invalid negative position count: {value}")]
    InvalidPositionCount {
        /// Raw native position count.
        value: i32,
    },
    /// Summing the token counts of the collection overflowed `usize`.
    #[error("MTMD input chunk token count overflowed usize")]
    TokenCountOverflow,
    /// Summing the native position counts overflowed `i32`.
    #[error("MTMD input chunk position count overflowed i32")]
    PositionCountOverflow,
}

/// Errors that can occur during tokenization
#[derive(thiserror::Error, Debug)]
pub enum MtmdTokenizeError {
    /// The output chunk collection could not be allocated.
    #[error(transparent)]
    InputChunks(#[from] MtmdInputChunksError),
    /// Number of bitmaps does not match number of markers in text
    #[error("Number of bitmaps does not match number of markers")]
    BitmapCountMismatch,
    /// Image preprocessing error occurred
    #[error("Image preprocessing error")]
    ImagePreprocessingError,
    /// Text contains characters that cannot be converted to C string
    #[error("Failed to create CString from text: {0}")]
    CStringError(#[from] std::ffi::NulError),
    /// The exception-safe native bridge rejected or failed the operation.
    #[error(transparent)]
    Native(#[from] MtmdNativeError),
    /// Unknown error occurred during tokenization
    #[error("Unknown error: {0}")]
    UnknownError(i32),
}

/// Errors that can occur during encoding
#[derive(thiserror::Error, Debug)]
pub enum MtmdEncodeError {
    /// The exception-safe native bridge rejected or failed the operation.
    #[error(transparent)]
    Native(#[from] MtmdNativeError),
    /// The installed llama.cpp revision returned a chunk kind this wrapper does not understand.
    #[error("unsupported MTMD input chunk type: {raw}")]
    UnsupportedType {
        /// Raw native enum value.
        raw: i64,
    },
    /// Encode operation failed
    #[error("Encode failed with code: {0}")]
    EncodeFailure(i32),
}

/// Errors that can occur during evaluation
#[derive(thiserror::Error, Debug)]
pub enum MtmdEvalError {
    /// The exception-safe native bridge rejected or failed the operation.
    #[error(transparent)]
    Native(#[from] MtmdNativeError),
    /// The native helper requires a positive physical batch size.
    #[error("Eval requires a positive batch size, got: {0}")]
    InvalidBatchSize(i32),
    /// The projector and text context were initialized from different models.
    #[error("the MTMD projector and llama context belong to different text models")]
    MismatchedModel,
    /// Native collection metadata named an index but returned no chunk pointer.
    #[error("MTMD input chunk collection returned no chunk at in-range index {index}")]
    MissingChunk {
        /// In-range native index.
        index: usize,
    },
    /// The installed llama.cpp revision returned a chunk kind this wrapper cannot evaluate safely.
    #[error("unsupported MTMD input chunk type: {raw}")]
    UnsupportedType {
        /// Raw native enum value.
        raw: i64,
    },
    /// Evaluation operation failed
    #[error("Eval failed with code: {0}")]
    EvalFailure(i32),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_token_budget_narrowing_is_checked() {
        let params = MtmdContextParams {
            image_max_tokens: NonZeroU32::new(i32::MAX as u32 + 1),
            ..MtmdContextParams::default()
        };
        assert_eq!(
            params.to_raw().unwrap_err(),
            MtmdContextParamsError::ImageTokenBudgetOverflow {
                field: "image_max_tokens",
                value: i32::MAX as u32 + 1,
            }
        );
    }

    #[test]
    fn bitmap_data_handles_native_empty_and_null_conventions() {
        assert!(unsafe { bitmap_data_from_raw(ptr::null(), 0) }
            .unwrap()
            .is_empty());
        assert_eq!(
            unsafe { bitmap_data_from_raw(ptr::null(), 7) }.unwrap_err(),
            MtmdBitmapDataError::NullData { length: 7 }
        );
    }

    #[test]
    fn text_token_data_handles_native_empty_and_null_conventions() {
        assert!(unsafe { text_tokens_from_raw(ptr::null(), 0) }
            .unwrap()
            .is_empty());
        assert!(matches!(
            unsafe { text_tokens_from_raw(ptr::null(), 3) },
            Err(MtmdInputChunkError::NullTokenData { length: 3 })
        ));
    }

    #[test]
    fn byte_view_decoder_handles_empty_and_rejects_nonempty_null() {
        let empty = llama_cpp_sys_2::llama_rs_bytes_view {
            data: ptr::null(),
            len: 0,
        };
        assert_eq!(decode_mtmd_view(empty, "test").unwrap(), "");

        let invalid = llama_cpp_sys_2::llama_rs_bytes_view {
            data: ptr::null(),
            len: 2,
        };
        assert!(matches!(
            decode_mtmd_view(invalid, "test"),
            Err(MtmdPreflightError::NullByteView {
                field: "test",
                length: 2
            })
        ));
    }

    #[test]
    fn path_conversion_rejects_interior_nul() {
        let error = path_to_c_string(Path::new("bad\0path")).unwrap_err();
        assert!(matches!(error, MtmdPathError::ContainsNul(_)));
    }

    #[test]
    fn capability_bridge_rejects_null_arguments_with_owned_diagnostic() {
        let _native_guard = mtmd_native_read();
        let mut error = ptr::null_mut();
        let status = unsafe {
            llama_cpp_sys_2::llama_rs_mtmd_capabilities_from_file(
                ptr::null(),
                ptr::null_mut(),
                &raw mut error,
            )
        };
        let decoded = check_mtmd_status(status, error).unwrap_err();
        assert!(matches!(
            decoded,
            MtmdPreflightError::InvalidArgument { .. }
        ));
    }

    #[test]
    fn missing_projector_preserves_capability_error() {
        let error = mtmd_capabilities_from_file(Path::new("/__magnitude_missing__/projector.gguf"))
            .unwrap_err();
        assert!(matches!(error, MtmdPreflightError::NativeException { .. }));
    }

    #[test]
    fn missing_projector_memory_is_not_reported_as_zero_bytes() {
        let error = mtmd_memory_usage(
            Path::new("/__magnitude_missing__/projector.gguf"),
            &MtmdContextParams::default(),
        )
        .unwrap_err();
        assert!(matches!(error, MtmdPreflightError::EmptyEstimate));
    }

    #[test]
    fn audio_sample_count_narrowing_is_rejected_by_the_bridge() {
        if usize::BITS <= u32::BITS {
            return;
        }
        let _native_guard = mtmd_native_read();
        let mut bitmap = ptr::null_mut();
        let mut error = ptr::null_mut();
        let status = unsafe {
            llama_cpp_sys_2::llama_rs_mtmd_bitmap_init_audio(
                u32::MAX as usize + 1,
                ptr::null(),
                &raw mut bitmap,
                &raw mut error,
            )
        };
        let decoded = check_mtmd_native_status(status, error).unwrap_err();
        assert!(bitmap.is_null());
        assert!(matches!(decoded, MtmdNativeError::InvalidArgument { .. }));
    }
}
