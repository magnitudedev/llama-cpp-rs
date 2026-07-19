//! A safe wrapper around `llama_model_params`.

#[cfg(feature = "common")]
use crate::context::params::LlamaContextParams;
use crate::model::params::kv_overrides::KvOverrides;
use crate::LlamaCppError;
use std::ffi::{c_char, c_void, CStr};
use std::fmt::{Debug, Formatter};
use std::pin::Pin;
use std::ptr::null;

#[cfg(feature = "common")]
pub mod fit;
pub mod kv_overrides;

/// Result of [`LlamaModelParams::fit_params`], containing the fitted context size.
#[cfg(feature = "common")]
#[derive(Debug, Clone)]
pub struct FitResult {
    /// The context size after fitting (may have been reduced from the requested value).
    pub n_ctx: u32,
}

/// Error returned by [`LlamaModelParams::fit_params`].
#[cfg(feature = "common")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum FitError {
    /// The margin vector is shorter than `llama_max_devices()`.
    #[error("fit margins contain {provided} entries but at least {required} are required")]
    InvalidMargins {
        /// Entries supplied by the caller.
        provided: usize,
        /// Entries required by the native build.
        required: usize,
    },
    /// Could not find allocations that are projected to fit available memory.
    #[error("could not find allocations that fit available memory")]
    Failure,
    /// A hard error occurred during fitting (e.g. model not found at the specified path).
    #[error("hard error during parameter fitting")]
    Error,
}

#[allow(clippy::cast_possible_wrap)]
#[allow(clippy::cast_possible_truncation)]
const LLAMA_SPLIT_MODE_NONE: i8 = llama_cpp_sys_2::LLAMA_SPLIT_MODE_NONE as i8;
#[allow(clippy::cast_possible_wrap)]
#[allow(clippy::cast_possible_truncation)]
const LLAMA_SPLIT_MODE_LAYER: i8 = llama_cpp_sys_2::LLAMA_SPLIT_MODE_LAYER as i8;
#[allow(clippy::cast_possible_wrap)]
#[allow(clippy::cast_possible_truncation)]
const LLAMA_SPLIT_MODE_ROW: i8 = llama_cpp_sys_2::LLAMA_SPLIT_MODE_ROW as i8;
#[allow(clippy::cast_possible_wrap)]
#[allow(clippy::cast_possible_truncation)]
const LLAMA_SPLIT_MODE_TENSOR: i8 = llama_cpp_sys_2::LLAMA_SPLIT_MODE_TENSOR as i8;

/// A rusty wrapper around `llama_split_mode`.
#[repr(i8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LlamaSplitMode {
    /// Single GPU
    None = LLAMA_SPLIT_MODE_NONE,
    /// Split layers and KV across GPUs
    Layer = LLAMA_SPLIT_MODE_LAYER,
    /// Split layers and KV across GPUs, use tensor parallelism if supported
    Row = LLAMA_SPLIT_MODE_ROW,
    /// Experimental tensor parallelism across GPUs
    Tensor = LLAMA_SPLIT_MODE_TENSOR,
}

/// Typed GPU-layer selection understood by llama.cpp model loading.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LlamaGpuLayers {
    /// Preserve llama.cpp's native `-1` setting.
    ///
    /// Callers seeking `llama-server` auto-placement must run `common/fit` before loading; bare
    /// libllama interprets this value as offloading all available layers.
    Auto,
    /// Offload every supported layer (`-2`).
    All,
    /// Offload an explicit number of layers.
    Count(u32),
}

impl LlamaGpuLayers {
    /// Decode llama.cpp's signed sentinel representation.
    #[must_use]
    pub fn from_raw(value: i32) -> Self {
        match value {
            -1 => Self::Auto,
            value if value < -1 => Self::All,
            value => Self::Count(value.cast_unsigned()),
        }
    }

    /// Encode this selection using llama.cpp's signed sentinel representation.
    #[must_use]
    pub fn as_raw(self) -> i32 {
        match self {
            Self::Auto => -1,
            Self::All => -2,
            Self::Count(value) => i32::try_from(value).unwrap_or(i32::MAX),
        }
    }
}

/// An error that occurs when unknown split mode is encountered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LlamaSplitModeParseError(pub i32);

/// Create a `LlamaSplitMode` from a `i32`.
///
/// # Errors
/// Returns `LlamaSplitModeParseError` if the value does not correspond to a valid `LlamaSplitMode`.
impl TryFrom<i32> for LlamaSplitMode {
    type Error = LlamaSplitModeParseError;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        let i8_value = value
            .try_into()
            .map_err(|_| LlamaSplitModeParseError(value))?;
        match i8_value {
            LLAMA_SPLIT_MODE_NONE => Ok(Self::None),
            LLAMA_SPLIT_MODE_LAYER => Ok(Self::Layer),
            LLAMA_SPLIT_MODE_ROW => Ok(Self::Row),
            LLAMA_SPLIT_MODE_TENSOR => Ok(Self::Tensor),
            _ => Err(LlamaSplitModeParseError(value)),
        }
    }
}

/// Create a `LlamaSplitMode` from a `u32`.
///
/// # Errors
/// Returns `LlamaSplitModeParseError` if the value does not correspond to a valid `LlamaSplitMode`.
impl TryFrom<u32> for LlamaSplitMode {
    type Error = LlamaSplitModeParseError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        let i8_value = value
            .try_into()
            .map_err(|_| LlamaSplitModeParseError(value.try_into().unwrap_or(i32::MAX)))?;
        match i8_value {
            LLAMA_SPLIT_MODE_NONE => Ok(Self::None),
            LLAMA_SPLIT_MODE_LAYER => Ok(Self::Layer),
            LLAMA_SPLIT_MODE_ROW => Ok(Self::Row),
            LLAMA_SPLIT_MODE_TENSOR => Ok(Self::Tensor),
            _ => Err(LlamaSplitModeParseError(
                value.try_into().unwrap_or(i32::MAX),
            )),
        }
    }
}

/// Create a `i32` from a `LlamaSplitMode`.
impl From<LlamaSplitMode> for i32 {
    fn from(value: LlamaSplitMode) -> Self {
        match value {
            LlamaSplitMode::None => LLAMA_SPLIT_MODE_NONE.into(),
            LlamaSplitMode::Layer => LLAMA_SPLIT_MODE_LAYER.into(),
            LlamaSplitMode::Row => LLAMA_SPLIT_MODE_ROW.into(),
            LlamaSplitMode::Tensor => LLAMA_SPLIT_MODE_TENSOR.into(),
        }
    }
}

/// Create a `u32` from a `LlamaSplitMode`.
impl From<LlamaSplitMode> for u32 {
    fn from(value: LlamaSplitMode) -> Self {
        match value {
            LlamaSplitMode::None => LLAMA_SPLIT_MODE_NONE as u32,
            LlamaSplitMode::Layer => LLAMA_SPLIT_MODE_LAYER as u32,
            LlamaSplitMode::Row => LLAMA_SPLIT_MODE_ROW as u32,
            LlamaSplitMode::Tensor => LLAMA_SPLIT_MODE_TENSOR as u32,
        }
    }
}

/// The default split mode is `Layer` in llama.cpp.
impl Default for LlamaSplitMode {
    fn default() -> Self {
        LlamaSplitMode::Layer
    }
}

/// The maximum number of devices supported.
///
/// The real maximum number of devices is the lesser one of this value and the value returned by
/// `llama_cpp_2::max_devices()`.
pub const LLAMA_CPP_MAX_DEVICES: usize = 16;

/// Invalid per-device model tensor split weights.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum TensorSplitError {
    /// More entries were supplied than llama.cpp can consume.
    #[error("tensor split has {provided} entries but llama.cpp supports at most {maximum}")]
    TooMany {
        /// Number of supplied weights.
        provided: usize,
        /// Maximum native device count.
        maximum: usize,
    },
    /// A weight was negative or non-finite.
    #[error("tensor split weight {index} must be finite and non-negative, received {value}")]
    InvalidWeight {
        /// Index of the invalid weight.
        index: usize,
        /// Invalid value.
        value: f32,
    },
    /// An explicit split must assign a non-zero proportion to a device.
    #[error("tensor split must contain at least one positive weight")]
    AllZero,
}

/// A safe wrapper around `llama_model_params`.
#[allow(clippy::module_name_repetitions)]
pub struct LlamaModelParams {
    pub(crate) params: llama_cpp_sys_2::llama_model_params,
    kv_overrides: Vec<llama_cpp_sys_2::llama_model_kv_override>,
    buft_overrides: Vec<llama_cpp_sys_2::llama_model_tensor_buft_override>,
    devices: Pin<Box<[llama_cpp_sys_2::ggml_backend_dev_t; LLAMA_CPP_MAX_DEVICES]>>,
    tensor_split: Vec<f32>,
    progress_callback: Option<Box<dyn FnMut(f32) -> bool>>,
}

#[allow(clippy::missing_fields_in_debug)] // Callback closures and pointer backing stores are opaque.
impl Debug for LlamaModelParams {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaModelParams")
            .field("n_gpu_layers", &self.params.n_gpu_layers)
            .field("main_gpu", &self.params.main_gpu)
            .field("vocab_only", &self.params.vocab_only)
            .field("use_mmap", &self.params.use_mmap)
            .field("use_mlock", &self.params.use_mlock)
            .field("split_mode", &self.split_mode())
            .field("devices", &self.devices)
            .field("kv_overrides", &"vec of kv_overrides")
            .finish()
    }
}

impl LlamaModelParams {
    /// See [`KvOverrides`]
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use llama_cpp_2::model::params::LlamaModelParams;
    /// let params = Box::pin(LlamaModelParams::default());
    /// let kv_overrides = params.kv_overrides();
    /// let count = kv_overrides.into_iter().count();
    /// assert_eq!(count, 0);
    /// ```
    #[must_use]
    pub fn kv_overrides(&self) -> KvOverrides<'_> {
        KvOverrides::new(self)
    }

    /// Appends a key-value override to the model parameters. It must be pinned as this creates a self-referential struct.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use std::ffi::{CStr, CString};
    /// use std::pin::pin;
    /// # use llama_cpp_2::model::params::LlamaModelParams;
    /// # use llama_cpp_2::model::params::kv_overrides::ParamOverrideValue;
    /// let mut params = pin!(LlamaModelParams::default());
    /// let key = CString::new("key").expect("CString::new failed");
    /// params.as_mut().append_kv_override(&key, ParamOverrideValue::Int(50));
    ///
    /// let kv_overrides = params.kv_overrides().into_iter().collect::<Vec<_>>();
    /// assert_eq!(kv_overrides.len(), 1);
    ///
    /// let (k, v) = &kv_overrides[0];
    /// assert_eq!(v, &ParamOverrideValue::Int(50));
    ///
    /// assert_eq!(k.to_bytes(), b"key", "expected key to be 'key', was {:?}", k);
    /// ```
    #[allow(clippy::missing_panics_doc)] // panics are just to enforce internal invariants, not user errors
    pub fn append_kv_override(
        mut self: Pin<&mut Self>,
        key: &CStr,
        value: kv_overrides::ParamOverrideValue,
    ) {
        let kv_override = self
            .kv_overrides
            .get_mut(0)
            .expect("kv_overrides did not have a next allocated");

        assert_eq!(kv_override.key[0], 0, "last kv_override was not empty");

        // There should be some way to do this without iterating over everything.
        for (i, &c) in key.to_bytes_with_nul().iter().enumerate() {
            kv_override.key[i] = c_char::try_from(c).expect("invalid character in key");
        }

        kv_override.tag = value.tag();
        kv_override.__bindgen_anon_1 = value.value();

        // set to null pointer for panic safety (as push may move the vector, invalidating the pointer)
        self.params.kv_overrides = null();

        // push the next one to ensure we maintain the iterator invariant of ending with a 0
        self.kv_overrides
            .push(llama_cpp_sys_2::llama_model_kv_override {
                key: [0; 128],
                tag: 0,
                __bindgen_anon_1: llama_cpp_sys_2::llama_model_kv_override__bindgen_ty_1 {
                    val_i64: 0,
                },
            });

        // set the pointer to the (potentially) new vector
        self.params.kv_overrides = self.kv_overrides.as_ptr();

        eprintln!("saved ptr: {:?}", self.params.kv_overrides);
    }
}

impl LlamaModelParams {
    /// Adds buffer type overides to move all mixture-of-experts layers to CPU.
    pub fn add_cpu_moe_override(self: Pin<&mut Self>) {
        self.add_cpu_buft_override(c"\\.ffn_(up|down|gate)_(ch|)exps");
    }

    /// Appends a buffer type override to the model parameters, to move layers matching pattern to CPU.
    /// It must be pinned as this creates a self-referential struct.
    ///
    /// # Panics
    ///
    /// Panics if the sentinel entry is missing, the pattern is not representable as native chars,
    /// or the previous sentinel has already been populated unexpectedly.
    pub fn add_cpu_buft_override(mut self: Pin<&mut Self>, key: &CStr) {
        let buft_override = self
            .buft_overrides
            .get_mut(0)
            .expect("buft_overrides did not have a next allocated");

        assert!(
            buft_override.pattern.is_null(),
            "last buft_override was not empty"
        );

        // There should be some way to do this without iterating over everything.
        for &c in key.to_bytes_with_nul() {
            c_char::try_from(c).expect("invalid character in key");
        }

        buft_override.pattern = key.as_ptr();
        buft_override.buft = unsafe { llama_cpp_sys_2::ggml_backend_cpu_buffer_type() };

        // set to null pointer for panic safety (as push may move the vector, invalidating the pointer)
        self.params.tensor_buft_overrides = null();

        // push the next one to ensure we maintain the iterator invariant of ending with a 0
        self.buft_overrides
            .push(llama_cpp_sys_2::llama_model_tensor_buft_override {
                pattern: std::ptr::null(),
                buft: std::ptr::null_mut(),
            });

        // set the pointer to the (potentially) new vector
        self.params.tensor_buft_overrides = self.buft_overrides.as_ptr();
    }
}

#[cfg(feature = "common")]
impl LlamaModelParams {
    /// Automatically fit model parameters to available device memory.
    ///
    /// Wraps llama.cpp's `common_fit_params` (libcommon), which determines optimal `n_gpu_layers`,
    /// `tensor_split`, and `tensor_buft_overrides` based on available VRAM. On success
    /// the model and context params are updated in place.
    ///
    /// # Requirements
    ///
    /// Per the C API docstring, only parameters that still hold their default value
    /// are modified. In practice this means:
    /// - `n_gpu_layers` must be at its default (`-1`). Do not call
    ///   [`with_n_gpu_layers`](Self::with_n_gpu_layers) before this.
    /// - No `tensor_buft_overrides` may be set. Do not call
    ///   [`add_cpu_buft_override`](Self::add_cpu_buft_override) or
    ///   [`add_cpu_moe_override`](Self::add_cpu_moe_override) before this.
    /// - `cparams.n_ctx` is only auto-selected if it is `0`; otherwise it is left alone.
    ///
    /// # Arguments
    ///
    /// - `model_path` — path to the GGUF model file.
    /// - `cparams` — context parameters; `n_ctx` may be modified (see above).
    /// - `margins` — memory margin per device in bytes. Must have at least
    ///   `llama_max_devices()` elements.
    /// - `n_ctx_min` — minimum context size to preserve when reducing memory usage.
    /// - `log_level` — minimum log level for fitting output; lower levels are routed to the debug
    ///   log.
    ///
    /// # Concurrency
    ///
    /// The upstream fit implementation temporarily replaces llama.cpp's process-global logger
    /// with a callback backed by call-local state. Concurrent fit calls are therefore unsupported.
    /// This binding intentionally uses the pinned upstream implementation unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`FitError::Failure`] if no fitting allocation could be found, or
    /// [`FitError::Error`] on a hard error (e.g. the model file could not be read).
    pub fn fit_params(
        mut self: Pin<&mut Self>,
        model_path: &CStr,
        cparams: &mut LlamaContextParams,
        margins: &mut [usize],
        n_ctx_min: u32,
        log_level: llama_cpp_sys_2::ggml_log_level,
    ) -> Result<FitResult, FitError> {
        let _logger_guard = crate::log::lock_native_logger();
        let max_devices = unsafe { llama_cpp_sys_2::llama_max_devices() };
        if margins.len() < max_devices {
            return Err(FitError::InvalidMargins {
                provided: margins.len(),
                required: max_devices,
            });
        }
        let max_buft = unsafe { llama_cpp_sys_2::llama_max_tensor_buft_overrides() };

        // Allocate tensor_split output buffer.
        self.tensor_split.clear();
        self.tensor_split.resize(max_devices, 0.0);

        // Reset and resize buft_overrides for fit output (null-terminated).
        self.buft_overrides.clear();
        self.buft_overrides.resize(
            max_buft + 1,
            llama_cpp_sys_2::llama_model_tensor_buft_override {
                pattern: std::ptr::null(),
                buft: std::ptr::null_mut(),
            },
        );

        // Clear pointers before the call — fit writes directly into the buffers above.
        self.params.tensor_split = null::<f32>();
        self.params.tensor_buft_overrides = null();

        let status = unsafe {
            llama_cpp_sys_2::llama_rs_fit_params(
                model_path.as_ptr(),
                &raw mut self.params,
                &raw mut cparams.context_params,
                self.tensor_split.as_mut_ptr(),
                self.buft_overrides.as_mut_ptr(),
                margins.as_mut_ptr(),
                n_ctx_min,
                log_level,
            )
        };

        // The native fit path may point the raw params at these buffers even
        // when it reports failure, so restore their stable owned addresses
        // before any early return.
        self.params.tensor_split = self.tensor_split.as_ptr();
        self.params.tensor_buft_overrides = self.buft_overrides.as_ptr();

        // llama_rs_fit_params returns common_params_fit_status: 0 = success, 1 = failure, 2 = error.
        match status {
            0 => {}
            1 => return Err(FitError::Failure),
            _ => return Err(FitError::Error),
        }

        Ok(FitResult {
            n_ctx: cparams.context_params.n_ctx,
        })
    }
}

impl LlamaModelParams {
    /// Get the number of layers to offload to the GPU.
    #[must_use]
    pub fn n_gpu_layers(&self) -> i32 {
        self.params.n_gpu_layers
    }

    /// Return the typed native GPU-layer setting.
    #[must_use]
    pub fn gpu_layers(&self) -> LlamaGpuLayers {
        LlamaGpuLayers::from_raw(self.params.n_gpu_layers)
    }

    /// The GPU that is used for scratch and small tensors
    #[must_use]
    pub fn main_gpu(&self) -> i32 {
        self.params.main_gpu
    }

    /// only load the vocabulary, no weights
    #[must_use]
    pub fn vocab_only(&self) -> bool {
        self.params.vocab_only
    }

    /// use mmap if possible
    #[must_use]
    pub fn use_mmap(&self) -> bool {
        self.params.use_mmap
    }

    /// force system to keep model in RAM
    #[must_use]
    pub fn use_mlock(&self) -> bool {
        self.params.use_mlock
    }

    /// get the split mode
    ///
    /// # Errors
    /// Returns `LlamaSplitModeParseError` if the unknown split mode is encountered.
    pub fn split_mode(&self) -> Result<LlamaSplitMode, LlamaSplitModeParseError> {
        LlamaSplitMode::try_from(self.params.split_mode)
    }

    /// get the devices
    #[must_use]
    pub fn devices(&self) -> Vec<usize> {
        let mut backend_devices = Vec::new();
        for i in 0..unsafe { llama_cpp_sys_2::ggml_backend_dev_count() } {
            let dev = unsafe { llama_cpp_sys_2::ggml_backend_dev_get(i) };
            backend_devices.push(dev);
        }
        let mut devices = Vec::new();
        for &dev in self.devices.iter() {
            if dev.is_null() {
                break;
            }
            if let Some((index, _)) = backend_devices
                .iter()
                .enumerate()
                .find(|&(_i, &d)| d == dev)
            {
                devices.push(index);
            }
        }
        devices
    }

    /// sets the number of gpu layers to offload to the GPU.
    /// ```
    /// # use llama_cpp_2::model::params::LlamaModelParams;
    /// let params = LlamaModelParams::default();
    /// let params = params.with_n_gpu_layers(1);
    /// assert_eq!(params.n_gpu_layers(), 1);
    /// ```
    #[must_use]
    pub fn with_n_gpu_layers(mut self, n_gpu_layers: u32) -> Self {
        // The only way this conversion can fail is if u32 overflows the i32 - in which case we set
        // to MAX
        let n_gpu_layers = i32::try_from(n_gpu_layers).unwrap_or(i32::MAX);
        self.params.n_gpu_layers = n_gpu_layers;
        self
    }

    /// Set auto, all, or an explicit GPU-layer count without exposing native sentinels.
    #[must_use]
    pub fn with_gpu_layers(mut self, gpu_layers: LlamaGpuLayers) -> Self {
        self.params.n_gpu_layers = gpu_layers.as_raw();
        self
    }

    /// sets the main GPU
    ///
    /// To enable this option, you must set `split_mode` to `LlamaSplitMode::None` to enable single GPU mode.
    #[must_use]
    pub fn with_main_gpu(mut self, main_gpu: i32) -> Self {
        self.params.main_gpu = main_gpu;
        self
    }

    /// sets `vocab_only`
    #[must_use]
    pub fn with_vocab_only(mut self, vocab_only: bool) -> Self {
        self.params.vocab_only = vocab_only;
        self
    }

    /// sets `use_mmap`
    #[must_use]
    pub fn with_use_mmap(mut self, use_mmap: bool) -> Self {
        self.params.use_mmap = use_mmap;
        self
    }

    /// sets `use_mlock`
    #[must_use]
    pub fn with_use_mlock(mut self, use_mlock: bool) -> Self {
        self.params.use_mlock = use_mlock;
        self
    }

    /// sets `split_mode`
    #[must_use]
    pub fn with_split_mode(mut self, split_mode: LlamaSplitMode) -> Self {
        self.params.split_mode = split_mode.into();
        self
    }

    /// Set model-offload proportions for accelerator devices.
    ///
    /// The owned buffer is padded to llama.cpp's full device count because the
    /// native parameter does not carry an independent slice length. Passing an
    /// empty slice clears an earlier explicit split and restores native
    /// automatic placement.
    ///
    /// # Errors
    ///
    /// Returns [`TensorSplitError`] for too many, negative, non-finite, or
    /// entirely zero explicit weights.
    pub fn with_tensor_split(mut self, weights: &[f32]) -> Result<Self, TensorSplitError> {
        let maximum = crate::max_devices().min(LLAMA_CPP_MAX_DEVICES);
        if weights.len() > maximum {
            return Err(TensorSplitError::TooMany {
                provided: weights.len(),
                maximum,
            });
        }
        for (index, value) in weights.iter().copied().enumerate() {
            if !value.is_finite() || value < 0.0 {
                return Err(TensorSplitError::InvalidWeight { index, value });
            }
        }
        if !weights.is_empty() && !weights.iter().any(|value| *value > 0.0) {
            return Err(TensorSplitError::AllZero);
        }

        self.tensor_split.clear();
        if weights.is_empty() {
            self.params.tensor_split = null();
        } else {
            self.tensor_split.resize(maximum, 0.0);
            self.tensor_split[..weights.len()].copy_from_slice(weights);
            self.params.tensor_split = self.tensor_split.as_ptr();
        }
        Ok(self)
    }

    /// Return the owned full-width native tensor split, or an empty slice when
    /// automatic placement is active.
    #[must_use]
    pub fn tensor_split(&self) -> &[f32] {
        &self.tensor_split
    }

    /// sets `devices`
    ///
    /// The devices are specified as indices that correspond to the ggml backend device indices.
    ///
    /// The maximum number of devices is 16.
    ///
    /// You don't need to specify CPU or ACCEL devices.
    ///
    /// # Errors
    /// Returns `LlamaCppError::BackendDeviceNotFound` if any device index is invalid.
    pub fn with_devices(mut self, devices: &[usize]) -> Result<Self, LlamaCppError> {
        for dev in self.devices.iter_mut() {
            *dev = std::ptr::null_mut();
        }
        // Check device count
        let max_devices = crate::max_devices().min(LLAMA_CPP_MAX_DEVICES);
        if devices.len() > max_devices {
            return Err(LlamaCppError::MaxDevicesExceeded(max_devices));
        }
        for (i, &dev) in devices.iter().enumerate() {
            if dev >= unsafe { llama_cpp_sys_2::ggml_backend_dev_count() } {
                return Err(LlamaCppError::BackendDeviceNotFound(dev));
            }
            let backend_dev = unsafe { llama_cpp_sys_2::ggml_backend_dev_get(dev) };
            self.devices[i] = backend_dev;
        }
        if self.devices.is_empty() {
            self.params.devices = std::ptr::null_mut();
        } else {
            self.params.devices = self.devices.as_mut_ptr();
        }
        Ok(self)
    }

    /// Set `no_alloc`
    ///
    /// If this parameter is true, don't allocate memory for the tensor data
    ///
    /// You can't use `no_alloc` with `use_mmap`, so this also sets `use_mmap` to false.
    #[must_use]
    pub fn with_no_alloc(mut self, no_alloc: bool) -> Self {
        self.params.no_alloc = no_alloc;
        if no_alloc {
            self = self.with_use_mmap(false);
        }
        self
    }

    /// Get `no_alloc`
    ///
    /// If this parameter is true, don't allocate memory for the tensor data
    #[must_use]
    pub fn no_alloc(&self) -> bool {
        self.params.no_alloc
    }

    /// Sets a callback invoked during loading with progress in `0.0..=1.0`.
    /// Returning `false` aborts the load (it then fails with `NullResult`).
    #[must_use]
    pub fn with_progress_callback<F: FnMut(f32) -> bool + 'static>(mut self, callback: F) -> Self {
        unsafe extern "C" fn trampoline<F: FnMut(f32) -> bool>(
            progress: f32,
            user_data: *mut c_void,
        ) -> bool {
            let callback = unsafe { &mut *user_data.cast::<F>() };
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| callback(progress)))
                .unwrap_or(false)
        }

        let mut callback = Box::new(callback);
        self.params.progress_callback_user_data =
            std::ptr::from_mut(&mut *callback).cast::<c_void>();
        self.params.progress_callback = Some(trampoline::<F>);
        self.progress_callback = Some(callback);
        self
    }
}

/// Default parameters for `LlamaModel`. (as defined in llama.cpp by `llama_model_default_params`)
/// ```
/// # use llama_cpp_2::model::params::LlamaModelParams;
/// use llama_cpp_2::model::params::LlamaSplitMode;
/// let params = LlamaModelParams::default();
/// assert_eq!(params.n_gpu_layers(), -1, "n_gpu_layers should be -1");
/// assert_eq!(params.main_gpu(), 0, "main_gpu should be 0");
/// assert_eq!(params.vocab_only(), false, "vocab_only should be false");
/// assert_eq!(params.use_mmap(), true, "use_mmap should be true");
/// assert_eq!(params.use_mlock(), false, "use_mlock should be false");
/// assert_eq!(params.split_mode(), Ok(LlamaSplitMode::Layer), "split_mode should be LAYER");
/// assert_eq!(params.devices().len(), 0, "devices should be empty");
/// assert_eq!(params.no_alloc(), false, "no_alloc should be false");
/// ```
impl Default for LlamaModelParams {
    fn default() -> Self {
        let default_params = unsafe { llama_cpp_sys_2::llama_model_default_params() };
        LlamaModelParams {
            params: default_params,
            // push the next one to ensure we maintain the iterator invariant of ending with a 0
            kv_overrides: vec![llama_cpp_sys_2::llama_model_kv_override {
                key: [0; 128],
                tag: 0,
                __bindgen_anon_1: llama_cpp_sys_2::llama_model_kv_override__bindgen_ty_1 {
                    val_i64: 0,
                },
            }],
            buft_overrides: vec![llama_cpp_sys_2::llama_model_tensor_buft_override {
                pattern: std::ptr::null(),
                buft: std::ptr::null_mut(),
            }],
            devices: Box::pin([std::ptr::null_mut(); 16]),
            tensor_split: Vec::new(),
            progress_callback: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{LlamaModelParams, TensorSplitError};

    #[test]
    fn tensor_split_owns_and_pads_native_weights() {
        let params = LlamaModelParams::default()
            .with_tensor_split(&[3.0, 1.0])
            .expect("valid split");
        assert_eq!(&params.tensor_split()[..2], &[3.0, 1.0]);
        assert!(params.tensor_split()[2..].iter().all(|value| *value == 0.0));
        assert_eq!(params.params.tensor_split, params.tensor_split().as_ptr());

        let params = params.with_tensor_split(&[]).expect("clear split");
        assert!(params.tensor_split().is_empty());
        assert!(params.params.tensor_split.is_null());
    }

    #[test]
    fn tensor_split_rejects_invalid_weights() {
        assert!(matches!(
            LlamaModelParams::default().with_tensor_split(&[0.0, 0.0]),
            Err(TensorSplitError::AllZero)
        ));
        assert!(matches!(
            LlamaModelParams::default().with_tensor_split(&[f32::NAN]),
            Err(TensorSplitError::InvalidWeight { index: 0, .. })
        ));
    }

    #[test]
    fn progress_callback_round_trips_and_can_abort() {
        use std::cell::Cell;
        use std::rc::Rc;

        let calls = Rc::new(Cell::new(0_u32));
        let counter = Rc::clone(&calls);
        let params = LlamaModelParams::default().with_progress_callback(move |_progress| {
            counter.set(counter.get() + 1);
            false
        });

        assert!(params.params.progress_callback.is_some());
        assert!(!params.params.progress_callback_user_data.is_null());

        let trampoline = params.params.progress_callback.unwrap();
        let user_data = params.params.progress_callback_user_data;
        let first = unsafe { trampoline(0.5, user_data) };
        let second = unsafe { trampoline(1.0, user_data) };

        assert!(!first && !second, "returning false signals an abort");
        assert_eq!(calls.get(), 2);
    }

    #[test]
    fn progress_callback_contains_panics_at_the_ffi_boundary() {
        let params = LlamaModelParams::default().with_progress_callback(|_progress| {
            panic!("callback panic must not cross the native boundary");
        });

        let trampoline = params.params.progress_callback.unwrap();
        let user_data = params.params.progress_callback_user_data;

        assert!(!unsafe { trampoline(0.5, user_data) });
    }
}
