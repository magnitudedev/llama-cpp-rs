//! Typed diagnostics for llama.cpp's `common/fit` estimator.

use std::ffi::{c_char, CStr};
use std::mem::MaybeUninit;
use std::pin::Pin;
use std::ptr::{self, NonNull};

use crate::context::params::LlamaContextParams;
use crate::model::params::LlamaModelParams;

use llama_cpp_sys_2 as sys;

/// Stable identity of the native model-free ggml calibration procedure.
pub const FIT_CALIBRATION_METHOD: &str = "llama-native-ggml-decode-calibration-v2";

/// Stable identity of the native decode-workload projection.
pub const FIT_DECODE_WORKLOAD_METHOD: &str = "llama-native-decode-workload-v2";

/// Resolve the exact math-thread default from the pinned native common runtime.
#[must_use]
pub fn default_math_threads() -> u32 {
    let value = unsafe { sys::llama_rs_common_default_math_threads() };
    value.max(1).cast_unsigned()
}

/// Outcome returned by the pinned `common_fit_params` implementation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum FitStatus {
    /// The projected allocations satisfy the requested margins.
    Success,
    /// The fitter could not find a projected allocation satisfying the constraints.
    Failure,
    /// Model inspection or another native operation failed.
    Error,
}

/// A typed representation of llama.cpp's GPU-layer setting.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum FitGpuLayers {
    /// Let `common/fit` choose the number of offloaded layers (`-1`).
    Auto,
    /// Offload every supported layer (native values at or below `-2`).
    All,
    /// Request an explicit number of layers.
    Count(u32),
}

impl FitGpuLayers {
    fn from_raw(value: i32) -> Self {
        match value {
            -1 => Self::Auto,
            value if value < -1 => Self::All,
            value => Self::Count(value.cast_unsigned()),
        }
    }
}

/// Requested or fitted model/context configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct FitConfiguration {
    /// Raw context setting. `None` means use the model's trained context.
    pub context_tokens: Option<u32>,
    /// Context setting resolved against the inspected model metadata.
    pub resolved_context_tokens: u32,
    /// Typed GPU-layer setting.
    pub gpu_layers: FitGpuLayers,
    /// Exact native GPU-layer value, retained for parity checks and CLI reproduction.
    pub raw_gpu_layers: i32,
    /// Number of layers projected to be offloaded after resolving auto/all settings.
    pub resolved_gpu_layers: u32,
}

/// Stable model metadata needed to interpret a fit result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct FitModelInfo {
    /// Transformer layer count from the GGUF model.
    pub layer_count: u32,
    /// Maximum offload count, including llama.cpp's output-layer allocation.
    pub offloadable_layer_count: u32,
    /// Context length recorded by the model.
    pub context_tokens: u32,
    /// Number of experts, or zero for a dense model.
    pub expert_count: u32,
    /// Exact tensor storage bytes reported by llama.cpp, independent of active experts.
    pub tensor_bytes: u64,
}

/// One model-free throughput calibration for a native backend operation.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[cfg_attr(feature = "serde", serde(deny_unknown_fields))]
pub struct FitCalibrationMetric {
    /// Raw `ggml_backend_dev_type` value.
    pub backend_type: i32,
    /// Backend registration name.
    pub backend: String,
    /// Backend-reported physical device identity, when available.
    pub device_id: Option<String>,
    /// Raw `ggml_type` used by the synthetic weight tensor.
    pub tensor_type: i32,
    /// Whether the calibration used routed `MUL_MAT_ID` work.
    pub routed: bool,
    /// Effective weight bytes consumed per second by the operation.
    pub bytes_per_second: f64,
    /// Per-operation launch cost observed by native calibration.
    pub launch_microseconds: f64,
    /// Relative spread across the bounded calibration samples.
    pub relative_spread: f64,
    /// Number of independent timed blocks retained by adaptive calibration.
    pub sample_count: u32,
    /// Total native operation time represented by the retained samples.
    pub measured_microseconds: u64,
    /// Whether the retained samples converged within the calibration budget.
    pub stable: bool,
}

/// Serializable, model-free calibration of the enabled native backends.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct FitCalibration {
    /// Stable identity of the native calibration procedure.
    pub method: String,
    /// Native calibration metrics keyed by device, tensor type, and operation class.
    pub metrics: Vec<FitCalibrationMetric>,
    /// Wall time spent in bounded synthetic backend operations.
    pub elapsed_microseconds: u64,
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for FitCalibration {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct WireCalibration {
            method: String,
            metrics: Vec<FitCalibrationMetric>,
            elapsed_microseconds: u64,
        }

        let wire = WireCalibration::deserialize(deserializer)?;
        let calibration = Self {
            method: wire.method,
            metrics: wire.metrics,
            elapsed_microseconds: wire.elapsed_microseconds,
        };
        calibration.validate().map_err(serde::de::Error::custom)?;
        Ok(calibration)
    }
}

/// How llama.cpp uses one tensor during single-token decode.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum FitTensorWorkloadKind {
    /// The complete tensor participates in every generated token.
    AlwaysActive,
    /// The tensor is a complete routed-expert pool; the caller chooses the active fraction.
    RoutedExpert,
    /// The tensor is accessed through a native row lookup.
    RowLookup,
}

/// Native facts for one fitted model tensor.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct FitTensorWorkload {
    /// Canonical llama.cpp tensor name.
    pub name: String,
    /// Raw `ggml_backend_dev_type` of the fitted buffer device.
    pub backend_type: i32,
    /// Backend registration name.
    pub backend: String,
    /// Backend-reported physical device identity, when available.
    pub device_id: Option<String>,
    /// Raw tensor `ggml_type`.
    pub tensor_type: i32,
    /// Native access class.
    pub kind: FitTensorWorkloadKind,
    /// Whether ordinary target-model decode executes this tensor.
    pub baseline_executed: bool,
    /// Complete tensor storage.
    pub stored_bytes: u64,
    /// Bytes touched by one operation before routed-expert selection is applied.
    pub operation_bytes: u64,
}

/// One concrete attention row read for every occupied token.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct FitAttentionRowWorkload {
    /// Raw cache `ggml_type`.
    pub tensor_type: i32,
    /// Native row bytes for one occupied token.
    pub bytes_per_token: u64,
}

/// Architecture-specific attention storage for one fitted layer.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[cfg_attr(feature = "serde", serde(tag = "kind", rename_all = "snake_case"))]
pub enum FitAttentionWorkload {
    /// The layer has no context-dependent attention storage.
    None,
    /// Ordinary attention stores independently typed K and V rows.
    Conventional {
        /// Key-cache row.
        key: FitAttentionRowWorkload,
        /// Value-cache row.
        value: FitAttentionRowWorkload,
    },
    /// Multi-head latent attention stores one latent row.
    Mla {
        /// Latent attention-cache row.
        latent: FitAttentionRowWorkload,
    },
}

/// Native attention and recurrent facts for one fitted model layer.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct FitKvLayerWorkload {
    /// Zero-based transformer layer.
    pub layer: u32,
    /// Raw `ggml_backend_dev_type` of the fitted KV device.
    pub backend_type: i32,
    /// Backend registration name.
    pub backend: String,
    /// Backend-reported physical device identity, when available.
    pub device_id: Option<String>,
    /// Complete attention storage shape for this architecture.
    pub attention: FitAttentionWorkload,
    /// Native attention key-head width used by architecture-specific cache state.
    pub attention_head_size: u32,
    /// Raw `ggml_type` of architecture-specific fixed attention state.
    pub attention_state_type: i32,
    /// Sliding-window cap, or zero for full attention.
    pub sliding_window_tokens: u32,
    /// Native cache compression ratio, or zero for ordinary KV/recurrent layers.
    pub compression_ratio: u32,
    /// Whether the baseline graph executes a sparse context index for this layer.
    pub sparse_index: bool,
    /// Native sparse-index key row bytes for one occupied position.
    pub indexer_bytes_per_token: u64,
    /// Whether this layer is recurrent.
    pub recurrent: bool,
    /// Raw `ggml_type` of the fixed recurrent state.
    pub recurrent_type: i32,
    /// Fixed convolution-state bytes read or written for one sequence.
    pub recurrent_conv_bytes: u64,
    /// Fixed recurrent matrix-state bytes read or written for one sequence.
    pub recurrent_state_bytes: u64,
}

/// Native decode facts attached to a fitted no-allocation model.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct FitDecodeWorkload {
    /// Stable native workload schema identity.
    pub method: String,
    /// Canonical llama.cpp architecture identity for this artifact.
    pub architecture: String,
    /// Total routed experts declared by the model.
    pub expert_count: u32,
    /// Routed experts selected for one token.
    pub expert_used_count: u32,
    /// Auxiliary NextN/MTP layers stored by the artifact.
    pub nextn_layer_count: u32,
    /// Native compressed KV/MLA rank, or zero when unused.
    pub kv_lora_rank: u32,
    /// Sparse-index query head count, or zero when unused.
    pub indexer_head_count: u32,
    /// Sparse-index key width, or zero when unused.
    pub indexer_head_size: u32,
    /// Sparse positions gathered per layer, or zero when unused.
    pub indexer_top_k: u32,
    /// Whether the model mixes native layer architectures.
    pub hybrid_model: bool,
    /// Whether the model contains recurrent execution.
    pub recurrent_model: bool,
    /// Exact fitted tensor facts.
    pub tensors: Vec<FitTensorWorkload>,
    /// Exact per-layer KV facts.
    pub kv_layers: Vec<FitKvLayerWorkload>,
}

/// Availability of native decode facts. This type intentionally contains no speed formula.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[cfg_attr(feature = "serde", serde(tag = "status", rename_all = "snake_case"))]
pub enum FitDecodeWorkloadAssessment {
    /// Native workload extraction succeeded.
    Available {
        /// Exact native decode facts.
        workload: FitDecodeWorkload,
    },
    /// Workload extraction was not requested or could not be completed.
    Unavailable {
        /// Native diagnostic or the explicit not-requested reason.
        reason: String,
    },
}

/// Allocation classes reported by llama.cpp's graph planner.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct FitAllocations {
    /// Model tensor allocation.
    pub model_bytes: u64,
    /// Context and KV allocation.
    pub context_bytes: u64,
    /// Temporary compute-buffer allocation.
    pub compute_bytes: u64,
    /// Sum of model, context, and compute allocations.
    pub total_bytes: u64,
}

/// How an estimate compares with the caller's margin for this device.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct FitMemoryTarget {
    /// Memory available to the fitter before the projected allocations.
    /// Accelerator estimates use reported free memory; CPU-only estimates use
    /// total host memory, matching the pinned implementation.
    pub available_bytes: i64,
    /// Requested memory to leave unallocated.
    pub margin_bytes: u64,
    /// Maximum allocation allowed after applying the margin.
    pub max_allocation_bytes: i64,
    /// Memory projected to remain after the estimated allocations.
    pub projected_remaining_bytes: i64,
    /// Amount by which the projected remaining memory misses the margin.
    pub shortfall_bytes: u64,
}

/// A no-allocation memory estimate for one device and one configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct FitMemoryEstimate {
    /// Device-reported total memory.
    pub total_bytes: i64,
    /// Device-reported currently free memory.
    pub free_bytes: i64,
    /// Planned allocation breakdown.
    pub allocations: FitAllocations,
    /// Margin comparison when this device participates in fitting.
    pub target: Option<FitMemoryTarget>,
}

/// Role of a device in `common/fit`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum FitDeviceKind {
    /// GPU, integrated GPU, or another model offload device.
    Accelerator,
    /// CPU/host memory aggregate.
    Host,
}

/// Initial and fitted estimates for one stable device identity.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct FitDeviceEstimate {
    /// Stable index used by tensor splits and placement targets.
    pub index: usize,
    /// Host or accelerator role.
    pub kind: FitDeviceKind,
    /// Raw `ggml_backend_dev_type` value from the pinned backend.
    pub backend_type: i32,
    /// Backend registration name.
    pub backend: String,
    /// Backend-reported physical device identity, when available.
    pub device_id: Option<String>,
    /// Backend device name.
    pub name: String,
    /// Human-readable backend description.
    pub description: String,
    /// Estimate before `common_fit_params` mutates the configuration.
    pub initial: Option<FitMemoryEstimate>,
    /// Estimate after the fit attempt.
    pub fitted: Option<FitMemoryEstimate>,
    /// Margin applied to this device. Host memory has no target when an
    /// accelerator is present because upstream assumes host memory is unlimited.
    pub margin_bytes: Option<u64>,
}

impl FitDeviceEstimate {
    /// Typed backend device class corresponding to [`Self::backend_type`].
    #[must_use]
    pub fn backend_device_type(&self) -> crate::LlamaBackendDeviceType {
        match i64::from(self.backend_type) {
            value if value == i64::from(sys::GGML_BACKEND_DEVICE_TYPE_CPU) => {
                crate::LlamaBackendDeviceType::Cpu
            }
            value if value == i64::from(sys::GGML_BACKEND_DEVICE_TYPE_ACCEL) => {
                crate::LlamaBackendDeviceType::Accelerator
            }
            value if value == i64::from(sys::GGML_BACKEND_DEVICE_TYPE_GPU) => {
                crate::LlamaBackendDeviceType::Gpu
            }
            value if value == i64::from(sys::GGML_BACKEND_DEVICE_TYPE_IGPU) => {
                crate::LlamaBackendDeviceType::IntegratedGpu
            }
            _ => crate::LlamaBackendDeviceType::Unknown,
        }
    }
}

/// Target selected by a tensor buffer override.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[cfg_attr(feature = "serde", serde(tag = "type", rename_all = "snake_case"))]
pub enum FitPlacementTarget {
    /// A host buffer type.
    Host,
    /// A specific backend device.
    Device {
        /// Index in [`FitReport::devices`] when the target could be matched.
        index: Option<usize>,
        /// Backend device name.
        name: String,
        /// Human-readable backend description.
        description: String,
    },
    /// A buffer type without a host or concrete device classification.
    Other,
}

/// A tensor-pattern placement generated by `common/fit` (primarily for `MoE` overflow).
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct FitTensorPlacement {
    /// Upstream tensor-name regular expression.
    pub pattern: String,
    /// Upstream backend buffer type name.
    pub buffer_type: String,
    /// Semantic placement target.
    pub target: FitPlacementTarget,
}

/// Configuration changes made by the fitter.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[cfg_attr(feature = "serde", serde(tag = "type", rename_all = "snake_case"))]
pub enum FitAdjustment {
    /// Context length was reduced to satisfy memory constraints.
    ContextReduced {
        /// Resolved context before fitting.
        from: u32,
        /// Resolved context after fitting.
        to: u32,
    },
    /// The number of offloaded layers was reduced.
    GpuLayersReduced {
        /// Resolved offload count before fitting.
        from: u32,
        /// Resolved offload count after fitting.
        to: u32,
    },
    /// A multi-device tensor split was selected.
    TensorSplitApplied {
        /// Per-device split weights as written by `common/fit`.
        values: Vec<f32>,
    },
    /// Tensor-pattern buffer overrides were generated.
    TensorPlacementsApplied {
        /// Number of generated placement rules.
        count: usize,
    },
}

/// Measurement phase associated with a diagnostic.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum FitPhase {
    /// Configuration before fitting.
    Initial,
    /// Configuration after fitting.
    Fitted,
}

/// Typed caveats and failures associated with a report.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[cfg_attr(feature = "serde", serde(tag = "type", rename_all = "snake_case"))]
pub enum FitWarning {
    /// No accelerator was selected or available.
    NoAccelerator,
    /// Upstream fitting ignores host pressure when any accelerator is present.
    HostMemoryAssumedUnlimited,
    /// A GPU-like backend did not report a usable memory budget.
    DeviceMemoryUnknown {
        /// Device index in [`FitReport::devices`].
        device_index: usize,
        /// Backend device name.
        device_name: String,
    },
    /// A fitted estimate still misses its requested margin.
    MarginNotMet {
        /// Device index in [`FitReport::devices`].
        device_index: usize,
        /// Backend device name.
        device_name: String,
        /// Remaining deficit.
        shortfall_bytes: u64,
    },
    /// One of the structured measurements could not be produced.
    DiagnosticsUnavailable {
        /// Measurement phase.
        phase: FitPhase,
        /// Native exception message from the measurement path.
        message: String,
    },
    /// `common_fit_params` could not meet the constraints.
    FitFailed,
    /// `common_fit_params` encountered a hard error.
    FitError,
}

/// Errors in the Magnitude C bridge or report decoding.
#[derive(Debug, thiserror::Error)]
pub enum FitReportError {
    /// The margin vector is shorter than `llama_max_devices()`.
    #[error("fit margins contain {provided} entries but at least {required} are required")]
    InvalidMargins {
        /// Entries supplied by the caller.
        provided: usize,
        /// Entries required by the native build.
        required: usize,
    },
    /// The native bridge rejected the call.
    #[error("llama.cpp fit bridge failed with status {status}: {message}")]
    Native {
        /// Raw `llama_rs_status` value.
        status: i32,
        /// Native diagnostic.
        message: String,
    },
    /// The bridge returned a malformed report.
    #[error("malformed llama.cpp fit report: {0}")]
    Malformed(&'static str),
    /// The pinned adapter returned an enum value this safe wrapper does not know.
    #[error("unknown {kind} value {value} in llama.cpp fit report")]
    UnknownEnum {
        /// Enum being decoded.
        kind: &'static str,
        /// Raw enum value.
        value: i64,
    },
    /// A derived byte count could not be represented by the public DTO.
    #[error("fit report arithmetic overflow while calculating {field}")]
    ArithmeticOverflow {
        /// Derived value that overflowed.
        field: &'static str,
    },
    /// Rust could not reserve space for a native collection projection.
    #[error("could not reserve {requested} entries for fit report {collection}")]
    RustAllocation {
        /// Collection being decoded.
        collection: &'static str,
        /// Native element count.
        requested: usize,
    },
    /// A required string field was absent from the native report.
    #[error("llama.cpp fit report omitted required string field {field}")]
    MissingString {
        /// Stable DTO field being decoded.
        field: &'static str,
    },
    /// A native string could not be represented by this UTF-8 Rust API.
    #[error("llama.cpp fit report field {field} is not valid UTF-8")]
    InvalidUtf8 {
        /// Stable DTO field being decoded.
        field: &'static str,
    },
    /// A native or deserialized calibration value was outside its numeric domain.
    #[error("llama.cpp fit calibration field {field} is outside its numeric domain")]
    InvalidCalibrationNumber {
        /// Stable DTO field containing the invalid value.
        field: &'static str,
    },
    /// A deserialized calibration identity is empty or contains an embedded terminator.
    #[error("fit calibration field {field} contains an invalid identity")]
    InvalidCalibrationString {
        /// Stable DTO field containing the invalid string.
        field: &'static str,
    },
}

/// Typed, stable projection of llama.cpp `common/fit` and its memory diagnostics.
///
/// The estimates cover the fitted text model and context only. Projectors, draft models, other
/// loaded contexts, system/device safety margins beyond those passed to the fitter, and other ICN
/// reservations must be composed by the caller.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct FitReport {
    /// Native fit outcome. A report is returned for all three outcomes.
    pub status: FitStatus,
    /// Configuration supplied to the fitter.
    pub requested: FitConfiguration,
    /// Configuration left by the fitter.
    pub fitted: FitConfiguration,
    /// Model metadata used to resolve auto settings.
    pub model: FitModelInfo,
    /// Per-device initial/fitted memory estimates.
    pub devices: Vec<FitDeviceEstimate>,
    /// Per-accelerator split weights written by `common/fit`.
    pub tensor_split: Vec<f32>,
    /// Tensor buffer overrides written by `common/fit`.
    pub tensor_placements: Vec<FitTensorPlacement>,
    /// Typed configuration changes.
    pub adjustments: Vec<FitAdjustment>,
    /// Typed caveats, deficits, and failures.
    pub warnings: Vec<FitWarning>,
    /// Native decode facts for the effective fitted plan.
    pub decode_workload: FitDecodeWorkloadAssessment,
    /// Wall time for initial measurement, fitting, and fitted measurement.
    pub elapsed_microseconds: u64,
}

impl FitReport {
    /// Whether the fitted configuration is projected to satisfy upstream constraints.
    #[must_use]
    pub fn is_success(&self) -> bool {
        self.status == FitStatus::Success
    }
}

struct NativeFitReport(NonNull<sys::llama_rs_fit_report>);

struct NativeFitCalibration(NonNull<sys::llama_rs_fit_calibration>);

/// A no-allocation target model/context kept alive while fitting a linked MTP context.
#[derive(Debug)]
pub struct LinkedFitTarget<'a> {
    /// Exact target GGUF.
    pub model_path: &'a CStr,
    /// Exact target model loading parameters.
    pub model_params: &'a LlamaModelParams,
    /// Exact target context parameters.
    pub context_params: &'a LlamaContextParams,
}

impl Drop for NativeFitReport {
    fn drop(&mut self) {
        unsafe { sys::llama_rs_fit_report_free(self.0.as_ptr()) };
    }
}

impl Drop for NativeFitCalibration {
    fn drop(&mut self) {
        unsafe { sys::llama_rs_fit_calibration_free(self.0.as_ptr()) };
    }
}

impl FitCalibration {
    /// Run bounded model-free calibration against the initialized native backend registry.
    ///
    /// No model is loaded and no token decode is performed.
    ///
    /// # Errors
    ///
    /// Returns [`FitReportError`] when native calibration fails or returns malformed evidence.
    pub fn measure(_backend: &crate::llama_backend::LlamaBackend) -> Result<Self, FitReportError> {
        let _logger_guard = crate::log::lock_native_logger();
        let mut native = ptr::null_mut();
        let mut native_error: *mut c_char = ptr::null_mut();
        let status =
            unsafe { sys::llama_rs_fit_calibration_create(&raw mut native, &raw mut native_error) };
        if status != sys::LLAMA_RS_STATUS_OK {
            return Err(FitReportError::Native {
                status,
                message: take_native_error(native_error),
            });
        }
        if !native_error.is_null() {
            unsafe { sys::llama_rs_string_free(native_error) };
        }
        let native = NativeFitCalibration(
            NonNull::new(native).ok_or(FitReportError::Malformed("null calibration"))?,
        );
        let metric_count = unsafe { sys::llama_rs_fit_calibration_metric_count(native.0.as_ptr()) };
        let mut metrics = Vec::new();
        metrics
            .try_reserve_exact(metric_count)
            .map_err(|_| FitReportError::RustAllocation {
                collection: "calibration metrics",
                requested: metric_count,
            })?;
        for index in 0..metric_count {
            let mut raw = MaybeUninit::<sys::llama_rs_fit_calibration_metric>::uninit();
            if !unsafe {
                sys::llama_rs_fit_calibration_get_metric(native.0.as_ptr(), index, raw.as_mut_ptr())
            } {
                return Err(FitReportError::Malformed("missing calibration metric"));
            }
            let raw = unsafe { raw.assume_init() };
            validate_positive_finite(raw.bytes_per_second, "calibration.bytes_per_second")?;
            validate_nonnegative_finite(
                raw.launch_microseconds,
                "calibration.launch_microseconds",
            )?;
            validate_nonnegative_finite(raw.relative_spread, "calibration.relative_spread")?;
            metrics.push(FitCalibrationMetric {
                backend_type: raw.backend_type,
                backend: borrowed_string(raw.backend, "calibration.backend")?,
                device_id: optional_borrowed_string(raw.device_id, "calibration.device_id")?,
                tensor_type: raw.tensor_type,
                routed: raw.routed,
                bytes_per_second: raw.bytes_per_second,
                launch_microseconds: raw.launch_microseconds,
                relative_spread: raw.relative_spread,
                sample_count: raw.sample_count,
                measured_microseconds: raw.measured_microseconds,
                stable: raw.stable,
            });
        }
        let elapsed =
            unsafe { sys::llama_rs_fit_calibration_elapsed_microseconds(native.0.as_ptr()) };
        let method = borrowed_string(
            unsafe { sys::llama_rs_fit_calibration_method(native.0.as_ptr()) },
            "calibration.method",
        )?;
        let calibration = Self {
            method,
            metrics,
            elapsed_microseconds: elapsed.max(0).cast_unsigned(),
        };
        calibration.validate()?;
        Ok(calibration)
    }

    fn validate(&self) -> Result<(), FitReportError> {
        if self.method != FIT_CALIBRATION_METHOD {
            return Err(FitReportError::Malformed(
                "calibration method does not match the native calibration schema",
            ));
        }
        if self.metrics.is_empty() {
            return Err(FitReportError::Malformed("calibration has no metrics"));
        }
        for metric in &self.metrics {
            validate_positive_finite(metric.bytes_per_second, "calibration.bytes_per_second")?;
            validate_nonnegative_finite(
                metric.launch_microseconds,
                "calibration.launch_microseconds",
            )?;
            validate_nonnegative_finite(metric.relative_spread, "calibration.relative_spread")?;
            if metric.sample_count == 0 {
                return Err(FitReportError::Malformed(
                    "calibration metric has no retained samples",
                ));
            }
            if metric.measured_microseconds == 0 {
                return Err(FitReportError::Malformed(
                    "calibration metric has no measured duration",
                ));
            }
            if metric.backend.is_empty() || metric.backend.as_bytes().contains(&0) {
                return Err(FitReportError::InvalidCalibrationString {
                    field: "calibration.backend",
                });
            }
            if metric
                .device_id
                .as_ref()
                .is_some_and(|value| value.is_empty() || value.as_bytes().contains(&0))
            {
                return Err(FitReportError::InvalidCalibrationString {
                    field: "calibration.device_id",
                });
            }
        }
        Ok(())
    }
}

impl LlamaModelParams {
    /// Measure several execution contexts while constructing the no-allocation model once.
    ///
    /// This is an exact projection of llama.cpp model/context graphs. It does not run
    /// `common_fit_params` or alter the supplied parameters.
    ///
    /// # Errors
    ///
    /// Returns [`FitReportError`] for invalid buffers, native inspection failures, or malformed
    /// bridge results.
    pub fn measure_contexts(
        &self,
        model_path: &CStr,
        contexts: &[LlamaContextParams],
        margins: &[usize],
    ) -> Result<Vec<FitReport>, FitReportError> {
        self.measure_contexts_impl(model_path, contexts, margins, false)
    }

    /// Measure contexts and attach native decode-workload facts.
    ///
    /// The no-allocation model is constructed once for all contexts. This method performs no
    /// throughput calculation and does not run a model decode.
    ///
    /// # Errors
    ///
    /// Returns [`FitReportError`] for invalid buffers, native inspection failures, or malformed
    /// bridge results.
    pub fn measure_contexts_with_decode_workload(
        &self,
        model_path: &CStr,
        contexts: &[LlamaContextParams],
        margins: &[usize],
    ) -> Result<Vec<FitReport>, FitReportError> {
        self.measure_contexts_impl(model_path, contexts, margins, true)
    }

    fn measure_contexts_impl(
        &self,
        model_path: &CStr,
        contexts: &[LlamaContextParams],
        margins: &[usize],
        capture_decode_workload: bool,
    ) -> Result<Vec<FitReport>, FitReportError> {
        let _logger_guard = crate::log::lock_native_logger();
        let max_devices = unsafe { sys::llama_max_devices() };
        if margins.len() < max_devices {
            return Err(FitReportError::InvalidMargins {
                provided: margins.len(),
                required: max_devices,
            });
        }
        let raw_contexts = contexts
            .iter()
            .map(|context| context.context_params)
            .collect::<Vec<_>>();
        let mut raw_reports = vec![ptr::null_mut(); contexts.len()];
        let mut native_error: *mut c_char = ptr::null_mut();
        let status = unsafe {
            sys::llama_rs_fit_measure_reports_create(
                model_path.as_ptr(),
                &raw const self.params,
                raw_contexts.as_ptr(),
                raw_contexts.len(),
                margins.as_ptr(),
                margins.len(),
                capture_decode_workload,
                sys::GGML_LOG_LEVEL_ERROR,
                raw_reports.as_mut_ptr(),
                &raw mut native_error,
            )
        };
        if status != sys::LLAMA_RS_STATUS_OK {
            return Err(FitReportError::Native {
                status,
                message: take_native_error(native_error),
            });
        }
        if !native_error.is_null() {
            unsafe { sys::llama_rs_string_free(native_error) };
        }
        raw_reports
            .into_iter()
            .map(|report| {
                let report = NativeFitReport(
                    NonNull::new(report).ok_or(FitReportError::Malformed("null batch report"))?,
                );
                decode_report(&report)
            })
            .collect()
    }

    /// Fit unset model/context parameters and return structured memory diagnostics.
    ///
    /// The estimator uses llama.cpp's no-allocation model/context planning path:
    /// model tensor data is not loaded, but GGUF metadata and compute graphs are
    /// inspected.
    ///
    /// # Concurrency
    ///
    /// The upstream fit implementation temporarily replaces llama.cpp's process-global logger
    /// with a callback backed by call-local state. Concurrent fit calls are therefore unsupported.
    /// This binding intentionally uses the pinned upstream implementation unchanged.
    ///
    /// A report is returned even when its [`FitStatus`] is `Failure` or `Error`.
    /// The caller must only load the fitted parameters when the status is
    /// [`FitStatus::Success`].
    ///
    /// # Errors
    ///
    /// Returns [`FitReportError`] for invalid buffers, bridge failures, or a
    /// malformed result. Native fit failures are represented in the report.
    pub fn fit_params_report(
        self: Pin<&mut Self>,
        model_path: &CStr,
        cparams: &mut LlamaContextParams,
        margins: &mut [usize],
        n_ctx_min: u32,
    ) -> Result<FitReport, FitReportError> {
        self.fit_params_report_impl(model_path, cparams, None, margins, n_ctx_min, false)
    }

    /// Fit unset parameters and attach native decode-workload facts.
    ///
    /// # Errors
    ///
    /// Returns [`FitReportError`] for invalid buffers, native fit failures, or a malformed bridge
    /// result.
    pub fn fit_params_report_with_decode_workload(
        self: Pin<&mut Self>,
        model_path: &CStr,
        cparams: &mut LlamaContextParams,
        margins: &mut [usize],
        n_ctx_min: u32,
    ) -> Result<FitReport, FitReportError> {
        self.fit_params_report_impl(model_path, cparams, None, margins, n_ctx_min, true)
    }

    /// Fit an MTP model/context while a no-allocation target context is linked as `ctx_other`.
    ///
    /// The returned report contains only the fitted model/context allocations. Compose it with a
    /// separate target report to assess the full execution plan.
    ///
    /// # Errors
    ///
    /// Returns [`FitReportError`] for invalid buffers, native fit failures, or a malformed bridge
    /// result.
    pub fn fit_params_report_linked(
        self: Pin<&mut Self>,
        model_path: &CStr,
        cparams: &mut LlamaContextParams,
        target: LinkedFitTarget<'_>,
        margins: &mut [usize],
        n_ctx_min: u32,
    ) -> Result<FitReport, FitReportError> {
        self.fit_params_report_impl(model_path, cparams, Some(target), margins, n_ctx_min, false)
    }

    fn fit_params_report_impl(
        mut self: Pin<&mut Self>,
        model_path: &CStr,
        cparams: &mut LlamaContextParams,
        linked_target: Option<LinkedFitTarget<'_>>,
        margins: &mut [usize],
        n_ctx_min: u32,
        capture_decode_workload: bool,
    ) -> Result<FitReport, FitReportError> {
        let _logger_guard = crate::log::lock_native_logger();
        let max_devices = unsafe { sys::llama_max_devices() };
        if margins.len() < max_devices {
            return Err(FitReportError::InvalidMargins {
                provided: margins.len(),
                required: max_devices,
            });
        }
        let max_overrides = unsafe { sys::llama_max_tensor_buft_overrides() };

        // Keep an explicitly configured tensor split attached to `mparams` so
        // the initial measurement describes the plan the caller actually
        // requested. `common_fit_params` still requires a separate output
        // buffer for an automatically selected split.
        let original_tensor_split = self.params.tensor_split;
        let mut fitted_tensor_split = vec![0.0; max_devices];
        self.buft_overrides.clear();
        self.buft_overrides.resize(
            max_overrides + 1,
            sys::llama_model_tensor_buft_override {
                pattern: ptr::null(),
                buft: ptr::null_mut(),
            },
        );
        self.params.tensor_buft_overrides = ptr::null();

        let mut native_report = ptr::null_mut();
        let mut native_error: *mut c_char = ptr::null_mut();
        let status = unsafe {
            match linked_target {
                Some(target) => sys::llama_rs_fit_report_create_linked(
                    model_path.as_ptr(),
                    &raw mut self.params,
                    &raw mut cparams.context_params,
                    target.model_path.as_ptr(),
                    &raw const target.model_params.params,
                    &raw const target.context_params.context_params,
                    fitted_tensor_split.as_mut_ptr(),
                    self.buft_overrides.as_mut_ptr(),
                    margins.as_mut_ptr(),
                    margins.len(),
                    capture_decode_workload,
                    n_ctx_min,
                    sys::GGML_LOG_LEVEL_ERROR,
                    &raw mut native_report,
                    &raw mut native_error,
                ),
                None => sys::llama_rs_fit_report_create(
                    model_path.as_ptr(),
                    &raw mut self.params,
                    &raw mut cparams.context_params,
                    fitted_tensor_split.as_mut_ptr(),
                    self.buft_overrides.as_mut_ptr(),
                    margins.as_mut_ptr(),
                    margins.len(),
                    capture_decode_workload,
                    n_ctx_min,
                    sys::GGML_LOG_LEVEL_ERROR,
                    &raw mut native_report,
                    &raw mut native_error,
                ),
            }
        };

        // common/fit may point the raw params at these buffers even on a failed
        // fit attempt. Preserve an explicit caller split, otherwise retain the
        // fitted split in this parameter object's owned storage so the exact
        // parameters that produced the report can be consumed by model loading.
        if original_tensor_split.is_null() {
            self.tensor_split = fitted_tensor_split;
            self.params.tensor_split = self.tensor_split.as_ptr();
        } else {
            self.params.tensor_split = original_tensor_split;
        }
        self.params.tensor_buft_overrides = self.buft_overrides.as_ptr();

        if status != sys::LLAMA_RS_STATUS_OK {
            return Err(FitReportError::Native {
                status,
                message: take_native_error(native_error),
            });
        }
        if !native_error.is_null() {
            unsafe { sys::llama_rs_string_free(native_error) };
        }
        let native_report = NativeFitReport(
            NonNull::new(native_report).ok_or(FitReportError::Malformed("null report"))?,
        );
        decode_report(&native_report)
    }
}

fn decode_report(native: &NativeFitReport) -> Result<FitReport, FitReportError> {
    let mut summary = MaybeUninit::<sys::llama_rs_fit_summary>::uninit();
    if !unsafe { sys::llama_rs_fit_report_get_summary(native.0.as_ptr(), summary.as_mut_ptr()) } {
        return Err(FitReportError::Malformed("missing summary"));
    }
    let summary = unsafe { summary.assume_init() };
    let status = decode_status(summary.status)?;
    let (requested, fitted) = decode_configurations(&summary);

    let device_count = unsafe { sys::llama_rs_fit_report_device_count(native.0.as_ptr()) };
    let mut devices = Vec::new();
    devices
        .try_reserve_exact(device_count)
        .map_err(|_| FitReportError::RustAllocation {
            collection: "devices",
            requested: device_count,
        })?;
    for index in 0..device_count {
        devices.push(decode_device(native, index)?);
    }

    let tensor_split_count =
        unsafe { sys::llama_rs_fit_report_tensor_split_count(native.0.as_ptr()) };
    let mut tensor_split = Vec::new();
    tensor_split
        .try_reserve_exact(tensor_split_count)
        .map_err(|_| FitReportError::RustAllocation {
            collection: "tensor split",
            requested: tensor_split_count,
        })?;
    for index in 0..tensor_split_count {
        let mut value = 0.0;
        if !unsafe {
            sys::llama_rs_fit_report_get_tensor_split(native.0.as_ptr(), index, &raw mut value)
        } {
            return Err(FitReportError::Malformed("missing tensor split entry"));
        }
        tensor_split.push(value);
    }

    let placement_count = unsafe { sys::llama_rs_fit_report_placement_count(native.0.as_ptr()) };
    let mut placements = Vec::new();
    placements
        .try_reserve_exact(placement_count)
        .map_err(|_| FitReportError::RustAllocation {
            collection: "tensor placements",
            requested: placement_count,
        })?;
    for index in 0..placement_count {
        placements.push(decode_placement(native, index)?);
    }

    let initial_error = optional_borrowed_string(
        unsafe { sys::llama_rs_fit_report_initial_error(native.0.as_ptr()) },
        "initial_error",
    )?;
    let fitted_error = optional_borrowed_string(
        unsafe { sys::llama_rs_fit_report_fitted_error(native.0.as_ptr()) },
        "fitted_error",
    )?;
    let mut adjustments = derive_adjustments(requested, fitted, &tensor_split, placements.len());
    adjustments.shrink_to_fit();
    let warnings = derive_warnings(
        status,
        summary.accelerator_count,
        summary.initial_measurement_available,
        summary.fitted_measurement_available,
        initial_error,
        fitted_error,
        &devices,
    );
    let offloadable_layer_count =
        summary
            .model_layer_count
            .checked_add(1)
            .ok_or(FitReportError::ArithmeticOverflow {
                field: "offloadable layer count",
            })?;
    let decode_workload = decode_workload(native)?;

    Ok(FitReport {
        status,
        requested,
        fitted,
        model: FitModelInfo {
            layer_count: summary.model_layer_count,
            offloadable_layer_count,
            context_tokens: summary.model_context_tokens,
            expert_count: summary.model_expert_count,
            tensor_bytes: summary.model_tensor_bytes,
        },
        devices,
        tensor_split,
        tensor_placements: placements,
        adjustments,
        warnings,
        decode_workload,
        elapsed_microseconds: summary.elapsed_microseconds.max(0).cast_unsigned(),
    })
}

fn decode_workload(
    native: &NativeFitReport,
) -> Result<FitDecodeWorkloadAssessment, FitReportError> {
    let mut raw = MaybeUninit::<sys::llama_rs_fit_decode_workload_summary>::uninit();
    if !unsafe {
        sys::llama_rs_fit_report_get_decode_workload_summary(native.0.as_ptr(), raw.as_mut_ptr())
    } {
        return Err(FitReportError::Malformed("missing decode workload summary"));
    }
    let raw = unsafe { raw.assume_init() };
    if !raw.available {
        return Ok(FitDecodeWorkloadAssessment::Unavailable {
            reason: borrowed_string(raw.unavailable_reason, "decode_workload.unavailable_reason")?,
        });
    }
    let method = borrowed_string(raw.method, "decode_workload.method")?;
    if method != FIT_DECODE_WORKLOAD_METHOD {
        return Err(FitReportError::Malformed(
            "decode workload method does not match the native schema",
        ));
    }

    let tensor_count = unsafe { sys::llama_rs_fit_report_tensor_workload_count(native.0.as_ptr()) };
    if tensor_count == 0 {
        return Err(FitReportError::Malformed("decode workload has no tensors"));
    }
    let mut tensors = Vec::new();
    tensors
        .try_reserve_exact(tensor_count)
        .map_err(|_| FitReportError::RustAllocation {
            collection: "decode workload tensors",
            requested: tensor_count,
        })?;
    for index in 0..tensor_count {
        tensors.push(decode_tensor_workload(native, index)?);
    }

    let layer_count =
        unsafe { sys::llama_rs_fit_report_kv_layer_workload_count(native.0.as_ptr()) };
    let mut kv_layers = Vec::new();
    kv_layers
        .try_reserve_exact(layer_count)
        .map_err(|_| FitReportError::RustAllocation {
            collection: "decode workload KV layers",
            requested: layer_count,
        })?;
    for index in 0..layer_count {
        kv_layers.push(decode_kv_layer_workload(native, index, raw.mla)?);
    }

    Ok(FitDecodeWorkloadAssessment::Available {
        workload: FitDecodeWorkload {
            method,
            architecture: borrowed_string(raw.architecture, "decode_workload.architecture")?,
            expert_count: raw.expert_count,
            expert_used_count: raw.expert_used_count,
            nextn_layer_count: raw.nextn_layer_count,
            kv_lora_rank: raw.kv_lora_rank,
            indexer_head_count: raw.indexer_head_count,
            indexer_head_size: raw.indexer_head_size,
            indexer_top_k: raw.indexer_top_k,
            hybrid_model: raw.hybrid_model,
            recurrent_model: raw.recurrent_model,
            tensors,
            kv_layers,
        },
    })
}

fn decode_tensor_workload(
    native: &NativeFitReport,
    index: usize,
) -> Result<FitTensorWorkload, FitReportError> {
    let mut raw = MaybeUninit::<sys::llama_rs_fit_tensor_workload>::uninit();
    if !unsafe {
        sys::llama_rs_fit_report_get_tensor_workload(native.0.as_ptr(), index, raw.as_mut_ptr())
    } {
        return Err(FitReportError::Malformed("missing decode tensor workload"));
    }
    let raw = unsafe { raw.assume_init() };
    let kind = match raw.kind {
        sys::LLAMA_RS_FIT_TENSOR_ALWAYS_ACTIVE => FitTensorWorkloadKind::AlwaysActive,
        sys::LLAMA_RS_FIT_TENSOR_ROUTED_EXPERT => FitTensorWorkloadKind::RoutedExpert,
        sys::LLAMA_RS_FIT_TENSOR_ROW_LOOKUP => FitTensorWorkloadKind::RowLookup,
        value => {
            return Err(FitReportError::UnknownEnum {
                kind: "fit tensor workload kind",
                value: i64::from(value),
            });
        }
    };
    if raw.stored_bytes == 0 || raw.operation_bytes == 0 || raw.operation_bytes > raw.stored_bytes {
        return Err(FitReportError::Malformed(
            "decode tensor workload has invalid byte counts",
        ));
    }
    Ok(FitTensorWorkload {
        name: borrowed_string(raw.name, "decode_workload.tensors[].name")?,
        backend_type: raw.backend_type,
        backend: borrowed_string(raw.backend, "decode_workload.tensors[].backend")?,
        device_id: optional_borrowed_string(raw.device_id, "decode_workload.tensors[].device_id")?,
        tensor_type: raw.tensor_type,
        kind,
        baseline_executed: raw.baseline_executed,
        stored_bytes: raw.stored_bytes,
        operation_bytes: raw.operation_bytes,
    })
}

fn decode_kv_layer_workload(
    native: &NativeFitReport,
    index: usize,
    mla: bool,
) -> Result<FitKvLayerWorkload, FitReportError> {
    let mut raw = MaybeUninit::<sys::llama_rs_fit_kv_layer_workload>::uninit();
    if !unsafe {
        sys::llama_rs_fit_report_get_kv_layer_workload(native.0.as_ptr(), index, raw.as_mut_ptr())
    } {
        return Err(FitReportError::Malformed(
            "missing decode KV layer workload",
        ));
    }
    let raw = unsafe { raw.assume_init() };
    let attention = decode_attention_workload(
        raw.key_type,
        raw.value_type,
        raw.key_bytes_per_token,
        raw.value_bytes_per_token,
        mla,
    )?;
    Ok(FitKvLayerWorkload {
        layer: raw.layer,
        backend_type: raw.backend_type,
        backend: borrowed_string(raw.backend, "decode_workload.kv_layers[].backend")?,
        device_id: optional_borrowed_string(
            raw.device_id,
            "decode_workload.kv_layers[].device_id",
        )?,
        attention,
        attention_head_size: raw.attention_head_size,
        attention_state_type: raw.attention_state_type,
        sliding_window_tokens: raw.sliding_window_tokens,
        compression_ratio: raw.compression_ratio,
        sparse_index: raw.sparse_index,
        indexer_bytes_per_token: raw.indexer_bytes_per_token,
        recurrent: raw.recurrent,
        recurrent_type: raw.recurrent_type,
        recurrent_conv_bytes: raw.recurrent_conv_bytes,
        recurrent_state_bytes: raw.recurrent_state_bytes,
    })
}

fn decode_attention_workload(
    key_type: i32,
    value_type: i32,
    key_bytes: u64,
    value_bytes: u64,
    mla: bool,
) -> Result<FitAttentionWorkload, FitReportError> {
    match (mla, key_bytes, value_bytes) {
        (true, key_bytes, 0) if key_bytes > 0 => Ok(FitAttentionWorkload::Mla {
            latent: FitAttentionRowWorkload {
                tensor_type: key_type,
                bytes_per_token: key_bytes,
            },
        }),
        (false, 0, 0) => Ok(FitAttentionWorkload::None),
        (false, key_bytes, value_bytes) if key_bytes > 0 && value_bytes > 0 => {
            Ok(FitAttentionWorkload::Conventional {
                key: FitAttentionRowWorkload {
                    tensor_type: key_type,
                    bytes_per_token: key_bytes,
                },
                value: FitAttentionRowWorkload {
                    tensor_type: value_type,
                    bytes_per_token: value_bytes,
                },
            })
        }
        _ => Err(FitReportError::Malformed(
            "decode attention workload does not match its architecture",
        )),
    }
}

fn decode_configurations(
    summary: &sys::llama_rs_fit_summary,
) -> (FitConfiguration, FitConfiguration) {
    let requested = FitConfiguration {
        context_tokens: nonzero_option(summary.requested_context_tokens),
        resolved_context_tokens: summary.resolved_requested_context_tokens,
        gpu_layers: FitGpuLayers::from_raw(summary.requested_gpu_layers),
        raw_gpu_layers: summary.requested_gpu_layers,
        resolved_gpu_layers: summary.resolved_requested_gpu_layers,
    };
    let fitted = FitConfiguration {
        context_tokens: nonzero_option(summary.fitted_context_tokens),
        resolved_context_tokens: summary.resolved_fitted_context_tokens,
        gpu_layers: FitGpuLayers::from_raw(summary.fitted_gpu_layers),
        raw_gpu_layers: summary.fitted_gpu_layers,
        resolved_gpu_layers: summary.resolved_fitted_gpu_layers,
    };
    (requested, fitted)
}

fn decode_device(
    native: &NativeFitReport,
    index: usize,
) -> Result<FitDeviceEstimate, FitReportError> {
    let mut raw = MaybeUninit::<sys::llama_rs_fit_device>::uninit();
    if !unsafe { sys::llama_rs_fit_report_get_device(native.0.as_ptr(), index, raw.as_mut_ptr()) } {
        return Err(FitReportError::Malformed("missing device entry"));
    }
    let raw = unsafe { raw.assume_init() };
    let kind = match raw.kind {
        sys::LLAMA_RS_FIT_DEVICE_ACCELERATOR => FitDeviceKind::Accelerator,
        sys::LLAMA_RS_FIT_DEVICE_HOST => FitDeviceKind::Host,
        value => {
            return Err(FitReportError::UnknownEnum {
                kind: "fit device kind",
                value: i64::from(value),
            });
        }
    };
    let margin = raw.margin_applies.then_some(raw.margin_bytes);
    Ok(FitDeviceEstimate {
        index: raw.index,
        kind,
        backend_type: raw.backend_type,
        backend: borrowed_string(raw.backend, "devices[].backend")?,
        device_id: optional_borrowed_string(raw.device_id, "devices[].device_id")?,
        name: borrowed_string(raw.name, "devices[].name")?,
        description: borrowed_string(raw.description, "devices[].description")?,
        initial: raw
            .initial_available
            .then(|| decode_memory(raw.initial, kind, margin))
            .transpose()?,
        fitted: raw
            .fitted_available
            .then(|| decode_memory(raw.fitted, kind, margin))
            .transpose()?,
        margin_bytes: margin,
    })
}

fn decode_memory(
    raw: sys::llama_rs_fit_memory,
    kind: FitDeviceKind,
    margin: Option<u64>,
) -> Result<FitMemoryEstimate, FitReportError> {
    let total_bytes = raw.total_bytes;
    let free_bytes = raw.free_bytes;
    let allocation_total = raw
        .model_bytes
        .checked_add(raw.context_bytes)
        .and_then(|value| value.checked_add(raw.compute_bytes))
        .ok_or(FitReportError::ArithmeticOverflow {
            field: "known allocation total",
        })?;
    let target = margin
        .map(|margin_bytes| {
            let available_bytes = match kind {
                FitDeviceKind::Accelerator => free_bytes,
                FitDeviceKind::Host => total_bytes,
            };
            let max_allocation = i128::from(available_bytes) - i128::from(margin_bytes);
            let projected_remaining = i128::from(available_bytes) - i128::from(allocation_total);
            let shortfall = (i128::from(margin_bytes) - projected_remaining).max(0);
            Ok(FitMemoryTarget {
                available_bytes,
                margin_bytes,
                max_allocation_bytes: i64::try_from(max_allocation).map_err(|_| {
                    FitReportError::ArithmeticOverflow {
                        field: "maximum allocation",
                    }
                })?,
                projected_remaining_bytes: i64::try_from(projected_remaining).map_err(|_| {
                    FitReportError::ArithmeticOverflow {
                        field: "projected remaining memory",
                    }
                })?,
                shortfall_bytes: u64::try_from(shortfall).map_err(|_| {
                    FitReportError::ArithmeticOverflow {
                        field: "projected margin shortfall",
                    }
                })?,
            })
        })
        .transpose()?;
    Ok(FitMemoryEstimate {
        total_bytes,
        free_bytes,
        allocations: FitAllocations {
            model_bytes: raw.model_bytes,
            context_bytes: raw.context_bytes,
            compute_bytes: raw.compute_bytes,
            total_bytes: allocation_total,
        },
        target,
    })
}

fn decode_placement(
    native: &NativeFitReport,
    index: usize,
) -> Result<FitTensorPlacement, FitReportError> {
    let mut raw = MaybeUninit::<sys::llama_rs_fit_placement>::uninit();
    if !unsafe {
        sys::llama_rs_fit_report_get_placement(native.0.as_ptr(), index, raw.as_mut_ptr())
    } {
        return Err(FitReportError::Malformed("missing tensor placement entry"));
    }
    let raw = unsafe { raw.assume_init() };
    let target = match raw.kind {
        sys::LLAMA_RS_FIT_PLACEMENT_HOST => FitPlacementTarget::Host,
        sys::LLAMA_RS_FIT_PLACEMENT_DEVICE => FitPlacementTarget::Device {
            index: usize::try_from(raw.device_index).ok(),
            name: borrowed_string(raw.device_name, "tensor_placements[].target.name")?,
            description: borrowed_string(
                raw.device_description,
                "tensor_placements[].target.description",
            )?,
        },
        sys::LLAMA_RS_FIT_PLACEMENT_OTHER => FitPlacementTarget::Other,
        value => {
            return Err(FitReportError::UnknownEnum {
                kind: "fit placement kind",
                value: i64::from(value),
            });
        }
    };
    Ok(FitTensorPlacement {
        pattern: borrowed_string(raw.pattern, "tensor_placements[].pattern")?,
        buffer_type: borrowed_string(raw.buffer_type, "tensor_placements[].buffer_type")?,
        target,
    })
}

fn derive_adjustments(
    requested: FitConfiguration,
    fitted: FitConfiguration,
    tensor_split: &[f32],
    placement_count: usize,
) -> Vec<FitAdjustment> {
    let mut result = Vec::new();
    if fitted.resolved_context_tokens < requested.resolved_context_tokens {
        result.push(FitAdjustment::ContextReduced {
            from: requested.resolved_context_tokens,
            to: fitted.resolved_context_tokens,
        });
    }
    if fitted.resolved_gpu_layers < requested.resolved_gpu_layers {
        result.push(FitAdjustment::GpuLayersReduced {
            from: requested.resolved_gpu_layers,
            to: fitted.resolved_gpu_layers,
        });
    }
    if tensor_split.iter().any(|value| *value != 0.0) {
        result.push(FitAdjustment::TensorSplitApplied {
            values: tensor_split.to_vec(),
        });
    }
    if placement_count > 0 {
        result.push(FitAdjustment::TensorPlacementsApplied {
            count: placement_count,
        });
    }
    result
}

fn derive_warnings(
    status: FitStatus,
    accelerator_count: usize,
    initial_available: bool,
    fitted_available: bool,
    initial_error: Option<String>,
    fitted_error: Option<String>,
    devices: &[FitDeviceEstimate],
) -> Vec<FitWarning> {
    let mut result = Vec::new();
    if accelerator_count == 0 {
        result.push(FitWarning::NoAccelerator);
    } else {
        result.push(FitWarning::HostMemoryAssumedUnlimited);
    }
    if !initial_available {
        result.push(FitWarning::DiagnosticsUnavailable {
            phase: FitPhase::Initial,
            message: initial_error.unwrap_or_else(|| "initial measurement unavailable".to_owned()),
        });
    }
    if !fitted_available {
        result.push(FitWarning::DiagnosticsUnavailable {
            phase: FitPhase::Fitted,
            message: fitted_error.unwrap_or_else(|| "fitted measurement unavailable".to_owned()),
        });
    }
    for device in devices {
        let estimate = device.fitted.or(device.initial);
        if device.kind == FitDeviceKind::Accelerator
            && estimate.is_some_and(|memory| memory.total_bytes == 0 && memory.free_bytes == 0)
        {
            result.push(FitWarning::DeviceMemoryUnknown {
                device_index: device.index,
                device_name: device.name.clone(),
            });
        }
        if let Some(shortfall_bytes) = device
            .fitted
            .and_then(|memory| memory.target)
            .map(|target| target.shortfall_bytes)
            .filter(|shortfall| *shortfall > 0)
        {
            result.push(FitWarning::MarginNotMet {
                device_index: device.index,
                device_name: device.name.clone(),
                shortfall_bytes,
            });
        }
    }
    match status {
        FitStatus::Success => {}
        FitStatus::Failure => result.push(FitWarning::FitFailed),
        FitStatus::Error => result.push(FitWarning::FitError),
    }
    result
}

fn decode_status(value: sys::llama_rs_fit_status) -> Result<FitStatus, FitReportError> {
    match value {
        sys::LLAMA_RS_FIT_STATUS_SUCCESS => Ok(FitStatus::Success),
        sys::LLAMA_RS_FIT_STATUS_FAILURE => Ok(FitStatus::Failure),
        sys::LLAMA_RS_FIT_STATUS_ERROR => Ok(FitStatus::Error),
        value => Err(FitReportError::UnknownEnum {
            kind: "fit status",
            value: i64::from(value),
        }),
    }
}

fn nonzero_option(value: u32) -> Option<u32> {
    (value != 0).then_some(value)
}

fn validate_positive_finite(value: f64, field: &'static str) -> Result<(), FitReportError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(FitReportError::InvalidCalibrationNumber { field })
    }
}

fn validate_nonnegative_finite(value: f64, field: &'static str) -> Result<(), FitReportError> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(FitReportError::InvalidCalibrationNumber { field })
    }
}

fn borrowed_string(value: *const c_char, field: &'static str) -> Result<String, FitReportError> {
    optional_borrowed_string(value, field)?.ok_or(FitReportError::MissingString { field })
}

fn optional_borrowed_string(
    value: *const c_char,
    field: &'static str,
) -> Result<Option<String>, FitReportError> {
    if value.is_null() {
        return Ok(None);
    }
    let value = unsafe { CStr::from_ptr(value) };
    value
        .to_str()
        .map(|value| Some(value.to_owned()))
        .map_err(|_| FitReportError::InvalidUtf8 { field })
}

fn take_native_error(value: *mut c_char) -> String {
    if value.is_null() {
        return "native bridge returned no diagnostic".to_owned();
    }
    let message = unsafe { CStr::from_ptr(value).to_string_lossy().into_owned() };
    unsafe { sys::llama_rs_string_free(value) };
    message
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accelerator_target_uses_current_free_memory() {
        let estimate = decode_memory(
            sys::llama_rs_fit_memory {
                total_bytes: 10_000,
                free_bytes: 8_000,
                model_bytes: 4_000,
                context_bytes: 1_000,
                compute_bytes: 500,
            },
            FitDeviceKind::Accelerator,
            Some(1_000),
        )
        .expect("valid estimate");
        assert_eq!(
            estimate.target,
            Some(FitMemoryTarget {
                available_bytes: 8_000,
                margin_bytes: 1_000,
                max_allocation_bytes: 7_000,
                projected_remaining_bytes: 2_500,
                shortfall_bytes: 0,
            })
        );
    }

    #[test]
    fn cpu_only_target_matches_upstream_total_memory_rule() {
        let estimate = decode_memory(
            sys::llama_rs_fit_memory {
                total_bytes: 10_000,
                free_bytes: 2_000,
                model_bytes: 8_500,
                context_bytes: 1_000,
                compute_bytes: 750,
            },
            FitDeviceKind::Host,
            Some(1_000),
        )
        .expect("valid estimate");
        assert_eq!(estimate.target.expect("target").available_bytes, 10_000);
        assert_eq!(estimate.target.expect("target").shortfall_bytes, 1_250);
    }

    #[test]
    fn adjustments_are_derived_from_resolved_values() {
        let requested = FitConfiguration {
            context_tokens: None,
            resolved_context_tokens: 32_768,
            gpu_layers: FitGpuLayers::Auto,
            raw_gpu_layers: -1,
            resolved_gpu_layers: 29,
        };
        let fitted = FitConfiguration {
            context_tokens: Some(8_192),
            resolved_context_tokens: 8_192,
            gpu_layers: FitGpuLayers::Count(20),
            raw_gpu_layers: 20,
            resolved_gpu_layers: 20,
        };
        assert_eq!(
            derive_adjustments(requested, fitted, &[12.0, 8.0], 1),
            vec![
                FitAdjustment::ContextReduced {
                    from: 32_768,
                    to: 8_192,
                },
                FitAdjustment::GpuLayersReduced { from: 29, to: 20 },
                FitAdjustment::TensorSplitApplied {
                    values: vec![12.0, 8.0],
                },
                FitAdjustment::TensorPlacementsApplied { count: 1 },
            ]
        );
    }

    #[test]
    fn allocation_total_overflow_is_an_error() {
        let result = decode_memory(
            sys::llama_rs_fit_memory {
                total_bytes: i64::MAX,
                free_bytes: i64::MAX,
                model_bytes: u64::MAX,
                context_bytes: 1,
                compute_bytes: 0,
            },
            FitDeviceKind::Accelerator,
            None,
        );
        assert!(matches!(
            result,
            Err(FitReportError::ArithmeticOverflow {
                field: "known allocation total"
            })
        ));
    }

    #[test]
    fn report_strings_reject_missing_and_invalid_utf8() {
        assert!(matches!(
            borrowed_string(ptr::null(), "name"),
            Err(FitReportError::MissingString { field: "name" })
        ));

        let invalid = [0xff_u8, 0];
        assert!(matches!(
            borrowed_string(invalid.as_ptr().cast(), "name"),
            Err(FitReportError::InvalidUtf8 { field: "name" })
        ));
    }

    #[test]
    fn kv_row_bytes_are_both_present_or_both_absent() {
        assert!(decode_attention_workload(1, 1, 0, 0, false).is_ok());
        assert!(decode_attention_workload(1, 1, 8, 8, false).is_ok());
        assert!(decode_attention_workload(1, 1, 0, 8, false).is_err());
        assert!(decode_attention_workload(1, 1, 8, 0, false).is_err());
        assert!(decode_attention_workload(1, 1, 8, 0, true).is_ok());
        assert!(decode_attention_workload(1, 1, 8, 8, true).is_err());
    }

    #[test]
    fn calibration_projection_rejects_invalid_rates_and_identities() {
        let invalid_rate = FitCalibration {
            method: FIT_CALIBRATION_METHOD.to_owned(),
            metrics: vec![FitCalibrationMetric {
                backend_type: 1,
                backend: "CPU".to_owned(),
                device_id: None,
                tensor_type: 1,
                routed: false,
                bytes_per_second: f64::NAN,
                launch_microseconds: 0.0,
                relative_spread: 0.0,
                sample_count: 1,
                measured_microseconds: 1,
                stable: true,
            }],
            elapsed_microseconds: 1,
        };
        assert!(matches!(
            invalid_rate.validate(),
            Err(FitReportError::InvalidCalibrationNumber {
                field: "calibration.bytes_per_second"
            })
        ));

        let invalid_identity = FitCalibration {
            method: FIT_CALIBRATION_METHOD.to_owned(),
            metrics: vec![FitCalibrationMetric {
                bytes_per_second: 1.0,
                backend: "CPU\0unexpected".to_owned(),
                ..invalid_rate.metrics[0].clone()
            }],
            elapsed_microseconds: 1,
        };
        assert!(matches!(
            invalid_identity.validate(),
            Err(FitReportError::InvalidCalibrationString {
                field: "calibration.backend"
            })
        ));

        let missing_evidence = FitCalibration {
            method: FIT_CALIBRATION_METHOD.to_owned(),
            metrics: vec![FitCalibrationMetric {
                bytes_per_second: 1.0,
                sample_count: 0,
                ..invalid_rate.metrics[0].clone()
            }],
            elapsed_microseconds: 1,
        };
        assert!(matches!(
            missing_evidence.validate(),
            Err(FitReportError::Malformed(
                "calibration metric has no retained samples"
            ))
        ));
    }

    #[test]
    #[ignore = "runs bounded synthetic backend calibration"]
    fn native_calibration_is_model_free_finite_and_bounded() {
        let backend = crate::llama_backend::LlamaBackend::init()
            .expect("initialize native backend for synthetic calibration");
        let calibration =
            FitCalibration::measure(&backend).expect("run model-free synthetic calibration");
        assert_eq!(calibration.method, FIT_CALIBRATION_METHOD);
        assert!(!calibration.metrics.is_empty());
        assert!(calibration.elapsed_microseconds > 0);
        assert!(calibration.elapsed_microseconds <= 120_000_000);
        assert!(calibration.metrics.iter().any(|metric| !metric.routed));
        assert!(calibration.metrics.iter().any(|metric| metric.routed));
        for metric in calibration.metrics {
            assert!(!metric.backend.is_empty());
            assert!(metric.bytes_per_second.is_finite() && metric.bytes_per_second > 0.0);
            assert!(metric.launch_microseconds.is_finite() && metric.launch_microseconds >= 0.0);
            assert!(metric.relative_spread.is_finite() && metric.relative_spread >= 0.0);
            assert!((5..=11).contains(&metric.sample_count));
            assert!(metric.measured_microseconds > 0);
            if metric.stable {
                assert!(metric.relative_spread <= 0.05);
            }
        }
    }
}
