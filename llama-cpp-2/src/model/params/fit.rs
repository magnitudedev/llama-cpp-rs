//! Typed diagnostics for llama.cpp's `common/fit` estimator.

use std::ffi::{c_char, CStr};
use std::mem::MaybeUninit;
use std::pin::Pin;
use std::ptr::{self, NonNull};

use crate::context::params::LlamaContextParams;
use crate::model::params::LlamaModelParams;

use llama_cpp_sys_2 as sys;

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
        value: i32,
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

impl LlamaModelParams {
    /// Measure several execution contexts while constructing the no-allocation model once.
    ///
    /// This is an exact projection of llama.cpp model/context graphs. It does not run
    /// `common_fit_params` or alter the supplied parameters.
    pub fn measure_contexts(
        &self,
        model_path: &CStr,
        contexts: &[LlamaContextParams],
        margins: &[usize],
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
        self.fit_params_report_impl(model_path, cparams, None, margins, n_ctx_min)
    }

    /// Fit an MTP model/context while a no-allocation target context is linked as `ctx_other`.
    ///
    /// The returned report contains only the fitted model/context allocations. Compose it with a
    /// separate target report to assess the full execution plan.
    pub fn fit_params_report_linked(
        self: Pin<&mut Self>,
        model_path: &CStr,
        cparams: &mut LlamaContextParams,
        target: LinkedFitTarget<'_>,
        margins: &mut [usize],
        n_ctx_min: u32,
    ) -> Result<FitReport, FitReportError> {
        self.fit_params_report_impl(model_path, cparams, Some(target), margins, n_ctx_min)
    }

    fn fit_params_report_impl(
        mut self: Pin<&mut Self>,
        model_path: &CStr,
        cparams: &mut LlamaContextParams,
        linked_target: Option<LinkedFitTarget<'_>>,
        margins: &mut [usize],
        n_ctx_min: u32,
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
        elapsed_microseconds: summary.elapsed_microseconds.max(0).cast_unsigned(),
    })
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
                value: value.cast_signed(),
            });
        }
    };
    let margin = raw.margin_applies.then_some(raw.margin_bytes);
    Ok(FitDeviceEstimate {
        index: raw.index,
        kind,
        backend_type: raw.backend_type,
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
                value: value.cast_signed(),
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
            value: value.cast_signed(),
        }),
    }
}

fn nonzero_option(value: u32) -> Option<u32> {
    (value != 0).then_some(value)
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
}
