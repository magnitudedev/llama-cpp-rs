//! Typed snapshots of llama.cpp model, context, and compute allocations.

use std::ffi::{c_char, CStr};
use std::mem::MaybeUninit;
use std::ptr::{self, NonNull};
use std::slice;
use std::str;

use llama_cpp_sys_2 as sys;

use super::LlamaContext;

/// Allocations attributed by llama.cpp to one device or backend buffer type.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct ContextMemoryAllocations {
    /// Model tensor allocation.
    pub model_bytes: u64,
    /// Context and KV allocation.
    pub context_bytes: u64,
    /// Temporary compute-buffer allocation.
    pub compute_bytes: u64,
    /// Checked sum of model, context, and compute allocations.
    pub total_bytes: u64,
}

/// Allocations and device-memory snapshot for one backend device.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct ContextDeviceMemory {
    /// Index in the model's native device list, when this is a model device.
    ///
    /// Host memory and devices discovered only through a buffer type have no model-device index.
    pub device_index: Option<usize>,
    /// Raw `ggml_backend_dev_type` value from the pinned backend.
    pub backend_type: i32,
    /// Backend device name.
    pub name: String,
    /// Human-readable backend description.
    pub description: String,
    /// Device-reported total memory, when the backend reports a usable budget.
    pub total_bytes: Option<u64>,
    /// Device-reported free memory, when the backend reports a usable budget.
    pub free_bytes: Option<u64>,
    /// Allocations attributed to this context and its model.
    pub allocations: ContextMemoryAllocations,
    /// Device-reported used memory minus the allocations attributed by this report.
    ///
    /// This is deliberately not called overhead: it can include other processes, other contexts,
    /// driver allocations, and sampling-time changes in the backend memory counters.
    pub unaccounted_device_bytes: Option<i64>,
}

/// Allocations associated with a buffer type that cannot be assigned to a concrete device.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct BufferTypeMemory {
    /// Native backend buffer-type name.
    pub name: String,
    /// Allocations attributed to this buffer type.
    pub allocations: ContextMemoryAllocations,
}

/// Owned snapshot of llama.cpp's live memory breakdown.
///
/// This mirrors the pinned `llama_get_memory_breakdown` staging API. That upstream map does not
/// include the context's private logits/embeddings output buffer, so this report must not be used
/// as an exact process- or context-allocation total. In particular, `n_outputs_max` affects memory
/// that is outside this report.
#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct ContextMemoryReport {
    /// Model devices in native order, followed by unmatched devices and host memory.
    pub devices: Vec<ContextDeviceMemory>,
    /// Buffer types that could not be associated with a device, sorted by name.
    pub other_buffer_types: Vec<BufferTypeMemory>,
}

/// Failure to acquire or safely decode a native memory report.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum ContextMemoryReportError {
    /// The native bridge rejected the operation or caught a C++ exception.
    #[error("llama.cpp context memory bridge failed with status {status}: {message}")]
    Native {
        /// Raw `llama_rs_status` value.
        status: i32,
        /// Native diagnostic.
        message: String,
    },
    /// The bridge violated its pointer/count or ownership contract.
    #[error("malformed llama.cpp context memory report: {0}")]
    Malformed(&'static str),
    /// A native name or description was not valid UTF-8.
    #[error("invalid UTF-8 in context memory {field} at byte {valid_up_to}")]
    InvalidUtf8 {
        /// Projection field containing invalid bytes.
        field: &'static str,
        /// Valid prefix length reported by Rust's UTF-8 validator.
        valid_up_to: usize,
    },
    /// A derived byte count could not be represented by the public DTO.
    #[error("context memory arithmetic overflow while calculating {field}")]
    ArithmeticOverflow {
        /// Derived value that overflowed.
        field: &'static str,
    },
    /// Rust could not reserve space for a native collection projection.
    #[error("could not reserve {requested} entries for context memory {collection}")]
    RustAllocation {
        /// Collection being decoded.
        collection: &'static str,
        /// Native element count.
        requested: usize,
    },
}

struct NativeContextMemoryReport(NonNull<sys::llama_rs_context_memory_report>);

impl Drop for NativeContextMemoryReport {
    fn drop(&mut self) {
        unsafe { sys::llama_rs_context_memory_report_free(self.0.as_ptr()) };
    }
}

impl LlamaContext<'_> {
    /// Capture a typed snapshot of model, KV/context, and compute-buffer allocations.
    ///
    /// Device total/free values are sampled while the report is created and may change
    /// immediately. The context must not be decoded concurrently with this call. The pinned native
    /// breakdown omits the private logits/embeddings output buffer; see [`ContextMemoryReport`].
    ///
    /// # Errors
    ///
    /// Returns [`ContextMemoryReportError`] when llama.cpp rejects the request, the bridge returns
    /// malformed data, native strings are not UTF-8, or a derived total exceeds its public type.
    pub fn memory_report(&self) -> Result<ContextMemoryReport, ContextMemoryReportError> {
        let mut native_report = ptr::null_mut();
        let mut native_error: *mut c_char = ptr::null_mut();
        let status = unsafe {
            sys::llama_rs_context_memory_report_create(
                self.context.as_ptr(),
                &raw mut native_report,
                &raw mut native_error,
            )
        };
        if status != sys::LLAMA_RS_STATUS_OK {
            return Err(ContextMemoryReportError::Native {
                status,
                message: take_native_error(native_error),
            });
        }
        if !native_error.is_null() {
            unsafe { sys::llama_rs_string_free(native_error) };
        }
        let native = NativeContextMemoryReport(
            NonNull::new(native_report)
                .ok_or(ContextMemoryReportError::Malformed("null report owner"))?,
        );
        decode_report(&native)
    }
}

fn decode_report(
    native: &NativeContextMemoryReport,
) -> Result<ContextMemoryReport, ContextMemoryReportError> {
    let device_count =
        unsafe { sys::llama_rs_context_memory_report_device_count(native.0.as_ptr()) };
    let mut devices = Vec::new();
    devices.try_reserve_exact(device_count).map_err(|_| {
        ContextMemoryReportError::RustAllocation {
            collection: "devices",
            requested: device_count,
        }
    })?;
    for index in 0..device_count {
        devices.push(decode_device(native, index)?);
    }

    let other_count = unsafe { sys::llama_rs_context_memory_report_other_count(native.0.as_ptr()) };
    let mut other_buffer_types = Vec::new();
    other_buffer_types
        .try_reserve_exact(other_count)
        .map_err(|_| ContextMemoryReportError::RustAllocation {
            collection: "other buffer types",
            requested: other_count,
        })?;
    for index in 0..other_count {
        other_buffer_types.push(decode_other(native, index)?);
    }
    Ok(ContextMemoryReport {
        devices,
        other_buffer_types,
    })
}

fn decode_device(
    native: &NativeContextMemoryReport,
    index: usize,
) -> Result<ContextDeviceMemory, ContextMemoryReportError> {
    let mut raw = MaybeUninit::<sys::llama_rs_context_device_memory>::uninit();
    call_accessor(
        unsafe {
            sys::llama_rs_context_memory_report_device_get(
                native.0.as_ptr(),
                index,
                raw.as_mut_ptr(),
                ptr::null_mut(),
            )
        },
        "missing device entry",
    )?;
    let raw = unsafe { raw.assume_init() };
    let allocations = decode_allocations(raw.allocations, "device allocation total")?;
    let total_bytes = raw.has_total_bytes.then_some(raw.total_bytes);
    let free_bytes = raw.has_free_bytes.then_some(raw.free_bytes);
    Ok(ContextDeviceMemory {
        device_index: raw.has_device_index.then_some(raw.device_index),
        backend_type: raw.backend_type,
        name: decode_view(raw.name, "device name")?,
        description: decode_view(raw.description, "device description")?,
        total_bytes,
        free_bytes,
        allocations,
        unaccounted_device_bytes: derive_unaccounted(
            total_bytes,
            free_bytes,
            allocations.total_bytes,
        )?,
    })
}

fn decode_other(
    native: &NativeContextMemoryReport,
    index: usize,
) -> Result<BufferTypeMemory, ContextMemoryReportError> {
    let mut raw = MaybeUninit::<sys::llama_rs_buffer_type_memory>::uninit();
    call_accessor(
        unsafe {
            sys::llama_rs_context_memory_report_other_get(
                native.0.as_ptr(),
                index,
                raw.as_mut_ptr(),
                ptr::null_mut(),
            )
        },
        "missing buffer-type entry",
    )?;
    let raw = unsafe { raw.assume_init() };
    Ok(BufferTypeMemory {
        name: decode_view(raw.name, "buffer type name")?,
        allocations: decode_allocations(raw.allocations, "buffer-type allocation total")?,
    })
}

fn call_accessor(
    status: sys::llama_rs_status,
    malformed: &'static str,
) -> Result<(), ContextMemoryReportError> {
    if status == sys::LLAMA_RS_STATUS_OK {
        Ok(())
    } else {
        Err(ContextMemoryReportError::Malformed(malformed))
    }
}

fn decode_allocations(
    raw: sys::llama_rs_context_memory_values,
    field: &'static str,
) -> Result<ContextMemoryAllocations, ContextMemoryReportError> {
    let total_bytes = raw
        .model_bytes
        .checked_add(raw.context_bytes)
        .and_then(|value| value.checked_add(raw.compute_bytes))
        .ok_or(ContextMemoryReportError::ArithmeticOverflow { field })?;
    Ok(ContextMemoryAllocations {
        model_bytes: raw.model_bytes,
        context_bytes: raw.context_bytes,
        compute_bytes: raw.compute_bytes,
        total_bytes,
    })
}

fn derive_unaccounted(
    total_bytes: Option<u64>,
    free_bytes: Option<u64>,
    attributed_bytes: u64,
) -> Result<Option<i64>, ContextMemoryReportError> {
    let (Some(total_bytes), Some(free_bytes)) = (total_bytes, free_bytes) else {
        return Ok(None);
    };
    let value = i128::from(total_bytes) - i128::from(free_bytes) - i128::from(attributed_bytes);
    i64::try_from(value)
        .map(Some)
        .map_err(|_| ContextMemoryReportError::ArithmeticOverflow {
            field: "unaccounted device bytes",
        })
}

fn decode_view(
    view: sys::llama_rs_bytes_view,
    field: &'static str,
) -> Result<String, ContextMemoryReportError> {
    if view.len == 0 {
        return Ok(String::new());
    }
    if view.data.is_null() {
        return Err(ContextMemoryReportError::Malformed(
            "non-empty byte view has a null pointer",
        ));
    }
    let bytes = unsafe { slice::from_raw_parts(view.data, view.len) };
    str::from_utf8(bytes).map(str::to_owned).map_err(|error| {
        ContextMemoryReportError::InvalidUtf8 {
            field,
            valid_up_to: error.valid_up_to(),
        }
    })
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
    fn allocation_total_is_checked() {
        let result = decode_allocations(
            sys::llama_rs_context_memory_values {
                model_bytes: u64::MAX,
                context_bytes: 1,
                compute_bytes: 0,
            },
            "test total",
        );
        assert_eq!(
            result,
            Err(ContextMemoryReportError::ArithmeticOverflow {
                field: "test total"
            })
        );
    }

    #[test]
    fn unaccounted_memory_can_be_positive_or_negative() {
        assert_eq!(derive_unaccounted(Some(100), Some(40), 25), Ok(Some(35)));
        assert_eq!(derive_unaccounted(Some(100), Some(90), 25), Ok(Some(-15)));
        assert_eq!(derive_unaccounted(None, Some(90), 25), Ok(None));
    }

    #[test]
    fn empty_null_byte_view_is_safe() {
        assert_eq!(
            decode_view(
                sys::llama_rs_bytes_view {
                    data: ptr::null(),
                    len: 0,
                },
                "test",
            ),
            Ok(String::new())
        );
    }

    #[test]
    fn nonempty_null_byte_view_is_rejected() {
        assert!(matches!(
            decode_view(
                sys::llama_rs_bytes_view {
                    data: ptr::null(),
                    len: 1,
                },
                "test",
            ),
            Err(ContextMemoryReportError::Malformed(_))
        ));
    }

    #[test]
    fn native_constructor_rejects_null_context_transactionally() {
        let mut report = NonNull::<sys::llama_rs_context_memory_report>::dangling().as_ptr();
        let mut error = ptr::null_mut();
        let status = unsafe {
            sys::llama_rs_context_memory_report_create(ptr::null(), &raw mut report, &raw mut error)
        };
        assert_eq!(status, sys::LLAMA_RS_STATUS_INVALID_ARGUMENT);
        assert!(report.is_null());
        assert!(!error.is_null());
        unsafe { sys::llama_rs_string_free(error) };
    }

    #[test]
    fn native_accessors_reject_missing_owner() {
        let mut device = MaybeUninit::<sys::llama_rs_context_device_memory>::uninit();
        let mut buffer = MaybeUninit::<sys::llama_rs_buffer_type_memory>::uninit();
        let device_status = unsafe {
            sys::llama_rs_context_memory_report_device_get(
                ptr::null(),
                0,
                device.as_mut_ptr(),
                ptr::null_mut(),
            )
        };
        let buffer_status = unsafe {
            sys::llama_rs_context_memory_report_other_get(
                ptr::null(),
                0,
                buffer.as_mut_ptr(),
                ptr::null_mut(),
            )
        };
        assert_eq!(device_status, sys::LLAMA_RS_STATUS_INVALID_ARGUMENT);
        assert_eq!(buffer_status, sys::LLAMA_RS_STATUS_INVALID_ARGUMENT);
    }
}
