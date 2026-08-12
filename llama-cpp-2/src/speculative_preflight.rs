//! Native speculative-decoding preflight.

use std::ffi::{c_char, CStr, CString};
use std::path::{Path, PathBuf};
use std::ptr;

use crate::context::params::LlamaContextParams;
use crate::model::params::LlamaModelParams;
use crate::speculative::SpeculativeMethod;

/// Native parameters used to validate an speculative execution without allocating model tensors.
#[derive(Debug)]
pub struct SpeculativePreflightParams<'a> {
    /// Selected speculative-decoding method and method-specific threshold.
    pub method: SpeculativeMethod,
    /// Requested maximum number of draft tokens.
    pub n_max: i32,
    /// Requested minimum number of draft tokens.
    pub n_min: i32,
    /// Parameters that will load the target model.
    pub target_model: &'a LlamaModelParams,
    /// Parameters that will construct the target context.
    pub target_context: &'a LlamaContextParams,
    /// Parameters that will load a separate draft model, when supplied.
    pub draft_model: Option<&'a LlamaModelParams>,
    /// Parameters that will construct a separate draft context, when supplied.
    pub draft_context: Option<&'a LlamaContextParams>,
}

/// A successfully validated speculative artifact configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpeculativePreflight {
    /// Artifact-bounded maximum number of draft tokens used at runtime.
    pub effective_n_max: i32,
    /// Artifact-bounded minimum number of draft tokens used at runtime.
    pub effective_n_min: i32,
}

/// Failure to validate an speculative artifact configuration.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum SpeculativePreflightError {
    /// A path cannot be represented by llama.cpp's string API.
    #[error("model path is not valid UTF-8: {0}")]
    InvalidPath(PathBuf),
    /// A path contains an interior null byte.
    #[error("model path contains an interior null byte")]
    InvalidPathString,
    /// Draft parameters and the optional draft path disagree.
    #[error("draft model and context parameters must be supplied together with a draft path")]
    InvalidDraftParameters,
    /// The native linked context cannot be constructed with these parameters.
    #[error("native speculative context construction is unsupported")]
    ContextUnsupported,
    /// The selected method does not match the draft artifact's trained heads.
    #[error("selected speculative method does not match the draft artifact")]
    MethodUnsupported,
    /// The native bridge failed before producing a semantic result.
    #[error("native speculative preflight failed with status {status}: {message}")]
    Native {
        /// Raw bridge status retained for diagnostics.
        status: i32,
        /// Exception-safe native diagnostic.
        message: String,
    },
}

/// Validate the exact target and optional separate draft artifacts by constructing linked no-alloc
/// contexts and the same native speculative controller used during serving.
pub fn preflight_speculative(
    target_path: &Path,
    draft_path: Option<&Path>,
    params: &SpeculativePreflightParams<'_>,
) -> Result<SpeculativePreflight, SpeculativePreflightError> {
    let target = path_string(target_path)?;
    let draft = draft_path.map(path_string).transpose()?;
    let draft_params = match (draft.as_ref(), params.draft_model, params.draft_context) {
        (None, None, None) => None,
        (Some(_), Some(model), Some(context)) => Some((model, context)),
        _ => return Err(SpeculativePreflightError::InvalidDraftParameters),
    };

    let mut result = llama_cpp_sys_2::llama_rs_speculative_preflight_result {
        code: llama_cpp_sys_2::LLAMA_RS_SPECULATIVE_PREFLIGHT_CONTEXT_UNSUPPORTED,
        effective_n_max: 0,
        effective_n_min: 0,
    };
    let mut error: *mut c_char = ptr::null_mut();
    let (draft_model, draft_context) = draft_params
        .map(|(model, context)| (&raw const model.params, &raw const context.context_params))
        .unwrap_or((ptr::null(), ptr::null()));
    let status = unsafe {
        llama_cpp_sys_2::llama_rs_speculative_preflight(
            target.as_ptr(),
            draft.as_ref().map_or(ptr::null(), |path| path.as_ptr()),
            params.method.native(),
            params.n_max,
            params.n_min,
            params.method.threshold(),
            &raw const params.target_model.params,
            &raw const params.target_context.context_params,
            draft_model,
            draft_context,
            &raw mut result,
            &raw mut error,
        )
    };
    if status != llama_cpp_sys_2::LLAMA_RS_STATUS_OK {
        let message = take_native_error(error);
        return Err(SpeculativePreflightError::Native { status, message });
    }
    if !error.is_null() {
        unsafe { llama_cpp_sys_2::llama_rs_string_free(error) };
    }

    match result.code {
        llama_cpp_sys_2::LLAMA_RS_SPECULATIVE_PREFLIGHT_SUPPORTED => Ok(SpeculativePreflight {
            effective_n_max: result.effective_n_max,
            effective_n_min: result.effective_n_min,
        }),
        llama_cpp_sys_2::LLAMA_RS_SPECULATIVE_PREFLIGHT_CONTEXT_UNSUPPORTED => {
            Err(SpeculativePreflightError::ContextUnsupported)
        }
        llama_cpp_sys_2::LLAMA_RS_SPECULATIVE_PREFLIGHT_METHOD_UNSUPPORTED => {
            Err(SpeculativePreflightError::MethodUnsupported)
        }
        _ => Err(SpeculativePreflightError::Native {
            status: llama_cpp_sys_2::LLAMA_RS_STATUS_INVALID_STATE,
            message: "native speculative preflight returned an unknown result".into(),
        }),
    }
}

fn path_string(path: &Path) -> Result<CString, SpeculativePreflightError> {
    let value = path
        .to_str()
        .ok_or_else(|| SpeculativePreflightError::InvalidPath(path.to_path_buf()))?;
    CString::new(value).map_err(|_| SpeculativePreflightError::InvalidPathString)
}

fn take_native_error(error: *mut c_char) -> String {
    if error.is_null() {
        return String::new();
    }
    let message = unsafe { CStr::from_ptr(error) }
        .to_string_lossy()
        .into_owned();
    unsafe { llama_cpp_sys_2::llama_rs_string_free(error) };
    message
}
