//! Safe multi-token prediction support.

use std::ffi::{c_char, CStr, CString};
use std::path::{Path, PathBuf};
use std::ptr;

use crate::context::params::LlamaContextParams;
use crate::model::params::LlamaModelParams;

pub use crate::speculative::{
    MtpOperations, MtpSpeculative as MtpSession, MtpSpeculativeError as MtpError,
    MtpSpeculativeParams as MtpParams,
};

/// Native parameters used to validate an MTP execution without allocating model tensors.
#[derive(Debug)]
pub struct MtpPreflightParams<'a> {
    /// Parameters that will load the target model.
    pub target_model: &'a LlamaModelParams,
    /// Parameters that will construct the target context.
    pub target_context: &'a LlamaContextParams,
    /// Parameters that will load a separate draft model, when supplied.
    pub draft_model: Option<&'a LlamaModelParams>,
    /// Parameters that will construct a separate draft context, when supplied.
    pub draft_context: Option<&'a LlamaContextParams>,
}

/// A successfully validated MTP artifact configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MtpPreflight {
}

/// Failure to validate an MTP artifact configuration.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum MtpPreflightError {
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
    #[error("native MTP context construction is unsupported")]
    ContextUnsupported,
    /// The native bridge failed before producing a semantic result.
    #[error("native MTP preflight failed with status {status}: {message}")]
    Native {
        /// Raw bridge status retained for diagnostics.
        status: i32,
        /// Exception-safe native diagnostic.
        message: String,
    },
}

/// Validate the exact target and optional separate draft artifacts by constructing linked no-alloc
/// contexts and the same native speculative controller used during serving.
pub fn preflight_mtp(
    target_path: &Path,
    draft_path: Option<&Path>,
    params: &MtpPreflightParams<'_>,
) -> Result<MtpPreflight, MtpPreflightError> {
    let target = path_string(target_path)?;
    let draft = draft_path.map(path_string).transpose()?;
    let draft_params = match (draft.as_ref(), params.draft_model, params.draft_context) {
        (None, None, None) => None,
        (Some(_), Some(model), Some(context)) => Some((model, context)),
        _ => return Err(MtpPreflightError::InvalidDraftParameters),
    };

    let mut result = llama_cpp_sys_2::llama_rs_mtp_preflight_result {
        code: llama_cpp_sys_2::LLAMA_RS_MTP_PREFLIGHT_CONTEXT_UNSUPPORTED,
    };
    let mut error: *mut c_char = ptr::null_mut();
    let (draft_model, draft_context) = draft_params
        .map(|(model, context)| (&raw const model.params, &raw const context.context_params))
        .unwrap_or((ptr::null(), ptr::null()));
    let status = unsafe {
        llama_cpp_sys_2::llama_rs_mtp_preflight(
            target.as_ptr(),
            draft.as_ref().map_or(ptr::null(), |path| path.as_ptr()),
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
        return Err(MtpPreflightError::Native { status, message });
    }
    if !error.is_null() {
        unsafe { llama_cpp_sys_2::llama_rs_string_free(error) };
    }

    match result.code {
        llama_cpp_sys_2::LLAMA_RS_MTP_PREFLIGHT_SUPPORTED => Ok(MtpPreflight {}),
        llama_cpp_sys_2::LLAMA_RS_MTP_PREFLIGHT_CONTEXT_UNSUPPORTED => {
            Err(MtpPreflightError::ContextUnsupported)
        }
        _ => Err(MtpPreflightError::Native {
            status: llama_cpp_sys_2::LLAMA_RS_STATUS_INVALID_STATE,
            message: "native MTP preflight returned an unknown result".into(),
        }),
    }
}

fn path_string(path: &Path) -> Result<CString, MtpPreflightError> {
    let value = path
        .to_str()
        .ok_or_else(|| MtpPreflightError::InvalidPath(path.to_path_buf()))?;
    CString::new(value).map_err(|_| MtpPreflightError::InvalidPathString)
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
