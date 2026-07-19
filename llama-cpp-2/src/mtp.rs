//! Safe multi-token prediction support.

use std::ffi::{c_char, CStr, CString};
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::ptr;

use crate::context::params::LlamaContextParams;
use crate::model::params::LlamaModelParams;
use crate::model::LlamaModel;

pub use crate::speculative::{
    MtpOperations, MtpSpeculative as MtpSession, MtpSpeculativeError as MtpError,
    MtpSpeculativeParams as MtpParams,
};

/// How a loaded model participates in multi-token prediction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MtpModelKind {
    /// The target and prediction layers are stored in one model.
    Bundled,
    /// The model is a separate prediction model linked to a target context.
    Draft,
}

/// Executable MTP capability reported by the loaded native model implementation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MtpModelInfo {
    /// Artifact role implemented by this model.
    pub kind: MtpModelKind,
    /// Number of prediction layers used by the native graph.
    pub prediction_layers: NonZeroU32,
    /// Whether context construction requires a separate target context.
    pub requires_target_context: bool,
}

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
    /// Native artifact role selected by the preflight.
    pub model: MtpModelInfo,
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
    /// The target contains no executable bundled MTP implementation.
    #[error("model contains no executable MTP implementation")]
    NoMtp,
    /// The target path identifies a draft-only model.
    #[error("target artifact is an MTP draft model")]
    TargetIsDraft,
    /// The separate artifact does not implement a supported MTP draft graph.
    #[error("artifact is not a supported MTP draft model")]
    DraftNotSupported,
    /// Target and draft token vocabularies differ.
    #[error("target and MTP draft vocabularies differ")]
    VocabularyMismatch,
    /// Target output and draft input widths differ.
    #[error("target output and MTP draft input widths differ")]
    EmbeddingMismatch,
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

/// Inspect one GGUF's executable native MTP role without allocating model tensors.
///
/// `Ok(None)` means the native model implementation exposes no MTP graph for the artifact.
pub fn inspect_mtp_model(
    path: &Path,
    model_params: &LlamaModelParams,
) -> Result<Option<MtpModelInfo>, MtpPreflightError> {
    let path = path_string(path)?;
    let mut result = llama_cpp_sys_2::llama_rs_mtp_model_info_result {
        kind: llama_cpp_sys_2::LLAMA_MTP_MODEL_NONE,
        prediction_layers: 0,
        requires_target_context: false,
    };
    let mut error: *mut c_char = ptr::null_mut();
    let status = unsafe {
        llama_cpp_sys_2::llama_rs_mtp_model_info_from_file(
            path.as_ptr(),
            &raw const model_params.params,
            &raw mut result,
            &raw mut error,
        )
    };
    if status != llama_cpp_sys_2::LLAMA_RS_STATUS_OK {
        return Err(MtpPreflightError::Native {
            status,
            message: take_native_error(error),
        });
    }
    if !error.is_null() {
        unsafe { llama_cpp_sys_2::llama_rs_string_free(error) };
    }

    model_info(
        result.kind,
        result.prediction_layers,
        result.requires_target_context,
    )
}

/// Validate the exact target and optional separate draft artifacts without allocating model
/// tensors or serving contexts.
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
        kind: llama_cpp_sys_2::LLAMA_MTP_MODEL_NONE,
        prediction_layers: 0,
        requires_target_context: false,
        code: llama_cpp_sys_2::LLAMA_RS_MTP_PREFLIGHT_NO_MTP,
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
        llama_cpp_sys_2::LLAMA_RS_MTP_PREFLIGHT_SUPPORTED => {
            let model = model_info(
                result.kind,
                result.prediction_layers,
                result.requires_target_context,
            )?
            .ok_or_else(|| MtpPreflightError::Native {
                status: llama_cpp_sys_2::LLAMA_RS_STATUS_INVALID_STATE,
                message: "native MTP preflight returned no model capability".into(),
            })?;
            Ok(MtpPreflight { model })
        }
        llama_cpp_sys_2::LLAMA_RS_MTP_PREFLIGHT_NO_MTP => Err(MtpPreflightError::NoMtp),
        llama_cpp_sys_2::LLAMA_RS_MTP_PREFLIGHT_TARGET_IS_DRAFT => {
            Err(MtpPreflightError::TargetIsDraft)
        }
        llama_cpp_sys_2::LLAMA_RS_MTP_PREFLIGHT_DRAFT_NOT_SUPPORTED => {
            Err(MtpPreflightError::DraftNotSupported)
        }
        llama_cpp_sys_2::LLAMA_RS_MTP_PREFLIGHT_VOCABULARY_MISMATCH => {
            Err(MtpPreflightError::VocabularyMismatch)
        }
        llama_cpp_sys_2::LLAMA_RS_MTP_PREFLIGHT_EMBEDDING_MISMATCH => {
            Err(MtpPreflightError::EmbeddingMismatch)
        }
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

fn model_info(
    kind: llama_cpp_sys_2::llama_mtp_model_kind,
    prediction_layers: u32,
    requires_target_context: bool,
) -> Result<Option<MtpModelInfo>, MtpPreflightError> {
    if kind == llama_cpp_sys_2::LLAMA_MTP_MODEL_NONE {
        return if prediction_layers == 0 {
            Ok(None)
        } else {
            Err(MtpPreflightError::Native {
                status: llama_cpp_sys_2::LLAMA_RS_STATUS_INVALID_STATE,
                message: "native non-MTP result returned prediction layers".into(),
            })
        };
    }
    let prediction_layers =
        NonZeroU32::new(prediction_layers).ok_or_else(|| MtpPreflightError::Native {
            status: llama_cpp_sys_2::LLAMA_RS_STATUS_INVALID_STATE,
            message: "native MTP result returned zero prediction layers".into(),
        })?;
    let kind = match kind {
        llama_cpp_sys_2::LLAMA_MTP_MODEL_BUNDLED => MtpModelKind::Bundled,
        llama_cpp_sys_2::LLAMA_MTP_MODEL_DRAFT => MtpModelKind::Draft,
        _ => {
            return Err(MtpPreflightError::Native {
                status: llama_cpp_sys_2::LLAMA_RS_STATUS_INVALID_STATE,
                message: "native MTP result returned an invalid model kind".into(),
            });
        }
    };
    Ok(Some(MtpModelInfo {
        kind,
        prediction_layers,
        requires_target_context,
    }))
}

impl LlamaModel {
    /// Return executable MTP capability for this loaded model and tensor topology.
    #[must_use]
    pub fn mtp_info(&self) -> Option<MtpModelInfo> {
        let info = unsafe { llama_cpp_sys_2::llama_model_mtp_info(self.model.as_ptr()) };
        model_info(
            info.kind,
            info.prediction_layers,
            info.requires_target_context,
        )
        .ok()
        .flatten()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_none_maps_to_no_capability() {
        assert_eq!(
            model_info(llama_cpp_sys_2::LLAMA_MTP_MODEL_NONE, 0, false).unwrap(),
            None
        );
    }

    #[test]
    fn executable_capability_requires_prediction_layers() {
        let error = model_info(llama_cpp_sys_2::LLAMA_MTP_MODEL_BUNDLED, 0, false).unwrap_err();
        assert!(matches!(error, MtpPreflightError::Native { .. }));
    }

    #[test]
    fn bundled_capability_is_typed() {
        let info = model_info(llama_cpp_sys_2::LLAMA_MTP_MODEL_BUNDLED, 2, false)
            .unwrap()
            .unwrap();
        assert_eq!(info.kind, MtpModelKind::Bundled);
        assert_eq!(info.prediction_layers.get(), 2);
        assert!(!info.requires_target_context);
    }
}
