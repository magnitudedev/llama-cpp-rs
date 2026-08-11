//! Experimental wrappers for llama.cpp speculative decoding helpers.

use std::ffi::{c_char, CStr};
use std::ptr::{self, NonNull};

use crate::context::params::LlamaContextParams;
use crate::context::LlamaContext;
use crate::llama_backend::LlamaBackend;
use crate::llama_batch::LlamaBatch;
use crate::model::LlamaModel;
use crate::status_is_ok;
use crate::token::LlamaToken;

/// Parameters for same-model MTP speculative decoding.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MtpSpeculativeParams {
    /// Maximum number of draft tokens to propose.
    pub n_max: i32,
    /// Minimum number of draft tokens required before returning a draft.
    pub n_min: i32,
    /// Minimum draft probability accepted by llama.cpp's MTP drafter.
    pub p_min: f32,
}

impl Default for MtpSpeculativeParams {
    fn default() -> Self {
        Self {
            n_max: 3,
            n_min: 0,
            p_min: 0.0,
        }
    }
}

/// Result of resolving one target verification pass.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MtpVerificationResolution {
    /// Target and draft state were restored; verify the accepted prefix again.
    Replay,
    /// The accepted prefix was committed to the native speculative state.
    Committed,
}

/// Errors returned by the MTP speculative wrapper.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum MtpSpeculativeError {
    /// Invalid parameters were provided.
    #[error("invalid MTP speculative parameters")]
    InvalidParams,
    /// llama.cpp returned a null speculative handle.
    #[error("llama.cpp failed to initialize MTP speculative decoding")]
    InitFailed,
    /// llama.cpp rejected a wrapper call.
    #[error("llama.cpp MTP {operation} failed with status {status}: {message}")]
    Native {
        /// Operation that failed.
        operation: &'static str,
        /// Native bridge status.
        status: i32,
        /// Native diagnostic text.
        message: String,
    },
    /// The draft output exceeded the caller-provided bound.
    #[error("llama.cpp MTP draft exceeded configured maximum")]
    DraftOverflow,
}

/// RAII owner for a same-model MTP speculative context.
///
/// The handle owns independent native draft state for every configured target
/// sequence while sharing the target and MTP contexts.
#[derive(Debug)]
pub struct MtpSpeculative<'model> {
    raw: NonNull<llama_cpp_sys_2::llama_rs_mtp_speculative>,
    target_context: LlamaContext<'model>,
    draft_context: LlamaContext<'model>,
    n_max: usize,
    n_seq: u32,
}

/// Mutable speculative operations split from the owned target context.
#[derive(Debug)]
pub struct MtpOperations<'a> {
    raw: &'a mut NonNull<llama_cpp_sys_2::llama_rs_mtp_speculative>,
    n_max: usize,
    n_seq: u32,
}

impl<'model> MtpSpeculative<'model> {
    /// Construct and own the linked MTP context together with the target.
    pub fn new_linked(
        target_context: LlamaContext<'model>,
        draft_model: &'model LlamaModel,
        backend: &LlamaBackend,
        draft_context_params: LlamaContextParams,
        params: MtpSpeculativeParams,
        n_seq: u32,
    ) -> Result<Self, MtpSpeculativeError> {
        let draft_context = draft_model
            .new_mtp_context_linked(backend, draft_context_params, &target_context)
            .map_err(|_| MtpSpeculativeError::InitFailed)?;
        Self::new(target_context, draft_context, params, n_seq)
    }

    /// Create a new MTP speculative helper from a target context and an MTP
    /// draft context.
    ///
    /// # Errors
    ///
    /// Returns an error if parameters are invalid or llama.cpp cannot
    /// initialize the speculative implementation for the loaded model.
    pub fn new(
        target_context: LlamaContext<'model>,
        draft_context: LlamaContext<'model>,
        params: MtpSpeculativeParams,
        n_seq: u32,
    ) -> Result<Self, MtpSpeculativeError> {
        if params.n_max <= 0 || params.n_min < 0 || params.n_min > params.n_max || n_seq == 0 {
            return Err(MtpSpeculativeError::InvalidParams);
        }
        let n_max =
            usize::try_from(params.n_max).map_err(|_| MtpSpeculativeError::InvalidParams)?;

        let raw = unsafe {
            llama_cpp_sys_2::llama_rs_mtp_speculative_init(
                target_context.context.as_ptr(),
                draft_context.context.as_ptr(),
                params.n_max,
                params.n_min,
                params.p_min,
                n_seq,
                true,
            )
        };
        let raw = NonNull::new(raw).ok_or(MtpSpeculativeError::InitFailed)?;

        Ok(Self {
            raw,
            target_context,
            draft_context,
            n_max,
            n_seq,
        })
    }

    /// Access the target context.
    #[must_use]
    pub fn target_context(&self) -> &LlamaContext<'model> {
        &self.target_context
    }

    /// Access the target context for decode and cache rollback operations.
    pub fn target_context_mut(&mut self) -> &mut LlamaContext<'model> {
        &mut self.target_context
    }

    /// Access the draft context for cache rollback operations.
    pub fn draft_context_mut(&mut self) -> &mut LlamaContext<'model> {
        &mut self.draft_context
    }

    /// Borrow the target context and speculative controller independently.
    pub fn split_mut(&mut self) -> (&mut LlamaContext<'model>, MtpOperations<'_>) {
        (
            &mut self.target_context,
            MtpOperations {
                raw: &mut self.raw,
                n_max: self.n_max,
                n_seq: self.n_seq,
            },
        )
    }

    /// Borrow both owned contexts and the speculative controller as disjoint mutable fields.
    pub fn split_all_mut(
        &mut self,
    ) -> (
        &mut LlamaContext<'model>,
        &mut LlamaContext<'model>,
        MtpOperations<'_>,
    ) {
        (
            &mut self.target_context,
            &mut self.draft_context,
            MtpOperations {
                raw: &mut self.raw,
                n_max: self.n_max,
                n_seq: self.n_seq,
            },
        )
    }

    /// Begin a new generation from the given prompt tokens.
    ///
    /// # Errors
    ///
    /// Returns an error if llama.cpp rejects the call.
    pub fn begin(
        &mut self,
        sequence_id: i32,
        prompt_tokens: &[LlamaToken],
    ) -> Result<(), MtpSpeculativeError> {
        self.validate_sequence(sequence_id)?;
        let prompt = tokens_to_raw(prompt_tokens);
        native_call("begin", |out_error| unsafe {
            llama_cpp_sys_2::llama_rs_mtp_speculative_begin(
                self.raw.as_ptr(),
                sequence_id,
                prompt.as_ptr(),
                prompt.len(),
                out_error,
            )
        })
    }

    /// Process a batch that was just decoded by the target context.
    ///
    /// Every token must belong to exactly one configured sequence.
    ///
    /// # Errors
    ///
    /// Returns an error if llama.cpp cannot update the MTP draft context.
    pub fn process(&mut self, batch: &LlamaBatch<'_>) -> Result<(), MtpSpeculativeError> {
        native_call("process", |out_error| unsafe {
            llama_cpp_sys_2::llama_rs_mtp_speculative_process(
                self.raw.as_ptr(),
                std::ptr::from_ref(&batch.raw),
                out_error,
            )
        })
    }

    /// Generate draft tokens after `id_last`.
    ///
    /// # Errors
    ///
    /// Returns an error if llama.cpp rejects the draft operation or emits more
    /// draft tokens than requested.
    pub fn prepare_draft(
        &mut self,
        sequence_id: i32,
        n_past: i32,
        id_last: LlamaToken,
        prompt_tokens: &[LlamaToken],
        n_max: usize,
    ) -> Result<(), MtpSpeculativeError> {
        if n_past < 0 {
            return Err(MtpSpeculativeError::InvalidParams);
        }
        self.validate_sequence(sequence_id)?;
        if n_max == 0 || n_max > self.n_max {
            return Err(MtpSpeculativeError::InvalidParams);
        }

        let prompt = tokens_to_raw(prompt_tokens);
        let n_max = i32::try_from(n_max).map_err(|_| MtpSpeculativeError::InvalidParams)?;
        native_call("prepare draft", |out_error| unsafe {
            llama_cpp_sys_2::llama_rs_mtp_speculative_prepare_draft(
                self.raw.as_ptr(),
                sequence_id,
                n_past,
                id_last.0,
                prompt.as_ptr(),
                prompt.len(),
                n_max,
                out_error,
            )
        })
    }

    /// Generate every sequence draft prepared with [`Self::prepare_draft`].
    pub fn draft_all(&mut self) -> Result<(), MtpSpeculativeError> {
        native_call("draft", |out_error| unsafe {
            llama_cpp_sys_2::llama_rs_mtp_speculative_draft(self.raw.as_ptr(), out_error)
        })
    }

    /// Copy the most recently generated draft for one sequence.
    pub fn take_draft(&mut self, sequence_id: i32) -> Result<Vec<LlamaToken>, MtpSpeculativeError> {
        self.validate_sequence(sequence_id)?;
        let mut raw_out = vec![0; self.n_max];
        let mut out_len = 0_usize;
        let mut native_error = ptr::null_mut();
        let status = unsafe {
            llama_cpp_sys_2::llama_rs_mtp_speculative_get_draft(
                self.raw.as_ptr(),
                sequence_id,
                raw_out.as_mut_ptr(),
                raw_out.len(),
                &raw mut out_len,
                &raw mut native_error,
            )
        };
        if status == llama_cpp_sys_2::LLAMA_RS_STATUS_ALLOCATION_FAILED {
            take_native_error(native_error);
            return Err(MtpSpeculativeError::DraftOverflow);
        }
        status_to_result("get draft", status, native_error)?;
        raw_out.truncate(out_len);
        Ok(raw_out.into_iter().map(LlamaToken).collect())
    }

    /// Atomically commit a verified prefix or restore the checkpoint for replay.
    pub fn resolve_verification(
        &mut self,
        sequence_id: i32,
        proposed_count: usize,
        accepted_count: usize,
        next_position: i32,
    ) -> Result<MtpVerificationResolution, MtpSpeculativeError> {
        self.validate_sequence(sequence_id)?;
        resolve_verification(
            self.raw.as_ptr(),
            sequence_id,
            proposed_count,
            accepted_count,
            next_position,
        )
    }

    fn validate_sequence(&self, sequence_id: i32) -> Result<(), MtpSpeculativeError> {
        if sequence_id < 0 || u32::try_from(sequence_id).map_or(true, |id| id >= self.n_seq) {
            Err(MtpSpeculativeError::InvalidParams)
        } else {
            Ok(())
        }
    }
}

impl MtpOperations<'_> {
    /// Maximum draft length configured for this handle.
    #[must_use]
    pub fn max_draft_tokens(&self) -> usize {
        self.n_max
    }

    /// Begin speculative state for one target sequence after its prompt is decoded.
    pub fn begin(
        &mut self,
        sequence_id: i32,
        prompt_tokens: &[LlamaToken],
    ) -> Result<(), MtpSpeculativeError> {
        validate_sequence(self.n_seq, sequence_id)?;
        let prompt = tokens_to_raw(prompt_tokens);
        native_call("begin", |out_error| unsafe {
            llama_cpp_sys_2::llama_rs_mtp_speculative_begin(
                self.raw.as_ptr(),
                sequence_id,
                prompt.as_ptr(),
                prompt.len(),
                out_error,
            )
        })
    }

    /// Mirror a target batch into the linked MTP context.
    pub fn process(&mut self, batch: &LlamaBatch<'_>) -> Result<(), MtpSpeculativeError> {
        native_call("process", |out_error| unsafe {
            llama_cpp_sys_2::llama_rs_mtp_speculative_process(
                self.raw.as_ptr(),
                std::ptr::from_ref(&batch.raw),
                out_error,
            )
        })
    }

    /// Register one sequence for the next combined draft pass.
    pub fn prepare_draft(
        &mut self,
        sequence_id: i32,
        n_past: i32,
        id_last: LlamaToken,
        prompt_tokens: &[LlamaToken],
        n_max: usize,
    ) -> Result<(), MtpSpeculativeError> {
        validate_sequence(self.n_seq, sequence_id)?;
        if n_past < 0 || n_max == 0 || n_max > self.n_max {
            return Err(MtpSpeculativeError::InvalidParams);
        }
        let prompt = tokens_to_raw(prompt_tokens);
        let n_max = i32::try_from(n_max).map_err(|_| MtpSpeculativeError::InvalidParams)?;
        native_call("prepare draft", |out_error| unsafe {
            llama_cpp_sys_2::llama_rs_mtp_speculative_prepare_draft(
                self.raw.as_ptr(),
                sequence_id,
                n_past,
                id_last.0,
                prompt.as_ptr(),
                prompt.len(),
                n_max,
                out_error,
            )
        })
    }

    /// Generate drafts for all sequences registered by [`Self::prepare_draft`].
    pub fn draft_all(&mut self) -> Result<(), MtpSpeculativeError> {
        native_call("draft", |out_error| unsafe {
            llama_cpp_sys_2::llama_rs_mtp_speculative_draft(self.raw.as_ptr(), out_error)
        })
    }

    /// Take the draft generated for one sequence.
    pub fn take_draft(&mut self, sequence_id: i32) -> Result<Vec<LlamaToken>, MtpSpeculativeError> {
        validate_sequence(self.n_seq, sequence_id)?;
        let mut raw_out = vec![0; self.n_max];
        let mut out_len = 0;
        let mut native_error = ptr::null_mut();
        let status = unsafe {
            llama_cpp_sys_2::llama_rs_mtp_speculative_get_draft(
                self.raw.as_ptr(),
                sequence_id,
                raw_out.as_mut_ptr(),
                raw_out.len(),
                &raw mut out_len,
                &raw mut native_error,
            )
        };
        if status == llama_cpp_sys_2::LLAMA_RS_STATUS_ALLOCATION_FAILED {
            take_native_error(native_error);
            return Err(MtpSpeculativeError::DraftOverflow);
        }
        status_to_result("get draft", status, native_error)?;
        raw_out.truncate(out_len);
        Ok(raw_out.into_iter().map(LlamaToken).collect())
    }

    /// Atomically commit a verified prefix or restore the checkpoint for replay.
    pub fn resolve_verification(
        &mut self,
        sequence_id: i32,
        proposed_count: usize,
        accepted_count: usize,
        next_position: i32,
    ) -> Result<MtpVerificationResolution, MtpSpeculativeError> {
        validate_sequence(self.n_seq, sequence_id)?;
        resolve_verification(
            self.raw.as_ptr(),
            sequence_id,
            proposed_count,
            accepted_count,
            next_position,
        )
    }

    /// Remove the same position range from both target and draft memories.
    pub fn remove_sequence_range(
        &mut self,
        sequence_id: i32,
        start: i32,
        end: i32,
    ) -> Result<(), MtpSpeculativeError> {
        validate_sequence(self.n_seq, sequence_id)?;
        native_call("remove sequence range", |out_error| unsafe {
            llama_cpp_sys_2::llama_rs_mtp_speculative_seq_rm(
                self.raw.as_ptr(),
                sequence_id,
                start,
                end,
                out_error,
            )
        })
    }
}

impl Drop for MtpSpeculative<'_> {
    fn drop(&mut self) {
        unsafe {
            llama_cpp_sys_2::llama_rs_mtp_speculative_free(self.raw.as_ptr());
        }
    }
}

fn tokens_to_raw(tokens: &[LlamaToken]) -> Vec<llama_cpp_sys_2::llama_token> {
    tokens.iter().map(|token| token.0).collect()
}

fn resolve_verification(
    raw: *mut llama_cpp_sys_2::llama_rs_mtp_speculative,
    sequence_id: i32,
    proposed_count: usize,
    accepted_count: usize,
    next_position: i32,
) -> Result<MtpVerificationResolution, MtpSpeculativeError> {
    if proposed_count == 0 || accepted_count > proposed_count || next_position < 0 {
        return Err(MtpSpeculativeError::InvalidParams);
    }
    let accepted_count =
        u16::try_from(accepted_count).map_err(|_| MtpSpeculativeError::InvalidParams)?;
    let mut replay = false;
    native_call("resolve verification", |out_error| unsafe {
        llama_cpp_sys_2::llama_rs_mtp_speculative_resolve(
            raw,
            sequence_id,
            proposed_count,
            accepted_count,
            next_position,
            &raw mut replay,
            out_error,
        )
    })?;
    Ok(if replay {
        MtpVerificationResolution::Replay
    } else {
        MtpVerificationResolution::Committed
    })
}

fn native_call(
    operation: &'static str,
    call: impl FnOnce(*mut *mut c_char) -> llama_cpp_sys_2::llama_rs_status,
) -> Result<(), MtpSpeculativeError> {
    let mut native_error = ptr::null_mut();
    let status = call(&raw mut native_error);
    status_to_result(operation, status, native_error)
}

fn status_to_result(
    operation: &'static str,
    status: llama_cpp_sys_2::llama_rs_status,
    native_error: *mut c_char,
) -> Result<(), MtpSpeculativeError> {
    if status_is_ok(status) {
        if !native_error.is_null() {
            unsafe { llama_cpp_sys_2::llama_rs_string_free(native_error) };
        }
        Ok(())
    } else {
        Err(MtpSpeculativeError::Native {
            operation,
            status,
            message: take_native_error(native_error),
        })
    }
}

fn take_native_error(native_error: *mut c_char) -> String {
    if native_error.is_null() {
        return "native bridge returned no diagnostic".to_owned();
    }
    let message = unsafe { CStr::from_ptr(native_error) }
        .to_string_lossy()
        .into_owned();
    unsafe { llama_cpp_sys_2::llama_rs_string_free(native_error) };
    message
}

fn validate_sequence(n_seq: u32, sequence_id: i32) -> Result<(), MtpSpeculativeError> {
    if sequence_id < 0 || u32::try_from(sequence_id).map_or(true, |id| id >= n_seq) {
        Err(MtpSpeculativeError::InvalidParams)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_mtp_error_preserves_operation_status_and_message() {
        let error = native_call("process", |out_error| unsafe {
            llama_cpp_sys_2::llama_rs_mtp_speculative_process(
                ptr::null_mut(),
                ptr::null(),
                out_error,
            )
        })
        .unwrap_err();

        assert_eq!(
            error,
            MtpSpeculativeError::Native {
                operation: "process",
                status: llama_cpp_sys_2::LLAMA_RS_STATUS_INVALID_ARGUMENT,
                message: "invalid MTP process arguments".to_owned(),
            }
        );
    }
}
