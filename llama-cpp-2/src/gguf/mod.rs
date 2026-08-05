//! Safe wrapper around `gguf_context` for reading GGUF file metadata.
//!
//! Provides metadata-only access to GGUF files without loading tensor data.
//! Useful for inspecting model architecture parameters before loading a model.

use std::ffi::{CStr, CString};
use std::path::Path;
use std::ptr::NonNull;

macro_rules! file_types {
    ($($variant:ident => $native:ident),+ $(,)?) => {
        /// A GGUF model file type supported by the pinned llama.cpp runtime.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #[repr(u32)]
        pub enum FileType {
            $(
                #[doc = concat!("The llama.cpp `", stringify!($native), "` file type.")]
                $variant = llama_cpp_sys_2::llama_ftype::$native as u32,
            )+
        }

        impl TryFrom<u32> for FileType {
            type Error = UnknownFileType;

            fn try_from(value: u32) -> Result<Self, Self::Error> {
                match value {
                    $(
                        value if value == Self::$variant as u32 => Ok(Self::$variant),
                    )+
                    value => Err(UnknownFileType(value)),
                }
            }
        }

        impl From<FileType> for llama_cpp_sys_2::llama_ftype {
            fn from(value: FileType) -> Self {
                match value {
                    $(
                        FileType::$variant => Self::$native,
                    )+
                }
            }
        }

        impl From<llama_cpp_sys_2::llama_ftype> for FileType {
            fn from(value: llama_cpp_sys_2::llama_ftype) -> Self {
                match value {
                    $(
                        llama_cpp_sys_2::llama_ftype::$native => Self::$variant,
                    )+
                }
            }
        }
    };
}

file_types! {
    AllF32 => LLAMA_FTYPE_ALL_F32,
    MostlyF16 => LLAMA_FTYPE_MOSTLY_F16,
    MostlyQ4_0 => LLAMA_FTYPE_MOSTLY_Q4_0,
    MostlyQ4_1 => LLAMA_FTYPE_MOSTLY_Q4_1,
    MostlyQ8_0 => LLAMA_FTYPE_MOSTLY_Q8_0,
    MostlyQ5_0 => LLAMA_FTYPE_MOSTLY_Q5_0,
    MostlyQ5_1 => LLAMA_FTYPE_MOSTLY_Q5_1,
    MostlyQ2K => LLAMA_FTYPE_MOSTLY_Q2_K,
    MostlyQ3KSmall => LLAMA_FTYPE_MOSTLY_Q3_K_S,
    MostlyQ3KMedium => LLAMA_FTYPE_MOSTLY_Q3_K_M,
    MostlyQ3KLarge => LLAMA_FTYPE_MOSTLY_Q3_K_L,
    MostlyQ4KSmall => LLAMA_FTYPE_MOSTLY_Q4_K_S,
    MostlyQ4KMedium => LLAMA_FTYPE_MOSTLY_Q4_K_M,
    MostlyQ5KSmall => LLAMA_FTYPE_MOSTLY_Q5_K_S,
    MostlyQ5KMedium => LLAMA_FTYPE_MOSTLY_Q5_K_M,
    MostlyQ6K => LLAMA_FTYPE_MOSTLY_Q6_K,
    MostlyIq2Xxs => LLAMA_FTYPE_MOSTLY_IQ2_XXS,
    MostlyIq2Xs => LLAMA_FTYPE_MOSTLY_IQ2_XS,
    MostlyQ2KSmall => LLAMA_FTYPE_MOSTLY_Q2_K_S,
    MostlyIq3Xs => LLAMA_FTYPE_MOSTLY_IQ3_XS,
    MostlyIq3Xxs => LLAMA_FTYPE_MOSTLY_IQ3_XXS,
    MostlyIq1Small => LLAMA_FTYPE_MOSTLY_IQ1_S,
    MostlyIq4Nl => LLAMA_FTYPE_MOSTLY_IQ4_NL,
    MostlyIq3Small => LLAMA_FTYPE_MOSTLY_IQ3_S,
    MostlyIq3Medium => LLAMA_FTYPE_MOSTLY_IQ3_M,
    MostlyIq2Small => LLAMA_FTYPE_MOSTLY_IQ2_S,
    MostlyIq2Medium => LLAMA_FTYPE_MOSTLY_IQ2_M,
    MostlyIq4Xs => LLAMA_FTYPE_MOSTLY_IQ4_XS,
    MostlyIq1Medium => LLAMA_FTYPE_MOSTLY_IQ1_M,
    MostlyBf16 => LLAMA_FTYPE_MOSTLY_BF16,
    MostlyTq1_0 => LLAMA_FTYPE_MOSTLY_TQ1_0,
    MostlyTq2_0 => LLAMA_FTYPE_MOSTLY_TQ2_0,
    MostlyMxfp4Moe => LLAMA_FTYPE_MOSTLY_MXFP4_MOE,
    MostlyNvfp4 => LLAMA_FTYPE_MOSTLY_NVFP4,
    MostlyQ1_0 => LLAMA_FTYPE_MOSTLY_Q1_0,
    MostlyQ2_0 => LLAMA_FTYPE_MOSTLY_Q2_0,
    Guessed => LLAMA_FTYPE_GUESSED,
}

impl FileType {
    /// The numeric value stored in GGUF `general.file_type` metadata.
    #[must_use]
    pub const fn as_raw(self) -> u32 {
        self as u32
    }

    /// llama.cpp's authoritative name for this file type.
    ///
    /// # Errors
    ///
    /// Returns an error if llama.cpp returns a null pointer or invalid UTF-8.
    pub fn name(self) -> Result<&'static str, FileTypeNameError> {
        let pointer = unsafe { llama_cpp_sys_2::llama_ftype_name(self.into()) };
        let pointer = NonNull::new(pointer.cast_mut()).ok_or(FileTypeNameError::Null)?;
        Ok(unsafe { CStr::from_ptr(pointer.as_ptr()) }.to_str()?)
    }
}

/// A numeric GGUF model file type not recognized by the pinned llama.cpp runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("unknown GGUF model file type {0}")]
pub struct UnknownFileType(pub u32);

/// Failure to read a model file type name from llama.cpp.
#[derive(Debug, thiserror::Error)]
pub enum FileTypeNameError {
    /// llama.cpp unexpectedly returned a null pointer.
    #[error("llama.cpp returned a null model file type name")]
    Null,
    /// llama.cpp returned a model file type name that was not valid UTF-8.
    #[error("llama.cpp returned a model file type name that was not valid UTF-8")]
    InvalidUtf8(#[from] std::str::Utf8Error),
}

/// Return the exact contiguous storage required by one GGML tensor shape.
///
/// This is pure shape arithmetic over the pinned GGML type table. It does not initialize a backend,
/// open a model, or allocate tensor storage. `dimensions` are in GGML order, with the quantized row
/// width first.
#[must_use]
pub fn tensor_storage_bytes(tensor_type: u32, dimensions: &[u64]) -> Option<u64> {
    if tensor_type >= llama_cpp_sys_2::GGML_TYPE_COUNT || dimensions.is_empty() {
        return None;
    }
    let row_elements = i64::try_from(dimensions[0]).ok()?;
    let block_size = unsafe { llama_cpp_sys_2::ggml_blck_size(tensor_type) };
    if row_elements <= 0 || block_size <= 0 || row_elements % block_size != 0 {
        return None;
    }
    let row_bytes =
        u64::try_from(unsafe { llama_cpp_sys_2::ggml_row_size(tensor_type, row_elements) }).ok()?;
    dimensions[1..]
        .iter()
        .try_fold(row_bytes, |bytes, dimension| bytes.checked_mul(*dimension))
}

/// A safe wrapper around `gguf_context`.
///
/// Opens a GGUF file and parses only the metadata header; tensor weights are
/// never loaded into memory (`no_alloc = true`).
#[derive(Debug)]
pub struct GgufContext {
    ctx: NonNull<llama_cpp_sys_2::gguf_context>,
}

impl GgufContext {
    /// Open a GGUF file and parse its metadata header.
    ///
    /// Returns `None` if the path contains a null byte, the file does not
    /// exist, or the file is not a valid GGUF file.
    #[must_use]
    pub fn from_file(path: &Path) -> Option<Self> {
        let c_path = CString::new(path.to_str()?).ok()?;
        let params = llama_cpp_sys_2::gguf_init_params {
            no_alloc: true,
            ctx: std::ptr::null_mut(),
        };
        let ptr = unsafe { llama_cpp_sys_2::gguf_init_from_file(c_path.as_ptr(), params) };
        Some(Self {
            ctx: NonNull::new(ptr)?,
        })
    }

    /// Total number of key-value pairs in the metadata.
    #[must_use]
    pub fn n_kv(&self) -> i64 {
        unsafe { llama_cpp_sys_2::gguf_get_n_kv(self.ctx.as_ptr()) }
    }

    /// Find the index of a key by name. Returns `-1` if not found.
    #[must_use]
    pub fn find_key(&self, key: &str) -> i64 {
        let Ok(c_key) = CString::new(key) else {
            return -1;
        };
        unsafe { llama_cpp_sys_2::gguf_find_key(self.ctx.as_ptr(), c_key.as_ptr()) }
    }

    /// Return the key name at the given index, or `None` if out of range.
    #[must_use]
    pub fn key_at(&self, idx: i64) -> Option<&str> {
        let ptr = unsafe { llama_cpp_sys_2::gguf_get_key(self.ctx.as_ptr(), idx) };
        if ptr.is_null() {
            return None;
        }
        unsafe { CStr::from_ptr(ptr).to_str().ok() }
    }

    /// Return the value type of the KV pair at `idx`.
    #[must_use]
    pub fn kv_type(&self, idx: i64) -> llama_cpp_sys_2::gguf_type {
        unsafe { llama_cpp_sys_2::gguf_get_kv_type(self.ctx.as_ptr(), idx) }
    }

    /// Read a `uint32` value. Panics (inside llama.cpp) if the stored type is
    /// not `GGUF_TYPE_UINT32` — check `kv_type` first if unsure.
    #[must_use]
    pub fn val_u32(&self, idx: i64) -> u32 {
        unsafe { llama_cpp_sys_2::gguf_get_val_u32(self.ctx.as_ptr(), idx) }
    }

    /// Read an `int32` value.
    #[must_use]
    pub fn val_i32(&self, idx: i64) -> i32 {
        unsafe { llama_cpp_sys_2::gguf_get_val_i32(self.ctx.as_ptr(), idx) }
    }

    /// Read a `uint64` value.
    #[must_use]
    pub fn val_u64(&self, idx: i64) -> u64 {
        unsafe { llama_cpp_sys_2::gguf_get_val_u64(self.ctx.as_ptr(), idx) }
    }

    /// Read a string value. Returns `None` if the pointer is null or not
    /// valid UTF-8.
    #[must_use]
    pub fn val_str(&self, idx: i64) -> Option<&str> {
        let ptr = unsafe { llama_cpp_sys_2::gguf_get_val_str(self.ctx.as_ptr(), idx) };
        if ptr.is_null() {
            return None;
        }
        unsafe { CStr::from_ptr(ptr).to_str().ok() }
    }

    /// Total number of tensors described in the file.
    #[must_use]
    pub fn n_tensors(&self) -> i64 {
        unsafe { llama_cpp_sys_2::gguf_get_n_tensors(self.ctx.as_ptr()) }
    }
}

impl Drop for GgufContext {
    fn drop(&mut self) {
        unsafe { llama_cpp_sys_2::gguf_free(self.ctx.as_ptr()) }
    }
}

#[cfg(test)]
mod tests;
