//! Representation of an initialized llama backend

use crate::LlamaCppError;
use llama_cpp_sys_2::ggml_log_level;
use std::marker::PhantomData;
use std::num::NonZeroI32;
use std::ptr::NonNull;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering::SeqCst;

/// Representation of an initialized llama backend
/// This is required as a parameter for most llama functions as the backend must be initialized
/// before any llama functions are called. This type is proof of initialization.
#[derive(Eq, PartialEq, Debug)]
pub struct LlamaBackend {}

/// Scheduling priority for a native GGML thread pool.
#[derive(Debug, Default, Eq, PartialEq, Copy, Clone)]
pub enum LlamaThreadPoolPriority {
    /// Lower than normal scheduling priority.
    Low,
    /// The platform's normal scheduling priority.
    #[default]
    Normal,
    /// Medium scheduling priority.
    Medium,
    /// High scheduling priority.
    High,
    /// Real-time scheduling priority. The host may reject this priority for unprivileged callers.
    Realtime,
}

impl From<LlamaThreadPoolPriority> for llama_cpp_sys_2::ggml_sched_priority {
    fn from(value: LlamaThreadPoolPriority) -> Self {
        match value {
            LlamaThreadPoolPriority::Low => llama_cpp_sys_2::GGML_SCHED_PRIO_LOW,
            LlamaThreadPoolPriority::Normal => llama_cpp_sys_2::GGML_SCHED_PRIO_NORMAL,
            LlamaThreadPoolPriority::Medium => llama_cpp_sys_2::GGML_SCHED_PRIO_MEDIUM,
            LlamaThreadPoolPriority::High => llama_cpp_sys_2::GGML_SCHED_PRIO_HIGH,
            LlamaThreadPoolPriority::Realtime => llama_cpp_sys_2::GGML_SCHED_PRIO_REALTIME,
        }
    }
}

/// Parameters used to create a persistent native GGML thread pool.
///
/// [`Self::new`] starts with llama.cpp's own `ggml_threadpool_params_default` values. In
/// particular, this gives the same empty CPU mask, non-strict affinity, polling interval, normal
/// priority, and unpaused state used by `llama-bench` unless the caller overrides one of them.
#[derive(Clone, Debug)]
pub struct LlamaThreadPoolParams {
    params: llama_cpp_sys_2::ggml_threadpool_params,
}

impl LlamaThreadPoolParams {
    /// Construct llama.cpp's default thread-pool parameters for `n_threads` workers.
    #[must_use]
    pub fn new(n_threads: NonZeroI32) -> Self {
        let params = unsafe { llama_cpp_sys_2::ggml_threadpool_params_default(n_threads.get()) };
        Self { params }
    }

    /// Set whether workers must adhere strictly to the configured CPU affinity mask.
    #[must_use]
    pub fn with_strict_cpu(mut self, strict_cpu: bool) -> Self {
        self.params.strict_cpu = strict_cpu;
        self
    }

    /// Set the active-wait polling interval used by worker threads.
    #[must_use]
    pub fn with_poll(mut self, poll: u32) -> Self {
        self.params.poll = poll;
        self
    }

    /// Set the native scheduling priority for worker threads.
    #[must_use]
    pub fn with_priority(mut self, priority: LlamaThreadPoolPriority) -> Self {
        self.params.prio = priority.into();
        self
    }

    /// Set whether the pool is created in a paused state.
    #[must_use]
    pub fn with_paused(mut self, paused: bool) -> Self {
        self.params.paused = paused;
        self
    }

    /// Return the configured worker count.
    #[must_use]
    pub fn n_threads(&self) -> i32 {
        self.params.n_threads
    }

    /// Return the configured polling interval.
    #[must_use]
    pub fn poll(&self) -> u32 {
        self.params.poll
    }

    /// Return whether strict CPU affinity is enabled.
    #[must_use]
    pub fn strict_cpu(&self) -> bool {
        self.params.strict_cpu
    }

    /// Return whether the pool starts paused.
    #[must_use]
    pub fn paused(&self) -> bool {
        self.params.paused
    }
}

type ThreadPoolFreeFn = unsafe extern "C" fn(*mut llama_cpp_sys_2::ggml_threadpool);

/// An owned, persistent native GGML thread pool.
///
/// The backend lifetime prevents the native backend registry from being torn down before the pool.
/// Attach this pool to a context with [`crate::context::LlamaContext::attach_threadpool`].
pub struct LlamaThreadPool<'backend> {
    threadpool: NonNull<llama_cpp_sys_2::ggml_threadpool>,
    free: ThreadPoolFreeFn,
    _backend: PhantomData<&'backend LlamaBackend>,
}

impl std::fmt::Debug for LlamaThreadPool<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LlamaThreadPool")
            .field("threadpool", &self.threadpool)
            .finish_non_exhaustive()
    }
}

/// Failure to resolve or create the CPU backend's native thread pool.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum LlamaThreadPoolCreateError {
    /// llama.cpp did not register a CPU backend device.
    #[error("llama.cpp CPU backend is unavailable")]
    CpuBackendUnavailable,
    /// The CPU backend does not export a function required to own a thread pool.
    #[error("llama.cpp CPU backend does not export {0}")]
    MissingBackendFunction(&'static str),
    /// The native thread-pool constructor returned a null pointer.
    #[error("llama.cpp failed to create a native thread pool")]
    NullReturn,
}

impl<'backend> LlamaThreadPool<'backend> {
    /// Create a persistent thread pool from the registered CPU backend.
    ///
    /// Resolving the constructor through the backend registry, as `llama-bench` does, supports both
    /// statically linked and dynamically loaded CPU backends.
    ///
    /// # Errors
    ///
    /// Returns an error when the CPU backend or either thread-pool function is unavailable, or when
    /// the native constructor returns null.
    pub fn new(
        _backend: &'backend LlamaBackend,
        params: &LlamaThreadPoolParams,
    ) -> Result<Self, LlamaThreadPoolCreateError> {
        type ThreadPoolNewFn = unsafe extern "C" fn(
            *mut llama_cpp_sys_2::ggml_threadpool_params,
        )
            -> *mut llama_cpp_sys_2::ggml_threadpool;

        let cpu_device = NonNull::new(unsafe {
            llama_cpp_sys_2::ggml_backend_dev_by_type(llama_cpp_sys_2::GGML_BACKEND_DEVICE_TYPE_CPU)
        })
        .ok_or(LlamaThreadPoolCreateError::CpuBackendUnavailable)?;
        let cpu_registry = NonNull::new(unsafe {
            llama_cpp_sys_2::ggml_backend_dev_backend_reg(cpu_device.as_ptr())
        })
        .ok_or(LlamaThreadPoolCreateError::CpuBackendUnavailable)?;

        let new_address = NonNull::new(unsafe {
            llama_cpp_sys_2::ggml_backend_reg_get_proc_address(
                cpu_registry.as_ptr(),
                c"ggml_threadpool_new".as_ptr(),
            )
        })
        .ok_or(LlamaThreadPoolCreateError::MissingBackendFunction(
            "ggml_threadpool_new",
        ))?;
        let free_address = NonNull::new(unsafe {
            llama_cpp_sys_2::ggml_backend_reg_get_proc_address(
                cpu_registry.as_ptr(),
                c"ggml_threadpool_free".as_ptr(),
            )
        })
        .ok_or(LlamaThreadPoolCreateError::MissingBackendFunction(
            "ggml_threadpool_free",
        ))?;

        // SAFETY: the addresses were resolved by their exact names from llama.cpp's CPU backend;
        // those exported functions have the signatures declared in ggml-cpu.h.
        let create: ThreadPoolNewFn = unsafe { std::mem::transmute(new_address.as_ptr()) };
        // SAFETY: see the matching constructor resolution above.
        let free: ThreadPoolFreeFn = unsafe { std::mem::transmute(free_address.as_ptr()) };
        let mut native_params = params.params;
        let threadpool = NonNull::new(unsafe { create(&raw mut native_params) })
            .ok_or(LlamaThreadPoolCreateError::NullReturn)?;

        Ok(Self {
            threadpool,
            free,
            _backend: PhantomData,
        })
    }

    pub(crate) fn as_ptr(&mut self) -> *mut llama_cpp_sys_2::ggml_threadpool {
        self.threadpool.as_ptr()
    }
}

impl Drop for LlamaThreadPool<'_> {
    fn drop(&mut self) {
        unsafe { (self.free)(self.threadpool.as_ptr()) }
    }
}

static LLAMA_BACKEND_INITIALIZED: AtomicBool = AtomicBool::new(false);

impl LlamaBackend {
    /// Mark the llama backend as initialized
    fn mark_init() -> crate::Result<()> {
        match LLAMA_BACKEND_INITIALIZED.compare_exchange(false, true, SeqCst, SeqCst) {
            Ok(_) => Ok(()),
            Err(_) => Err(LlamaCppError::BackendAlreadyInitialized),
        }
    }

    /// Initialize the llama backend (without numa).
    ///
    /// # Examples
    ///
    /// ```
    ///# use llama_cpp_2::llama_backend::LlamaBackend;
    ///# use llama_cpp_2::LlamaCppError;
    ///# use std::error::Error;
    ///
    ///# fn main() -> Result<(), Box<dyn Error>> {
    ///
    ///
    /// let backend = LlamaBackend::init()?;
    /// // the llama backend can only be initialized once
    /// assert_eq!(Err(LlamaCppError::BackendAlreadyInitialized), LlamaBackend::init());
    ///
    ///# Ok(())
    ///# }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error when another live backend owner already exists.
    #[tracing::instrument(skip_all)]
    pub fn init() -> crate::Result<LlamaBackend> {
        Self::mark_init()?;
        unsafe { llama_cpp_sys_2::llama_backend_init() }
        Ok(LlamaBackend {})
    }

    /// Initialize the llama backend (with numa).
    /// ```
    ///# use llama_cpp_2::llama_backend::LlamaBackend;
    ///# use std::error::Error;
    ///# use llama_cpp_2::llama_backend::NumaStrategy;
    ///
    ///# fn main() -> Result<(), Box<dyn Error>> {
    ///
    /// let llama_backend = LlamaBackend::init_numa(NumaStrategy::MIRROR)?;
    ///
    ///# Ok(())
    ///# }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error when another live backend owner already exists.
    #[tracing::instrument(skip_all)]
    pub fn init_numa(strategy: NumaStrategy) -> crate::Result<LlamaBackend> {
        Self::mark_init()?;
        unsafe {
            llama_cpp_sys_2::llama_numa_init(llama_cpp_sys_2::ggml_numa_strategy::from(strategy));
        }
        Ok(LlamaBackend {})
    }

    /// Was the code built for a GPU backend & is a supported one available.
    #[must_use]
    pub fn supports_gpu_offload(&self) -> bool {
        unsafe { llama_cpp_sys_2::llama_supports_gpu_offload() }
    }

    /// Does this platform support loading the model via mmap.
    #[must_use]
    pub fn supports_mmap(&self) -> bool {
        unsafe { llama_cpp_sys_2::llama_supports_mmap() }
    }

    /// Does this platform support locking the model in RAM.
    #[must_use]
    pub fn supports_mlock(&self) -> bool {
        unsafe { llama_cpp_sys_2::llama_supports_mlock() }
    }

    /// Change the output of llama.cpp's logging to be voided instead of pushed to `stderr`.
    pub fn void_logs(&mut self) {
        unsafe extern "C" fn void_log(
            _level: ggml_log_level,
            _text: *const ::std::os::raw::c_char,
            _user_data: *mut ::std::os::raw::c_void,
        ) {
        }

        let _logger_guard = crate::log::lock_native_logger();
        unsafe {
            llama_cpp_sys_2::llama_log_set(Some(void_log), std::ptr::null_mut());
        }
    }
}

/// A rusty wrapper around `numa_strategy`.
#[derive(Debug, Eq, PartialEq, Copy, Clone)]
pub enum NumaStrategy {
    /// The numa strategy is disabled.
    DISABLED,
    /// help wanted: what does this do?
    DISTRIBUTE,
    /// help wanted: what does this do?
    ISOLATE,
    /// help wanted: what does this do?
    NUMACTL,
    /// help wanted: what does this do?
    MIRROR,
    /// help wanted: what does this do?
    COUNT,
}

/// An invalid numa strategy was provided.
#[derive(Debug, Eq, PartialEq, Copy, Clone)]
pub struct InvalidNumaStrategy(
    /// The invalid numa strategy that was provided.
    pub llama_cpp_sys_2::ggml_numa_strategy,
);

impl TryFrom<llama_cpp_sys_2::ggml_numa_strategy> for NumaStrategy {
    type Error = InvalidNumaStrategy;

    fn try_from(value: llama_cpp_sys_2::ggml_numa_strategy) -> Result<Self, Self::Error> {
        match value {
            llama_cpp_sys_2::GGML_NUMA_STRATEGY_DISABLED => Ok(Self::DISABLED),
            llama_cpp_sys_2::GGML_NUMA_STRATEGY_DISTRIBUTE => Ok(Self::DISTRIBUTE),
            llama_cpp_sys_2::GGML_NUMA_STRATEGY_ISOLATE => Ok(Self::ISOLATE),
            llama_cpp_sys_2::GGML_NUMA_STRATEGY_NUMACTL => Ok(Self::NUMACTL),
            llama_cpp_sys_2::GGML_NUMA_STRATEGY_MIRROR => Ok(Self::MIRROR),
            llama_cpp_sys_2::GGML_NUMA_STRATEGY_COUNT => Ok(Self::COUNT),
            value => Err(InvalidNumaStrategy(value)),
        }
    }
}

impl From<NumaStrategy> for llama_cpp_sys_2::ggml_numa_strategy {
    fn from(value: NumaStrategy) -> Self {
        match value {
            NumaStrategy::DISABLED => llama_cpp_sys_2::GGML_NUMA_STRATEGY_DISABLED,
            NumaStrategy::DISTRIBUTE => llama_cpp_sys_2::GGML_NUMA_STRATEGY_DISTRIBUTE,
            NumaStrategy::ISOLATE => llama_cpp_sys_2::GGML_NUMA_STRATEGY_ISOLATE,
            NumaStrategy::NUMACTL => llama_cpp_sys_2::GGML_NUMA_STRATEGY_NUMACTL,
            NumaStrategy::MIRROR => llama_cpp_sys_2::GGML_NUMA_STRATEGY_MIRROR,
            NumaStrategy::COUNT => llama_cpp_sys_2::GGML_NUMA_STRATEGY_COUNT,
        }
    }
}

/// Drops the llama backend.
/// ```
///
///# use llama_cpp_2::llama_backend::LlamaBackend;
///# use std::error::Error;
///
///# fn main() -> Result<(), Box<dyn Error>> {
/// let backend = LlamaBackend::init()?;
/// drop(backend);
/// // can be initialized again after being dropped
/// let backend = LlamaBackend::init()?;
///# Ok(())
///# }
///
/// ```
impl Drop for LlamaBackend {
    fn drop(&mut self) {
        match LLAMA_BACKEND_INITIALIZED.compare_exchange(true, false, SeqCst, SeqCst) {
            Ok(_) => {}
            Err(_) => {
                unreachable!("This should not be reachable as the only ways to obtain a llama backend involve marking the backend as initialized.")
            }
        }
        unsafe { llama_cpp_sys_2::llama_backend_free() }
    }
}

/// Compile-time path to the built GGML backend modules directory.
/// Populated by build.rs from `DEP_LLAMA_BACKENDS_DIR` (emitted by llama-cpp-sys-2).
/// None on static builds or when the feature is disabled.
#[cfg(feature = "dynamic-backends")]
pub const BACKENDS_DIR: Option<&str> = option_env!("GGML_BACKENDS_DIR");

/// Load GGML backend modules from the given directory.
///
/// Call this before [`LlamaBackend::init`] to enable runtime hardware selection
/// (Vulkan, CPU-AVX512, CPU-AVX2, etc.) when built with the `dynamic-backends` feature.
#[cfg(feature = "dynamic-backends")]
pub fn load_backends_from_path(path: &std::path::Path) {
    let s = std::ffi::CString::new(path.to_str().expect("path must be valid UTF-8"))
        .expect("path must not contain null bytes");
    unsafe { llama_cpp_sys_2::ggml_backend_load_all_from_path(s.as_ptr()) }
}

/// Load GGML backend modules from the compile-time default directory ([`BACKENDS_DIR`]).
///
/// This is a no-op when `BACKENDS_DIR` is `None` (static builds or development builds
/// that have not set `GGML_BACKENDS_DIR`).
#[cfg(feature = "dynamic-backends")]
pub fn load_backends() {
    if let Some(dir) = BACKENDS_DIR {
        load_backends_from_path(std::path::Path::new(dir));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numa_from_and_to() {
        let numas = [
            NumaStrategy::DISABLED,
            NumaStrategy::DISTRIBUTE,
            NumaStrategy::ISOLATE,
            NumaStrategy::NUMACTL,
            NumaStrategy::MIRROR,
            NumaStrategy::COUNT,
        ];

        for numa in &numas {
            let from = llama_cpp_sys_2::ggml_numa_strategy::from(*numa);
            let to = NumaStrategy::try_from(from).expect("Failed to convert from and to");
            assert_eq!(*numa, to);
        }
    }

    #[test]
    fn check_invalid_numa() {
        let invalid = 800;
        let invalid = NumaStrategy::try_from(invalid);
        assert_eq!(invalid, Err(InvalidNumaStrategy(invalid.unwrap_err().0)));
    }
}
