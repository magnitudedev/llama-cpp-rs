//! Safe wrappers around llama.cpp's `common/chat` template and output parser.

use std::ffi::{c_char, CStr, CString, NulError};
use std::ptr::{self, NonNull};
use std::slice;
use std::string::FromUtf8Error;

use crate::model::LlamaModel;

use llama_cpp_sys_2 as sys;

/// Errors returned by the common chat bridge.
#[derive(Debug, thiserror::Error)]
pub enum CommonChatError {
    /// An input string contains an interior NUL byte.
    #[error("chat input contains an interior NUL byte: {0}")]
    InteriorNul(#[from] NulError),
    /// llama.cpp returned a string that is not valid UTF-8.
    #[error("llama.cpp returned invalid UTF-8: {0}")]
    InvalidUtf8(#[from] FromUtf8Error),
    /// The native parser detected invalid UTF-8 before committing stream state.
    #[error("llama.cpp chat stream produced invalid UTF-8: {message}")]
    InvalidNativeUtf8 {
        /// Native diagnostic, when available.
        message: String,
    },
    /// An operation was attempted after the native stream was finalized.
    #[error("invalid chat stream state: {message}")]
    InvalidStreamState {
        /// Native diagnostic, when available.
        message: String,
    },
    /// A native borrowed byte view violated its pointer/length contract.
    #[error("llama.cpp returned an invalid byte view for {0}")]
    InvalidNativeView(&'static str),
    /// Rust could not reserve storage for the native delta count.
    #[error("failed to reserve storage for {count} chat semantic deltas: {source}")]
    DeltaAllocation {
        /// Native-reported delta count.
        count: usize,
        /// Allocation or capacity-overflow diagnostic.
        #[source]
        source: std::collections::TryReserveError,
    },
    /// Rust could not reserve storage for the parsed tool-call count.
    #[error("failed to reserve storage for {count} parsed tool calls: {source}")]
    ParsedToolCallAllocation {
        /// Native-reported tool-call count.
        count: usize,
        /// Allocation or capacity-overflow diagnostic.
        #[source]
        source: std::collections::TryReserveError,
    },
    /// The requested option combination is invalid.
    #[error("invalid chat options: {0}")]
    InvalidOptions(&'static str),
    /// llama.cpp rejected the operation.
    #[error("llama.cpp common/chat failed with status {status}: {message}")]
    Native {
        /// Native `llama_rs_status` value.
        status: i32,
        /// Native diagnostic, when available.
        message: String,
    },
    /// A successful native call unexpectedly returned a null output.
    #[error("llama.cpp returned a null {0}")]
    NullOutput(&'static str),
    /// llama.cpp returned an enum value unknown to this pinned binding.
    #[error("llama.cpp returned unknown {kind} value {value}")]
    UnknownEnum {
        /// Enum being decoded.
        kind: &'static str,
        /// Raw enum value.
        value: i64,
    },
}

/// A message's typed content part.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChatContentPart {
    /// Part type supported by llama.cpp common chat.
    pub kind: ChatContentPartKind,
    /// Text or preprocessed media marker.
    pub text: String,
}

/// Supported typed content-part kinds.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChatContentPartKind {
    /// Plain text.
    Text,
    /// A media marker produced by the caller's multimodal preprocessing path.
    MediaMarker,
}

impl ChatContentPartKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::MediaMarker => "media_marker",
        }
    }
}

/// Text or typed message content.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ChatContent {
    /// Plain string content.
    Text(String),
    /// Typed text/media-marker parts.
    Parts(Vec<ChatContentPart>),
}

/// A function tool call in a chat message or parsed model response.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChatToolCall {
    /// Function name.
    pub name: String,
    /// JSON arguments, retained as text to support partial streaming parses.
    pub arguments: String,
    /// Model-provided call identifier, if present.
    pub id: Option<String>,
}

/// A normalized chat message accepted by llama.cpp common chat.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChatMessage {
    /// OpenAI-compatible role name.
    pub role: String,
    /// Message content. Tool-call-only assistant messages may omit it.
    pub content: Option<ChatContent>,
    /// Assistant tool calls.
    pub tool_calls: Vec<ChatToolCall>,
    /// Preserved assistant reasoning from chat history.
    pub reasoning_content: Option<String>,
    /// Function name for a tool result message.
    pub tool_name: Option<String>,
    /// Call identifier for a tool result message.
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    /// Construct a plain text message.
    #[must_use]
    pub fn text(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: Some(ChatContent::Text(content.into())),
            tool_calls: Vec::new(),
            reasoning_content: None,
            tool_name: None,
            tool_call_id: None,
        }
    }

    /// Construct a user text message.
    #[must_use]
    pub fn user(content: impl Into<String>) -> Self {
        Self::text("user", content)
    }

    /// Construct a system text message.
    #[must_use]
    pub fn system(content: impl Into<String>) -> Self {
        Self::text("system", content)
    }

    /// Construct an assistant text message.
    #[must_use]
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::text("assistant", content)
    }
}

/// A function declaration passed to the chat template and grammar builder.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChatTool {
    /// Function name.
    pub name: String,
    /// Human-readable function description.
    pub description: String,
    /// JSON Schema object encoded as JSON text.
    pub parameters_json: String,
}

/// A JSON-valued custom Jinja template argument.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChatTemplateKwarg {
    /// Template argument name.
    pub key: String,
    /// Argument value encoded as JSON text.
    pub value_json: String,
}

/// Tool-selection policy.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ChatToolChoice {
    /// Let the model choose between text and tool calls.
    #[default]
    Auto,
    /// Require a tool call.
    Required,
    /// Disable tool calls.
    None,
}

impl ChatToolChoice {
    fn raw(self) -> sys::llama_rs_chat_tool_choice {
        match self {
            Self::Auto => sys::LLAMA_RS_CHAT_TOOL_CHOICE_AUTO,
            Self::Required => sys::LLAMA_RS_CHAT_TOOL_CHOICE_REQUIRED,
            Self::None => sys::LLAMA_RS_CHAT_TOOL_CHOICE_NONE,
        }
    }
}

/// How the last assistant message should be continued.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ChatContinuation {
    /// Do not continue the last message.
    #[default]
    None,
    /// Infer whether reasoning or content is being continued.
    Auto,
    /// Continue reasoning content.
    Reasoning,
    /// Continue visible content.
    Content,
}

impl ChatContinuation {
    fn raw(self) -> sys::llama_rs_chat_continuation {
        match self {
            Self::None => sys::LLAMA_RS_CHAT_CONTINUATION_NONE,
            Self::Auto => sys::LLAMA_RS_CHAT_CONTINUATION_AUTO,
            Self::Reasoning => sys::LLAMA_RS_CHAT_CONTINUATION_REASONING,
            Self::Content => sys::LLAMA_RS_CHAT_CONTINUATION_CONTENT,
        }
    }
}

/// Reasoning extraction behavior used by llama.cpp's response parser.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Default)]
pub enum ChatReasoningFormat {
    /// Do not extract reasoning.
    None,
    /// Use llama.cpp's recommended automatic behavior.
    Auto,
    /// Legacy `DeepSeek` stream behavior.
    DeepSeekLegacy,
    /// Extract reasoning into dedicated fields, including while streaming.
    #[default]
    DeepSeek,
}

impl ChatReasoningFormat {
    fn raw(self) -> sys::llama_rs_chat_reasoning_format {
        match self {
            Self::None => sys::LLAMA_RS_CHAT_REASONING_FORMAT_NONE,
            Self::Auto => sys::LLAMA_RS_CHAT_REASONING_FORMAT_AUTO,
            Self::DeepSeekLegacy => sys::LLAMA_RS_CHAT_REASONING_FORMAT_DEEPSEEK_LEGACY,
            Self::DeepSeek => sys::LLAMA_RS_CHAT_REASONING_FORMAT_DEEPSEEK,
        }
    }

    fn from_raw(value: sys::llama_rs_chat_reasoning_format) -> Result<Self, CommonChatError> {
        match value {
            sys::LLAMA_RS_CHAT_REASONING_FORMAT_NONE => Ok(Self::None),
            sys::LLAMA_RS_CHAT_REASONING_FORMAT_AUTO => Ok(Self::Auto),
            sys::LLAMA_RS_CHAT_REASONING_FORMAT_DEEPSEEK_LEGACY => Ok(Self::DeepSeekLegacy),
            sys::LLAMA_RS_CHAT_REASONING_FORMAT_DEEPSEEK => Ok(Self::DeepSeek),
            value => Err(CommonChatError::UnknownEnum {
                kind: "reasoning format",
                value: i64::from(value),
            }),
        }
    }
}

/// Request inputs for applying a common chat template.
#[derive(Clone, Debug)]
#[allow(clippy::struct_excessive_bools)] // Mirrors independent llama.cpp request switches.
pub struct ChatPrepareOptions {
    /// Conversation history.
    pub messages: Vec<ChatMessage>,
    /// Explicit GBNF grammar.
    pub grammar: Option<String>,
    /// JSON Schema response constraint.
    pub json_schema: Option<String>,
    /// Append the assistant generation prompt.
    pub add_generation_prompt: bool,
    /// Continue the final assistant message.
    pub continuation: ChatContinuation,
    /// Use the full Jinja/autoparser path.
    pub use_jinja: bool,
    /// Available function tools.
    pub tools: Vec<ChatTool>,
    /// Tool-selection policy.
    pub tool_choice: ChatToolChoice,
    /// Override parallel-tool support. `None` follows template capabilities.
    pub parallel_tool_calls: Option<bool>,
    /// Reasoning response format.
    pub reasoning_format: ChatReasoningFormat,
    /// Explicitly control thinking. `None` uses llama.cpp's enabled default.
    pub enable_thinking: Option<bool>,
    /// Additional JSON-valued Jinja arguments.
    pub template_kwargs: Vec<ChatTemplateKwarg>,
    /// Render and parse all assistant output as plain content.
    pub force_pure_content: bool,
}

impl Default for ChatPrepareOptions {
    fn default() -> Self {
        Self {
            messages: Vec::new(),
            grammar: None,
            json_schema: None,
            add_generation_prompt: true,
            continuation: ChatContinuation::None,
            use_jinja: true,
            tools: Vec::new(),
            tool_choice: ChatToolChoice::Auto,
            parallel_tool_calls: None,
            reasoning_format: ChatReasoningFormat::DeepSeek,
            enable_thinking: None,
            template_kwargs: Vec::new(),
            force_pure_content: false,
        }
    }
}

/// Static capabilities detected from a Jinja chat template.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(clippy::struct_excessive_bools)] // Capability flags are independent native projections.
pub struct ChatCapabilities {
    /// Accepts string message content.
    pub supports_string_content: bool,
    /// Accepts typed message content.
    pub supports_typed_content: bool,
    /// Renders function definitions.
    pub supports_tools: bool,
    /// Renders assistant tool calls in history.
    pub supports_tool_calls: bool,
    /// Renders multiple tool calls in one assistant turn.
    pub supports_parallel_tool_calls: bool,
    /// Accepts a system message.
    pub supports_system_role: bool,
    /// Preserves reasoning in assistant history.
    pub supports_preserve_reasoning: bool,
    /// Accepts tool arguments as JSON objects.
    pub supports_object_arguments: bool,
    /// Responds to llama.cpp's `enable_thinking` control.
    pub supports_enable_thinking: bool,
}

/// Parser format selected by llama.cpp.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChatFormat {
    /// Plain content.
    ContentOnly,
    /// Simple PEG format.
    PegSimple,
    /// Native PEG format.
    PegNative,
    /// Gemma 4 PEG format.
    PegGemma4,
}

impl ChatFormat {
    fn from_raw(value: sys::llama_rs_chat_format) -> Result<Self, CommonChatError> {
        match value {
            sys::LLAMA_RS_CHAT_FORMAT_CONTENT_ONLY => Ok(Self::ContentOnly),
            sys::LLAMA_RS_CHAT_FORMAT_PEG_SIMPLE => Ok(Self::PegSimple),
            sys::LLAMA_RS_CHAT_FORMAT_PEG_NATIVE => Ok(Self::PegNative),
            sys::LLAMA_RS_CHAT_FORMAT_PEG_GEMMA4 => Ok(Self::PegGemma4),
            value => Err(CommonChatError::UnknownEnum {
                kind: "chat format",
                value: i64::from(value),
            }),
        }
    }
}

/// A lazy-grammar activation condition produced by llama.cpp.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ChatGrammarTrigger {
    /// Activate when a particular token is sampled.
    Token {
        /// Human-readable token value.
        value: String,
        /// Native token identifier.
        token: i32,
    },
    /// Activate on a literal word.
    Word(String),
    /// Activate on a regular-expression pattern.
    Pattern(String),
    /// Activate when the full generated text matches a pattern.
    PatternFull(String),
}

/// RAII owner of llama.cpp's compiled common chat templates.
#[derive(Debug)]
pub struct CommonChatTemplates {
    raw: NonNull<sys::llama_rs_chat_templates>,
}

impl CommonChatTemplates {
    /// Load the default and tool-use templates embedded in a model.
    ///
    /// # Errors
    ///
    /// Returns an error when template construction or native projection fails.
    pub fn from_model(model: &LlamaModel) -> Result<Self, CommonChatError> {
        Self::from_model_with_override(model, None)
    }

    /// Load templates from a model with an optional explicit Jinja override.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid strings or native template-construction failures.
    pub fn from_model_with_override(
        model: &LlamaModel,
        template_override: Option<&str>,
    ) -> Result<Self, CommonChatError> {
        let template = optional_c_string(template_override)?;
        Self::init(model.model.as_ptr(), template.as_ref(), None, None)
    }

    /// Compile an explicit template without loading a model.
    ///
    /// `bos_token` and `eos_token` supply values normally read from the model vocabulary.
    ///
    /// # Errors
    ///
    /// Returns an error for interior NUL bytes or native template-construction failures.
    pub fn from_template(
        template: &str,
        bos_token: Option<&str>,
        eos_token: Option<&str>,
    ) -> Result<Self, CommonChatError> {
        let template = CString::new(template)?;
        let bos = optional_c_string(bos_token)?;
        let eos = optional_c_string(eos_token)?;
        Self::init(
            ptr::null(),
            Some(&template),
            bos.as_ref(),
            eos.as_ref(),
        )
    }

    fn init(
        model: *const sys::llama_model,
        template: Option<&CString>,
        bos: Option<&CString>,
        eos: Option<&CString>,
    ) -> Result<Self, CommonChatError> {
        let mut raw = ptr::null_mut();
        let mut error = ptr::null_mut();
        let status = unsafe {
            sys::llama_rs_chat_templates_init(
                model,
                c_string_ptr(template),
                c_string_ptr(bos),
                c_string_ptr(eos),
                &raw mut raw,
                &raw mut error,
            )
        };
        check_native_status(status, error)?;
        let raw = NonNull::new(raw).ok_or(CommonChatError::NullOutput("chat template handle"))?;
        Ok(Self { raw })
    }

    /// Return the selected template source. `variant` currently accepts `"tool_use"`.
    ///
    /// # Errors
    ///
    /// Returns an error when the variant contains a NUL byte or native projection fails.
    pub fn source(&self, variant: Option<&str>) -> Result<String, CommonChatError> {
        let variant = optional_c_string(variant)?;
        let mut value = ptr::null_mut();
        let mut error = ptr::null_mut();
        let status = unsafe {
            sys::llama_rs_chat_templates_source(
                self.raw.as_ptr(),
                c_string_ptr(variant.as_ref()),
                &raw mut value,
                &raw mut error,
            )
        };
        check_native_status(status, error)?;
        take_owned_string(value, "template source")
    }

    /// Whether the model or caller supplied a template instead of llama.cpp's fallback.
    #[must_use]
    pub fn was_explicit(&self) -> bool {
        unsafe { sys::llama_rs_chat_templates_was_explicit(self.raw.as_ptr()) }
    }

    /// Detect message, tool, and thinking capabilities.
    ///
    /// # Errors
    ///
    /// Returns an error when llama.cpp cannot inspect the compiled template.
    pub fn capabilities(&self) -> Result<ChatCapabilities, CommonChatError> {
        let mut raw = sys::llama_rs_chat_capabilities {
            supports_string_content: false,
            supports_typed_content: false,
            supports_tools: false,
            supports_tool_calls: false,
            supports_parallel_tool_calls: false,
            supports_system_role: false,
            supports_preserve_reasoning: false,
            supports_object_arguments: false,
            supports_enable_thinking: false,
        };
        let mut error = ptr::null_mut();
        let status = unsafe {
            sys::llama_rs_chat_templates_capabilities(
                self.raw.as_ptr(),
                &raw mut raw,
                &raw mut error,
            )
        };
        check_native_status(status, error)?;
        Ok(ChatCapabilities {
            supports_string_content: raw.supports_string_content,
            supports_typed_content: raw.supports_typed_content,
            supports_tools: raw.supports_tools,
            supports_tool_calls: raw.supports_tool_calls,
            supports_parallel_tool_calls: raw.supports_parallel_tool_calls,
            supports_system_role: raw.supports_system_role,
            supports_preserve_reasoning: raw.supports_preserve_reasoning,
            supports_object_arguments: raw.supports_object_arguments,
            supports_enable_thinking: raw.supports_enable_thinking,
        })
    }

    /// Render a request and build its grammar and incremental output parser.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid request strings, option combinations, or native preparation.
    pub fn prepare(&self, options: &ChatPrepareOptions) -> Result<PreparedChat, CommonChatError> {
        if options.grammar.is_some() && options.json_schema.is_some() {
            return Err(CommonChatError::InvalidOptions(
                "grammar and json_schema are mutually exclusive",
            ));
        }
        if options.continuation != ChatContinuation::None && options.add_generation_prompt {
            return Err(CommonChatError::InvalidOptions(
                "continuation and add_generation_prompt cannot both be enabled",
            ));
        }

        let encoded = EncodedPrepareOptions::new(options)?;
        let raw_options = encoded.raw();
        let mut raw = ptr::null_mut();
        let mut error = ptr::null_mut();
        let status = unsafe {
            sys::llama_rs_chat_templates_prepare(
                self.raw.as_ptr(),
                &raw const raw_options,
                &raw mut raw,
                &raw mut error,
            )
        };
        check_native_status(status, error)?;
        let raw = NonNull::new(raw).ok_or(CommonChatError::NullOutput("prepared chat handle"))?;
        PreparedChat::from_raw(raw)
    }
}

impl Drop for CommonChatTemplates {
    fn drop(&mut self) {
        unsafe { sys::llama_rs_chat_templates_free(self.raw.as_ptr()) };
    }
}

/// A rendered prompt plus llama.cpp's grammar and parser configuration.
#[derive(Debug)]
pub struct PreparedChat {
    raw: NonNull<sys::llama_rs_chat_prepared>,
    format: ChatFormat,
    reasoning_format: ChatReasoningFormat,
    prompt: String,
    grammar: String,
    grammar_lazy: bool,
    generation_prompt: String,
    supports_thinking: bool,
    thinking_start_tag: Option<String>,
    thinking_end_tag: Option<String>,
    parser_definition: String,
    grammar_triggers: Vec<ChatGrammarTrigger>,
    preserved_tokens: Vec<String>,
    additional_stops: Vec<String>,
}

impl PreparedChat {
    fn from_raw(raw: NonNull<sys::llama_rs_chat_prepared>) -> Result<Self, CommonChatError> {
        let guard = PreparedHandle::new(raw);
        let raw_ptr = guard.as_ptr();
        let format = ChatFormat::from_raw(unsafe { sys::llama_rs_chat_prepared_format(raw_ptr) })?;
        let reasoning_format = ChatReasoningFormat::from_raw(unsafe {
            sys::llama_rs_chat_prepared_reasoning_format(raw_ptr)
        })?;
        let prompt = prepared_string(raw_ptr, sys::LLAMA_RS_CHAT_PREPARED_PROMPT)?;
        let grammar = prepared_string(raw_ptr, sys::LLAMA_RS_CHAT_PREPARED_GRAMMAR)?;
        let generation_prompt =
            prepared_string(raw_ptr, sys::LLAMA_RS_CHAT_PREPARED_GENERATION_PROMPT)?;
        let thinking_start_tag = empty_to_none(prepared_string(
            raw_ptr,
            sys::LLAMA_RS_CHAT_PREPARED_THINKING_START_TAG,
        )?);
        let thinking_end_tag = empty_to_none(prepared_string(
            raw_ptr,
            sys::LLAMA_RS_CHAT_PREPARED_THINKING_END_TAG,
        )?);
        let parser_definition = prepared_string(raw_ptr, sys::LLAMA_RS_CHAT_PREPARED_PARSER)?;
        let preserved_tokens =
            prepared_string_list(raw_ptr, sys::LLAMA_RS_CHAT_PREPARED_PRESERVED_TOKENS)?;
        let additional_stops =
            prepared_string_list(raw_ptr, sys::LLAMA_RS_CHAT_PREPARED_ADDITIONAL_STOPS)?;
        let grammar_triggers = prepared_grammar_triggers(raw_ptr)?;
        let grammar_lazy = unsafe { sys::llama_rs_chat_prepared_grammar_lazy(raw_ptr) };
        let supports_thinking = unsafe { sys::llama_rs_chat_prepared_supports_thinking(raw_ptr) };

        Ok(Self {
            raw: guard.release(),
            format,
            reasoning_format,
            prompt,
            grammar,
            grammar_lazy,
            generation_prompt,
            supports_thinking,
            thinking_start_tag,
            thinking_end_tag,
            parser_definition,
            grammar_triggers,
            preserved_tokens,
            additional_stops,
        })
    }

    /// Selected parser format.
    #[must_use]
    pub fn format(&self) -> ChatFormat {
        self.format
    }

    /// Selected reasoning response format.
    #[must_use]
    pub fn reasoning_format(&self) -> ChatReasoningFormat {
        self.reasoning_format
    }

    /// Fully rendered model prompt.
    #[must_use]
    pub fn prompt(&self) -> &str {
        &self.prompt
    }

    /// Generated GBNF grammar, or an empty string when unconstrained.
    #[must_use]
    pub fn grammar(&self) -> &str {
        &self.grammar
    }

    /// Whether the grammar is activated lazily by its triggers.
    #[must_use]
    pub fn grammar_lazy(&self) -> bool {
        self.grammar_lazy
    }

    /// Assistant prefix already represented in parser state.
    #[must_use]
    pub fn generation_prompt(&self) -> &str {
        &self.generation_prompt
    }

    /// Whether the chosen template supports a distinct thinking section.
    #[must_use]
    pub fn supports_thinking(&self) -> bool {
        self.supports_thinking
    }

    /// Thinking start marker detected by llama.cpp.
    #[must_use]
    pub fn thinking_start_tag(&self) -> Option<&str> {
        self.thinking_start_tag.as_deref()
    }

    /// Thinking end marker detected by llama.cpp.
    #[must_use]
    pub fn thinking_end_tag(&self) -> Option<&str> {
        self.thinking_end_tag.as_deref()
    }

    /// Serialized PEG parser definition retained for diagnostics.
    #[must_use]
    pub fn parser_definition(&self) -> &str {
        &self.parser_definition
    }

    /// Lazy grammar activation conditions.
    #[must_use]
    pub fn grammar_triggers(&self) -> &[ChatGrammarTrigger] {
        &self.grammar_triggers
    }

    /// Tokens that sampling must preserve verbatim.
    #[must_use]
    pub fn preserved_tokens(&self) -> &[String] {
        &self.preserved_tokens
    }

    /// Template-specific stop strings to add to request stops.
    #[must_use]
    pub fn additional_stops(&self) -> &[String] {
        &self.additional_stops
    }

    /// Instantiate an incremental parser from the prepared configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when llama.cpp cannot construct the parser.
    pub fn parser(&self, options: ChatParserOptions) -> Result<ChatParser, CommonChatError> {
        let raw_options = options.raw();
        let mut raw = ptr::null_mut();
        let mut error = ptr::null_mut();
        let status = unsafe {
            sys::llama_rs_chat_parser_init(
                self.raw.as_ptr(),
                &raw const raw_options,
                &raw mut raw,
                &raw mut error,
            )
        };
        check_native_status(status, error)?;
        let raw = NonNull::new(raw).ok_or(CommonChatError::NullOutput("chat parser handle"))?;
        Ok(ChatParser { raw })
    }

    /// Instantiate a stateful parser that emits native semantic deltas as text arrives.
    ///
    /// # Errors
    ///
    /// Returns an error when llama.cpp cannot construct the stream parser.
    pub fn stream_parser(
        &self,
        options: ChatParserOptions,
    ) -> Result<ChatStreamParser, CommonChatError> {
        let raw_options = options.raw();
        let mut raw = ptr::null_mut();
        let mut error = ptr::null_mut();
        let status = unsafe {
            sys::llama_rs_chat_stream_init(
                self.raw.as_ptr(),
                &raw const raw_options,
                &raw mut raw,
                &raw mut error,
            )
        };
        check_native_status(status, error)?;
        let raw = NonNull::new(raw).ok_or(CommonChatError::NullOutput("chat stream handle"))?;
        Ok(ChatStreamParser { raw })
    }
}

impl Drop for PreparedChat {
    fn drop(&mut self) {
        unsafe { sys::llama_rs_chat_prepared_free(self.raw.as_ptr()) };
    }
}

/// Runtime parser options matching llama-server's parser state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(clippy::struct_excessive_bools)] // Mirrors llama.cpp's parser option record.
pub struct ChatParserOptions {
    /// Inline reasoning in content for legacy stream compatibility.
    pub reasoning_in_content: bool,
    /// Parse tool calls when the prepared template supports them.
    pub parse_tool_calls: bool,
    /// Treat the request as a final-message continuation.
    pub is_continuation: bool,
    /// Include an assistant prefill in parsed output.
    pub echo: bool,
    /// Emit native PEG debug output.
    pub debug: bool,
}

impl Default for ChatParserOptions {
    fn default() -> Self {
        Self {
            reasoning_in_content: false,
            parse_tool_calls: true,
            is_continuation: false,
            echo: false,
            debug: false,
        }
    }
}

impl ChatParserOptions {
    fn raw(self) -> sys::llama_rs_chat_parser_options {
        sys::llama_rs_chat_parser_options {
            reasoning_in_content: self.reasoning_in_content,
            parse_tool_calls: self.parse_tool_calls,
            is_continuation: self.is_continuation,
            echo: self.echo,
            debug: self.debug,
        }
    }
}

/// RAII owner of a prepared llama.cpp output parser.
#[derive(Debug)]
pub struct ChatParser {
    raw: NonNull<sys::llama_rs_chat_parser>,
}

impl ChatParser {
    /// Parse the full generated text accumulated so far with lenient partial semantics.
    ///
    /// # Errors
    ///
    /// Returns an error when llama.cpp rejects or cannot project the partial parse.
    pub fn parse_partial(
        &self,
        generated_text: &str,
    ) -> Result<ParsedChatMessage, CommonChatError> {
        self.parse(generated_text, true)
    }

    /// Parse the complete generated text with final validation.
    ///
    /// # Errors
    ///
    /// Returns an error when the generated text is incomplete or invalid for the parser.
    pub fn parse_final(&self, generated_text: &str) -> Result<ParsedChatMessage, CommonChatError> {
        self.parse(generated_text, false)
    }

    fn parse(
        &self,
        generated_text: &str,
        is_partial: bool,
    ) -> Result<ParsedChatMessage, CommonChatError> {
        let generated_text = generated_text.as_bytes();
        let mut raw = ptr::null_mut();
        let mut error = ptr::null_mut();
        let status = unsafe {
            sys::llama_rs_chat_parser_parse(
                self.raw.as_ptr(),
                slice_ptr(generated_text),
                generated_text.len(),
                is_partial,
                &raw mut raw,
                &raw mut error,
            )
        };
        check_native_status(status, error)?;
        let raw = NonNull::new(raw).ok_or(CommonChatError::NullOutput("chat parse result"))?;
        parsed_message_from_raw(raw)
    }
}

impl Drop for ChatParser {
    fn drop(&mut self) {
        unsafe { sys::llama_rs_chat_parser_free(self.raw.as_ptr()) };
    }
}

/// A semantic change produced by llama.cpp's incremental chat parser.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ChatSemanticDelta {
    /// Newly visible reasoning text.
    Reasoning(String),
    /// Newly visible assistant content.
    Content(String),
    /// A tool-call header or argument fragment.
    ToolCall {
        /// Zero-based tool-call index in the parsed assistant message.
        index: usize,
        /// Model-provided call identifier when this delta carries it.
        id: Option<String>,
        /// Function name when this delta carries it.
        name: Option<String>,
        /// Newly parsed JSON argument text. Header-only deltas use an empty string.
        arguments: String,
    },
}

/// Stateful owner of llama.cpp's incremental semantic chat parser.
///
/// Input must already be assembled into valid UTF-8. Token-byte buffering belongs
/// to the inference executor because a single token piece may end inside a Unicode
/// scalar value.
#[derive(Debug)]
pub struct ChatStreamParser {
    raw: NonNull<sys::llama_rs_chat_stream>,
}

impl ChatStreamParser {
    /// Append generated text and return semantic changes since the previous push.
    ///
    /// # Errors
    ///
    /// Returns an error when the native parser rejects input or returns an invalid projection.
    pub fn push(&mut self, text: &str) -> Result<Vec<ChatSemanticDelta>, CommonChatError> {
        let bytes = text.as_bytes();
        let mut raw_batch = ptr::null_mut();
        let mut error = ptr::null_mut();
        let status = unsafe {
            sys::llama_rs_chat_stream_push(
                self.raw.as_ptr(),
                slice_ptr(bytes),
                bytes.len(),
                &raw mut raw_batch,
                &raw mut error,
            )
        };
        if status != sys::LLAMA_RS_STATUS_OK && !raw_batch.is_null() {
            unsafe { sys::llama_rs_chat_delta_batch_free(raw_batch) };
            raw_batch = ptr::null_mut();
        }
        check_native_status(status, error)?;
        let raw_batch =
            NonNull::new(raw_batch).ok_or(CommonChatError::NullOutput("chat delta batch"))?;
        semantic_deltas_from_raw(raw_batch)
    }

    /// Finalize parsing, returning the final structured message and any final deltas.
    ///
    /// Finalization can fail for incomplete structured output. Such a failure is
    /// transactional: callers may append more text and retry.
    ///
    /// # Errors
    ///
    /// Returns an error for incomplete output, invalid native projections, or a stale commit.
    pub fn finish(
        &mut self,
    ) -> Result<(ParsedChatMessage, Vec<ChatSemanticDelta>), CommonChatError> {
        let mut raw_finish = ptr::null_mut();
        let mut error = ptr::null_mut();
        let status = unsafe {
            sys::llama_rs_chat_stream_prepare_finish(
                self.raw.as_ptr(),
                &raw mut raw_finish,
                &raw mut error,
            )
        };
        check_native_status(status, error)?;

        let finish = FinishHandle(
            NonNull::new(raw_finish).ok_or(CommonChatError::NullOutput("prepared chat finish"))?,
        );
        let raw_batch = unsafe { sys::llama_rs_chat_finish_delta_batch(finish.0.as_ptr()) };
        if raw_batch.is_null() {
            return Err(CommonChatError::NullOutput("final chat delta batch"));
        }
        let raw_final = unsafe { sys::llama_rs_chat_finish_parse_result(finish.0.as_ptr()) };
        if raw_final.is_null() {
            return Err(CommonChatError::NullOutput("final parsed chat message"));
        }

        // Both projections are copied while the finish owner is alive. Any
        // fallible Rust allocation or validation therefore happens before the
        // terminal native state transition.
        let final_message = parsed_message_from_borrowed(raw_final)?;
        let deltas = semantic_deltas_from_ptr(raw_batch)?;

        let mut error = ptr::null_mut();
        let status = unsafe {
            sys::llama_rs_chat_stream_commit_finish(
                self.raw.as_ptr(),
                finish.0.as_ptr(),
                &raw mut error,
            )
        };
        check_native_status(status, error)?;
        Ok((final_message, deltas))
    }
}

impl Drop for ChatStreamParser {
    fn drop(&mut self) {
        unsafe { sys::llama_rs_chat_stream_free(self.raw.as_ptr()) };
    }
}

/// Fully owned semantic output from a partial or final native parse.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParsedChatMessage {
    /// Parsed role, normally `assistant`.
    pub role: String,
    /// Visible response content.
    pub content: String,
    /// Extracted reasoning content.
    pub reasoning_content: Option<String>,
    /// Parsed function tool calls.
    pub tool_calls: Vec<ChatToolCall>,
    /// Function name on a tool-result message, if present.
    pub tool_name: Option<String>,
    /// Tool call identifier on a tool-result message, if present.
    pub tool_call_id: Option<String>,
}

struct EncodedContentPart {
    kind: CString,
    text: CString,
}

impl EncodedContentPart {
    fn new(value: &ChatContentPart) -> Result<Self, CommonChatError> {
        Ok(Self {
            kind: CString::new(value.kind.as_str())?,
            text: CString::new(value.text.as_str())?,
        })
    }

    fn raw(&self) -> sys::llama_rs_chat_content_part_input {
        sys::llama_rs_chat_content_part_input {
            type_: self.kind.as_ptr(),
            text: self.text.as_ptr(),
        }
    }
}

struct EncodedToolCall {
    name: CString,
    arguments: CString,
    id: Option<CString>,
}

impl EncodedToolCall {
    fn new(value: &ChatToolCall) -> Result<Self, CommonChatError> {
        Ok(Self {
            name: CString::new(value.name.as_str())?,
            arguments: CString::new(value.arguments.as_str())?,
            id: optional_c_string(value.id.as_deref())?,
        })
    }

    fn raw(&self) -> sys::llama_rs_chat_tool_call_input {
        sys::llama_rs_chat_tool_call_input {
            name: self.name.as_ptr(),
            arguments: self.arguments.as_ptr(),
            id: c_string_ptr(self.id.as_ref()),
        }
    }
}

struct EncodedMessage {
    role: CString,
    content: Option<CString>,
    content_parts: Vec<EncodedContentPart>,
    raw_content_parts: Vec<sys::llama_rs_chat_content_part_input>,
    tool_calls: Vec<EncodedToolCall>,
    raw_tool_calls: Vec<sys::llama_rs_chat_tool_call_input>,
    reasoning_content: Option<CString>,
    tool_name: Option<CString>,
    tool_call_id: Option<CString>,
}

impl EncodedMessage {
    fn new(value: &ChatMessage) -> Result<Self, CommonChatError> {
        let (content, content_parts) = match &value.content {
            None => (None, Vec::new()),
            Some(ChatContent::Text(text)) => (Some(CString::new(text.as_str())?), Vec::new()),
            Some(ChatContent::Parts(parts)) => (
                None,
                parts
                    .iter()
                    .map(EncodedContentPart::new)
                    .collect::<Result<Vec<_>, _>>()?,
            ),
        };
        let raw_content_parts = content_parts.iter().map(EncodedContentPart::raw).collect();
        let tool_calls = value
            .tool_calls
            .iter()
            .map(EncodedToolCall::new)
            .collect::<Result<Vec<_>, _>>()?;
        let raw_tool_calls = tool_calls.iter().map(EncodedToolCall::raw).collect();
        Ok(Self {
            role: CString::new(value.role.as_str())?,
            content,
            content_parts,
            raw_content_parts,
            tool_calls,
            raw_tool_calls,
            reasoning_content: optional_c_string(value.reasoning_content.as_deref())?,
            tool_name: optional_c_string(value.tool_name.as_deref())?,
            tool_call_id: optional_c_string(value.tool_call_id.as_deref())?,
        })
    }

    fn raw(&self) -> sys::llama_rs_chat_message_input {
        // These owners keep the pointers in the raw projection valid for the call.
        let _ = (&self.content_parts, &self.tool_calls);
        sys::llama_rs_chat_message_input {
            role: self.role.as_ptr(),
            content: c_string_ptr(self.content.as_ref()),
            content_parts: slice_ptr(&self.raw_content_parts),
            content_parts_count: self.raw_content_parts.len(),
            tool_calls: slice_ptr(&self.raw_tool_calls),
            tool_calls_count: self.raw_tool_calls.len(),
            reasoning_content: c_string_ptr(self.reasoning_content.as_ref()),
            tool_name: c_string_ptr(self.tool_name.as_ref()),
            tool_call_id: c_string_ptr(self.tool_call_id.as_ref()),
        }
    }
}

struct EncodedTool {
    name: CString,
    description: CString,
    parameters_json: CString,
}

impl EncodedTool {
    fn new(value: &ChatTool) -> Result<Self, CommonChatError> {
        Ok(Self {
            name: CString::new(value.name.as_str())?,
            description: CString::new(value.description.as_str())?,
            parameters_json: CString::new(value.parameters_json.as_str())?,
        })
    }

    fn raw(&self) -> sys::llama_rs_chat_tool_input {
        sys::llama_rs_chat_tool_input {
            name: self.name.as_ptr(),
            description: self.description.as_ptr(),
            parameters_json: self.parameters_json.as_ptr(),
        }
    }
}

struct EncodedKwarg {
    key: CString,
    value_json: CString,
}

impl EncodedKwarg {
    fn new(value: &ChatTemplateKwarg) -> Result<Self, CommonChatError> {
        Ok(Self {
            key: CString::new(value.key.as_str())?,
            value_json: CString::new(value.value_json.as_str())?,
        })
    }

    fn raw(&self) -> sys::llama_rs_chat_template_kwarg_input {
        sys::llama_rs_chat_template_kwarg_input {
            key: self.key.as_ptr(),
            value_json: self.value_json.as_ptr(),
        }
    }
}

#[allow(clippy::struct_excessive_bools)] // Owns the exact native option projection.
struct EncodedPrepareOptions {
    messages: Vec<EncodedMessage>,
    raw_messages: Vec<sys::llama_rs_chat_message_input>,
    grammar: Option<CString>,
    json_schema: Option<CString>,
    tools: Vec<EncodedTool>,
    raw_tools: Vec<sys::llama_rs_chat_tool_input>,
    kwargs: Vec<EncodedKwarg>,
    raw_kwargs: Vec<sys::llama_rs_chat_template_kwarg_input>,
    add_generation_prompt: bool,
    continuation: ChatContinuation,
    use_jinja: bool,
    tool_choice: ChatToolChoice,
    parallel_tool_calls: Option<bool>,
    reasoning_format: ChatReasoningFormat,
    enable_thinking: Option<bool>,
    force_pure_content: bool,
}

impl EncodedPrepareOptions {
    fn new(options: &ChatPrepareOptions) -> Result<Self, CommonChatError> {
        let messages = options
            .messages
            .iter()
            .map(EncodedMessage::new)
            .collect::<Result<Vec<_>, _>>()?;
        let raw_messages = messages.iter().map(EncodedMessage::raw).collect();
        let tools = options
            .tools
            .iter()
            .map(EncodedTool::new)
            .collect::<Result<Vec<_>, _>>()?;
        let raw_tools = tools.iter().map(EncodedTool::raw).collect();
        let kwargs = options
            .template_kwargs
            .iter()
            .map(EncodedKwarg::new)
            .collect::<Result<Vec<_>, _>>()?;
        let raw_kwargs = kwargs.iter().map(EncodedKwarg::raw).collect();
        Ok(Self {
            messages,
            raw_messages,
            grammar: optional_c_string(options.grammar.as_deref())?,
            json_schema: optional_c_string(options.json_schema.as_deref())?,
            tools,
            raw_tools,
            kwargs,
            raw_kwargs,
            add_generation_prompt: options.add_generation_prompt,
            continuation: options.continuation,
            use_jinja: options.use_jinja,
            tool_choice: options.tool_choice,
            parallel_tool_calls: options.parallel_tool_calls,
            reasoning_format: options.reasoning_format,
            enable_thinking: options.enable_thinking,
            force_pure_content: options.force_pure_content,
        })
    }

    fn raw(&self) -> sys::llama_rs_chat_prepare_options {
        // These owners keep the pointers in the raw projection valid for the call.
        let _ = (&self.messages, &self.tools, &self.kwargs);
        sys::llama_rs_chat_prepare_options {
            messages: slice_ptr(&self.raw_messages),
            messages_count: self.raw_messages.len(),
            grammar: c_string_ptr(self.grammar.as_ref()),
            json_schema: c_string_ptr(self.json_schema.as_ref()),
            add_generation_prompt: self.add_generation_prompt,
            continuation: self.continuation.raw(),
            use_jinja: self.use_jinja,
            tools: slice_ptr(&self.raw_tools),
            tools_count: self.raw_tools.len(),
            tool_choice: self.tool_choice.raw(),
            parallel_tool_calls_set: self.parallel_tool_calls.is_some(),
            parallel_tool_calls: self.parallel_tool_calls.unwrap_or(false),
            reasoning_format: self.reasoning_format.raw(),
            enable_thinking: self.enable_thinking.unwrap_or(true),
            template_kwargs: slice_ptr(&self.raw_kwargs),
            template_kwargs_count: self.raw_kwargs.len(),
            force_pure_content: self.force_pure_content,
        }
    }
}

struct PreparedHandle(Option<NonNull<sys::llama_rs_chat_prepared>>);

impl PreparedHandle {
    fn new(raw: NonNull<sys::llama_rs_chat_prepared>) -> Self {
        Self(Some(raw))
    }

    fn as_ptr(&self) -> *mut sys::llama_rs_chat_prepared {
        self.0.expect("prepared handle is present").as_ptr()
    }

    fn release(mut self) -> NonNull<sys::llama_rs_chat_prepared> {
        self.0.take().expect("prepared handle is present")
    }
}

impl Drop for PreparedHandle {
    fn drop(&mut self) {
        if let Some(raw) = self.0 {
            unsafe { sys::llama_rs_chat_prepared_free(raw.as_ptr()) };
        }
    }
}

struct ParseResultHandle(NonNull<sys::llama_rs_chat_parse_result>);

impl Drop for ParseResultHandle {
    fn drop(&mut self) {
        unsafe { sys::llama_rs_chat_parse_result_free(self.0.as_ptr()) };
    }
}

struct DeltaBatchHandle(NonNull<sys::llama_rs_chat_delta_batch>);

impl Drop for DeltaBatchHandle {
    fn drop(&mut self) {
        unsafe { sys::llama_rs_chat_delta_batch_free(self.0.as_ptr()) };
    }
}

struct FinishHandle(NonNull<sys::llama_rs_chat_finish>);

impl Drop for FinishHandle {
    fn drop(&mut self) {
        unsafe { sys::llama_rs_chat_finish_free(self.0.as_ptr()) };
    }
}

fn prepared_string(
    prepared: *const sys::llama_rs_chat_prepared,
    field: sys::llama_rs_chat_prepared_string,
) -> Result<String, CommonChatError> {
    get_owned_string(
        |output| unsafe { sys::llama_rs_chat_prepared_get_string(prepared, field, output) },
        "prepared chat string",
    )
}

fn prepared_string_list(
    prepared: *const sys::llama_rs_chat_prepared,
    list: sys::llama_rs_chat_prepared_string_list,
) -> Result<Vec<String>, CommonChatError> {
    let count = unsafe { sys::llama_rs_chat_prepared_string_list_count(prepared, list) };
    (0..count)
        .map(|index| {
            get_owned_string(
                |output| unsafe {
                    sys::llama_rs_chat_prepared_string_list_get(prepared, list, index, output)
                },
                "prepared chat list item",
            )
        })
        .collect()
}

fn prepared_grammar_triggers(
    prepared: *const sys::llama_rs_chat_prepared,
) -> Result<Vec<ChatGrammarTrigger>, CommonChatError> {
    let count = unsafe { sys::llama_rs_chat_prepared_grammar_trigger_count(prepared) };
    (0..count)
        .map(|index| {
            let mut raw = sys::llama_rs_chat_grammar_trigger {
                type_: sys::LLAMA_RS_CHAT_GRAMMAR_TRIGGER_TOKEN,
                value: ptr::null_mut(),
                token: 0,
            };
            let status = unsafe {
                sys::llama_rs_chat_prepared_grammar_trigger_get(prepared, index, &raw mut raw)
            };
            check_status_without_error(status)?;
            let value = take_owned_string(raw.value, "grammar trigger value")?;
            raw.value = ptr::null_mut();
            match raw.type_ {
                sys::LLAMA_RS_CHAT_GRAMMAR_TRIGGER_TOKEN => Ok(ChatGrammarTrigger::Token {
                    value,
                    token: raw.token,
                }),
                sys::LLAMA_RS_CHAT_GRAMMAR_TRIGGER_WORD => Ok(ChatGrammarTrigger::Word(value)),
                sys::LLAMA_RS_CHAT_GRAMMAR_TRIGGER_PATTERN => {
                    Ok(ChatGrammarTrigger::Pattern(value))
                }
                sys::LLAMA_RS_CHAT_GRAMMAR_TRIGGER_PATTERN_FULL => {
                    Ok(ChatGrammarTrigger::PatternFull(value))
                }
                kind => Err(CommonChatError::UnknownEnum {
                    kind: "grammar trigger",
                    value: i64::from(kind),
                }),
            }
        })
        .collect()
}

fn parsed_message_from_raw(
    raw: NonNull<sys::llama_rs_chat_parse_result>,
) -> Result<ParsedChatMessage, CommonChatError> {
    let guard = ParseResultHandle(raw);
    parsed_message_from_borrowed(guard.0.as_ptr())
}

/// Copies every projection while the native parse-result owner is alive.
fn parsed_message_from_borrowed(
    raw: *const sys::llama_rs_chat_parse_result,
) -> Result<ParsedChatMessage, CommonChatError> {
    let role = parse_result_string(raw, sys::LLAMA_RS_CHAT_RESULT_ROLE)?;
    let content = parse_result_string(raw, sys::LLAMA_RS_CHAT_RESULT_CONTENT)?;
    let reasoning_content = empty_to_none(parse_result_string(
        raw,
        sys::LLAMA_RS_CHAT_RESULT_REASONING_CONTENT,
    )?);
    let tool_name = empty_to_none(parse_result_string(
        raw,
        sys::LLAMA_RS_CHAT_RESULT_TOOL_NAME,
    )?);
    let tool_call_id = empty_to_none(parse_result_string(
        raw,
        sys::LLAMA_RS_CHAT_RESULT_TOOL_CALL_ID,
    )?);
    let count = unsafe { sys::llama_rs_chat_parse_result_tool_call_count(raw) };
    let mut tool_calls = Vec::new();
    tool_calls
        .try_reserve_exact(count)
        .map_err(|source| CommonChatError::ParsedToolCallAllocation { count, source })?;
    for index in 0..count {
        let mut call = sys::llama_rs_chat_tool_call {
            name: empty_bytes_view(),
            arguments: empty_bytes_view(),
            id: empty_bytes_view(),
        };
        let status =
            unsafe { sys::llama_rs_chat_parse_result_tool_call_get(raw, index, &raw mut call) };
        check_status_without_error(status)?;
        tool_calls.push(ChatToolCall {
            name: bytes_view_to_string(call.name, "tool call name")?,
            arguments: bytes_view_to_string(call.arguments, "tool call arguments")?,
            id: empty_to_none(bytes_view_to_string(call.id, "tool call id")?),
        });
    }

    Ok(ParsedChatMessage {
        role,
        content,
        reasoning_content,
        tool_calls,
        tool_name,
        tool_call_id,
    })
}

fn semantic_deltas_from_raw(
    raw: NonNull<sys::llama_rs_chat_delta_batch>,
) -> Result<Vec<ChatSemanticDelta>, CommonChatError> {
    let guard = DeltaBatchHandle(raw);
    semantic_deltas_from_ptr(guard.0.as_ptr())
}

fn semantic_deltas_from_ptr(
    raw: *const sys::llama_rs_chat_delta_batch,
) -> Result<Vec<ChatSemanticDelta>, CommonChatError> {
    let count = unsafe { sys::llama_rs_chat_delta_batch_count(raw) };
    let mut result = Vec::new();
    result
        .try_reserve_exact(count)
        .map_err(|source| CommonChatError::DeltaAllocation { count, source })?;
    for index in 0..count {
        let mut delta = sys::llama_rs_chat_delta {
            kind: sys::LLAMA_RS_CHAT_DELTA_CONTENT,
            tool_call_index: 0,
            has_tool_call_id: false,
            has_tool_name: false,
            tool_call_id: sys::llama_rs_bytes_view {
                data: ptr::null(),
                len: 0,
            },
            tool_name: sys::llama_rs_bytes_view {
                data: ptr::null(),
                len: 0,
            },
            text: sys::llama_rs_bytes_view {
                data: ptr::null(),
                len: 0,
            },
        };
        let mut error = ptr::null_mut();
        let status = unsafe {
            sys::llama_rs_chat_delta_batch_get(raw, index, &raw mut delta, &raw mut error)
        };
        check_native_status(status, error)?;

        let text = bytes_view_to_string(delta.text, "chat delta text")?;
        let decoded = match delta.kind {
            sys::LLAMA_RS_CHAT_DELTA_REASONING => {
                ensure_non_tool_projection(&delta)?;
                ChatSemanticDelta::Reasoning(text)
            }
            sys::LLAMA_RS_CHAT_DELTA_CONTENT => {
                ensure_non_tool_projection(&delta)?;
                ChatSemanticDelta::Content(text)
            }
            sys::LLAMA_RS_CHAT_DELTA_TOOL_CALL => {
                let id = delta
                    .has_tool_call_id
                    .then(|| bytes_view_to_string(delta.tool_call_id, "tool call id"))
                    .transpose()?;
                if !delta.has_tool_call_id && delta.tool_call_id.len != 0 {
                    return Err(CommonChatError::InvalidNativeView("absent tool call id"));
                }
                let name = delta
                    .has_tool_name
                    .then(|| bytes_view_to_string(delta.tool_name, "tool call name"))
                    .transpose()?;
                if !delta.has_tool_name && delta.tool_name.len != 0 {
                    return Err(CommonChatError::InvalidNativeView("absent tool call name"));
                }
                ChatSemanticDelta::ToolCall {
                    index: delta.tool_call_index,
                    id,
                    name,
                    arguments: text,
                }
            }
            kind => {
                return Err(CommonChatError::UnknownEnum {
                    kind: "chat semantic delta",
                    value: i64::from(kind),
                });
            }
        };
        result.push(decoded);
    }
    Ok(result)
}

fn ensure_non_tool_projection(delta: &sys::llama_rs_chat_delta) -> Result<(), CommonChatError> {
    if delta.has_tool_call_id
        || delta.has_tool_name
        || delta.tool_call_id.len != 0
        || delta.tool_name.len != 0
    {
        Err(CommonChatError::InvalidNativeView(
            "non-tool semantic delta fields",
        ))
    } else {
        Ok(())
    }
}

fn bytes_view_to_string(
    view: sys::llama_rs_bytes_view,
    field: &'static str,
) -> Result<String, CommonChatError> {
    if view.len == 0 {
        return Ok(String::new());
    }
    if view.data.is_null() {
        return Err(CommonChatError::InvalidNativeView(field));
    }
    let bytes = unsafe { slice::from_raw_parts(view.data, view.len) };
    Ok(String::from_utf8(bytes.to_vec())?)
}

fn parse_result_string(
    result: *const sys::llama_rs_chat_parse_result,
    field: sys::llama_rs_chat_result_string,
) -> Result<String, CommonChatError> {
    let mut view = empty_bytes_view();
    let status =
        unsafe { sys::llama_rs_chat_parse_result_get_string(result, field, &raw mut view) };
    check_status_without_error(status)?;
    bytes_view_to_string(view, "parsed chat string")
}

fn empty_bytes_view() -> sys::llama_rs_bytes_view {
    sys::llama_rs_bytes_view {
        data: ptr::null(),
        len: 0,
    }
}

fn get_owned_string(
    call: impl FnOnce(*mut *mut c_char) -> sys::llama_rs_status,
    output_name: &'static str,
) -> Result<String, CommonChatError> {
    let mut value = ptr::null_mut();
    let status = call(&raw mut value);
    check_status_without_error(status)?;
    take_owned_string(value, output_name)
}

fn check_native_status(
    status: sys::llama_rs_status,
    error: *mut c_char,
) -> Result<(), CommonChatError> {
    if status == sys::LLAMA_RS_STATUS_OK {
        if !error.is_null() {
            unsafe { sys::llama_rs_string_free(error) };
        }
        return Ok(());
    }
    let message = if error.is_null() {
        String::new()
    } else {
        take_owned_string_lossy(error)
    };
    match status {
        sys::LLAMA_RS_STATUS_INVALID_UTF8 => Err(CommonChatError::InvalidNativeUtf8 { message }),
        sys::LLAMA_RS_STATUS_INVALID_STATE => Err(CommonChatError::InvalidStreamState { message }),
        _ => Err(CommonChatError::Native { status, message }),
    }
}

fn check_status_without_error(status: sys::llama_rs_status) -> Result<(), CommonChatError> {
    if status == sys::LLAMA_RS_STATUS_OK {
        Ok(())
    } else {
        Err(CommonChatError::Native {
            status,
            message: String::new(),
        })
    }
}

fn take_owned_string(
    value: *mut c_char,
    output_name: &'static str,
) -> Result<String, CommonChatError> {
    if value.is_null() {
        return Err(CommonChatError::NullOutput(output_name));
    }
    let bytes = unsafe { CStr::from_ptr(value).to_bytes().to_vec() };
    unsafe { sys::llama_rs_string_free(value) };
    Ok(String::from_utf8(bytes)?)
}

fn take_owned_string_lossy(value: *mut c_char) -> String {
    let message = unsafe { CStr::from_ptr(value).to_string_lossy().into_owned() };
    unsafe { sys::llama_rs_string_free(value) };
    message
}

fn optional_c_string(value: Option<&str>) -> Result<Option<CString>, NulError> {
    value.map(CString::new).transpose()
}

fn c_string_ptr(value: Option<&CString>) -> *const c_char {
    value.map_or(ptr::null(), |value| value.as_ptr())
}

fn slice_ptr<T>(values: &[T]) -> *const T {
    if values.is_empty() {
        ptr::null()
    } else {
        values.as_ptr()
    }
}

fn empty_to_none(value: String) -> Option<String> {
    if value.is_empty() {
        None
    } else {
        Some(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHATML: &str = r"{%- for message in messages -%}
{{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' -}}
{%- endfor -%}
{%- if add_generation_prompt -%}
{{- '<|im_start|>assistant\n' -}}
{%- endif -%}";

    // Keep this fixture in the safe crate so its published package tests do not
    // depend on a sibling sys-crate checkout being present.
    const QWEN3: &str = include_str!("../tests/fixtures/Qwen-Qwen3-0.6B.jinja");

    #[test]
    fn template_override_prepares_and_parses_content() {
        let templates = CommonChatTemplates::from_template(CHATML, None, None).unwrap();
        assert!(templates.was_explicit());
        assert_eq!(templates.source(None).unwrap(), CHATML);

        let capabilities = templates.capabilities().unwrap();
        assert!(capabilities.supports_string_content);
        assert!(!capabilities.supports_enable_thinking);

        let prepared = templates
            .prepare(&ChatPrepareOptions {
                messages: vec![ChatMessage::user("Hello")],
                reasoning_format: ChatReasoningFormat::None,
                ..ChatPrepareOptions::default()
            })
            .unwrap();
        assert_eq!(
            prepared.prompt(),
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        );
        assert!(!prepared.parser_definition().is_empty());

        let parser = prepared.parser(ChatParserOptions::default()).unwrap();
        assert_eq!(parser.parse_partial("Hel").unwrap().content, "Hel");
        assert_eq!(
            parser.parse_final("Hello there").unwrap().content,
            "Hello there"
        );
    }

    #[test]
    fn snapshot_content_parser_accepts_length_bearing_nul_input() {
        let templates = CommonChatTemplates::from_template(CHATML, None, None).unwrap();
        let prepared = templates
            .prepare(&ChatPrepareOptions {
                reasoning_format: ChatReasoningFormat::None,
                ..ChatPrepareOptions::default()
            })
            .unwrap();
        let parser = prepared.parser(ChatParserOptions::default()).unwrap();
        assert_eq!(
            parser.parse_partial("left\0right").unwrap().content,
            "left\0right"
        );
    }

    #[test]
    fn qwen_parser_extracts_reasoning_and_tool_calls() {
        let templates = CommonChatTemplates::from_template(QWEN3, None, None).unwrap();
        let prepared = templates
            .prepare(&ChatPrepareOptions {
                messages: vec![ChatMessage::user("What is the weather in Paris?")],
                tools: vec![ChatTool {
                    name: "get_weather".into(),
                    description: "Get the current weather".into(),
                    parameters_json:
                        r#"{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}"#
                            .into(),
                }],
                reasoning_format: ChatReasoningFormat::DeepSeek,
                ..ChatPrepareOptions::default()
            })
            .unwrap();
        assert!(prepared.supports_thinking());
        assert!(!prepared.grammar().is_empty());
        assert!(!prepared.grammar_triggers().is_empty());

        let parser = prepared.parser(ChatParserOptions::default()).unwrap();
        let partial = parser.parse_partial("<think>\nI should check").unwrap();
        assert_eq!(partial.reasoning_content.as_deref(), Some("I should check"));

        let parsed = parser
            .parse_final(
                "<think>\nI should check\n</think>\n\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\":\"Paris\"}}\n</tool_call>",
            )
            .unwrap();
        assert_eq!(
            parsed.reasoning_content.as_deref(),
            Some("I should check\n")
        );
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "get_weather");
        assert_eq!(parsed.tool_calls[0].arguments, r#"{"city":"Paris"}"#);
    }

    #[test]
    fn stateful_stream_emits_content_and_finishes_exactly_once() {
        let templates = CommonChatTemplates::from_template(CHATML, None, None).unwrap();
        let prepared = templates
            .prepare(&ChatPrepareOptions {
                messages: vec![ChatMessage::user("Hello")],
                reasoning_format: ChatReasoningFormat::None,
                ..ChatPrepareOptions::default()
            })
            .unwrap();
        let mut stream = prepared
            .stream_parser(ChatParserOptions::default())
            .unwrap();

        assert!(stream.push("").unwrap().is_empty());
        assert_eq!(
            stream.push("Hel").unwrap(),
            vec![ChatSemanticDelta::Content("Hel".into())]
        );
        assert_eq!(
            stream.push("lo ").unwrap(),
            vec![ChatSemanticDelta::Content("lo ".into())]
        );
        assert_eq!(
            stream.push("世界").unwrap(),
            vec![ChatSemanticDelta::Content("世界".into())]
        );

        let (final_message, final_deltas) = stream.finish().unwrap();
        assert_eq!(final_message.content, "Hello 世界");
        assert!(final_deltas.is_empty());
        assert!(matches!(
            stream.push("again"),
            Err(CommonChatError::InvalidStreamState { .. })
        ));
        assert!(matches!(
            stream.finish(),
            Err(CommonChatError::InvalidStreamState { .. })
        ));
    }

    #[test]
    fn stateful_stream_is_invariant_across_valid_utf8_chunk_splits() {
        let templates = CommonChatTemplates::from_template(CHATML, None, None).unwrap();
        let prepared = templates
            .prepare(&ChatPrepareOptions {
                reasoning_format: ChatReasoningFormat::None,
                ..ChatPrepareOptions::default()
            })
            .unwrap();
        let output = "alpha 世界 omega";
        let split_points = output
            .char_indices()
            .map(|(index, _)| index)
            .chain(std::iter::once(output.len()));

        for split in split_points {
            let mut stream = prepared
                .stream_parser(ChatParserOptions::default())
                .unwrap();
            let mut reconstructed = String::new();
            for delta in stream.push(&output[..split]).unwrap() {
                if let ChatSemanticDelta::Content(text) = delta {
                    reconstructed.push_str(&text);
                }
            }
            for delta in stream.push(&output[split..]).unwrap() {
                if let ChatSemanticDelta::Content(text) = delta {
                    reconstructed.push_str(&text);
                }
            }
            let (final_message, final_deltas) = stream.finish().unwrap();
            for delta in final_deltas {
                if let ChatSemanticDelta::Content(text) = delta {
                    reconstructed.push_str(&text);
                }
            }
            assert_eq!(reconstructed, output, "split at byte {split}");
            assert_eq!(final_message.content, output, "split at byte {split}");
        }
    }

    #[test]
    fn stateful_qwen_stream_uses_native_reasoning_and_tool_diffs() {
        let templates = CommonChatTemplates::from_template(QWEN3, None, None).unwrap();
        let prepared = templates
            .prepare(&ChatPrepareOptions {
                messages: vec![ChatMessage::user("What is the weather in Paris?")],
                tools: vec![ChatTool {
                    name: "get_weather".into(),
                    description: "Get the current weather".into(),
                    parameters_json:
                        r#"{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}"#
                            .into(),
                }],
                reasoning_format: ChatReasoningFormat::DeepSeek,
                ..ChatPrepareOptions::default()
            })
            .unwrap();
        let chunks = [
            "<think>\nI should",
            " check\n</think>\n\n<tool_call>\n{\"name\": \"get_",
            "weather\", \"arguments\": {\"city\":\"Paris\"}}\n</tool_call>",
        ];
        let mut stream = prepared
            .stream_parser(ChatParserOptions::default())
            .unwrap();
        let mut saw_reasoning = false;
        let mut saw_tool_name = false;
        let mut argument_text = String::new();
        for chunk in chunks {
            for delta in stream.push(chunk).unwrap() {
                match delta {
                    ChatSemanticDelta::Reasoning(text) => saw_reasoning |= !text.is_empty(),
                    ChatSemanticDelta::ToolCall {
                        index,
                        name,
                        arguments,
                        ..
                    } => {
                        assert_eq!(index, 0);
                        saw_tool_name |= name.as_deref() == Some("get_weather");
                        argument_text.push_str(&arguments);
                    }
                    ChatSemanticDelta::Content(_) => {}
                }
            }
        }
        let (parsed, final_deltas) = stream.finish().unwrap();
        for delta in final_deltas {
            if let ChatSemanticDelta::ToolCall {
                name, arguments, ..
            } = delta
            {
                saw_tool_name |= name.as_deref() == Some("get_weather");
                argument_text.push_str(&arguments);
            }
        }
        assert!(saw_reasoning);
        assert!(saw_tool_name);
        assert_eq!(parsed.tool_calls.len(), 1);
        assert_eq!(parsed.tool_calls[0].name, "get_weather");
        assert_eq!(parsed.tool_calls[0].arguments, r#"{"city":"Paris"}"#);
        assert_eq!(argument_text, parsed.tool_calls[0].arguments);
    }

    #[test]
    fn continuation_without_echo_does_not_reemit_assistant_prefill() {
        let templates = CommonChatTemplates::from_template(QWEN3, None, None).unwrap();
        let prepared = templates
            .prepare(&ChatPrepareOptions {
                messages: vec![
                    ChatMessage::user("Complete the sentence"),
                    ChatMessage::assistant("The answer is"),
                ],
                continuation: ChatContinuation::Content,
                add_generation_prompt: false,
                reasoning_format: ChatReasoningFormat::None,
                ..ChatPrepareOptions::default()
            })
            .unwrap();
        let mut stream = prepared
            .stream_parser(ChatParserOptions {
                is_continuation: true,
                echo: false,
                ..ChatParserOptions::default()
            })
            .unwrap();
        let deltas = stream.push(" 42").unwrap();
        assert_eq!(deltas, vec![ChatSemanticDelta::Content(" 42".into())]);
        let (parsed, final_deltas) = stream.finish().unwrap();
        assert!(final_deltas.is_empty());
        assert!(parsed.content.ends_with("The answer is 42"));
    }

    #[test]
    fn rejects_conflicting_constraints_before_ffi() {
        let templates = CommonChatTemplates::from_template(CHATML, None, None).unwrap();
        let error = templates
            .prepare(&ChatPrepareOptions {
                grammar: Some("root ::= \"ok\"".into()),
                json_schema: Some(r#"{"type":"string"}"#.into()),
                ..ChatPrepareOptions::default()
            })
            .unwrap_err();
        assert!(matches!(error, CommonChatError::InvalidOptions(_)));
    }
}
