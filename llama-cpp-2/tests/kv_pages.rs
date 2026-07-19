//! Model-backed tests for Magnitude's fork-specific immutable KV page ABI.

use std::num::NonZeroU32;

use llama_cpp_2::context::kv_cache::KvPageError;
use llama_cpp_2::context::params::{FlashAttentionPolicy, LlamaContextParams};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};

#[test]
fn retained_page_attach_reproduces_source_logits_without_copying_kv() {
    let Ok(model_path) = std::env::var("ICN_TEST_MODEL") else {
        eprintln!("skipping model-backed page test: ICN_TEST_MODEL is unset");
        return;
    };
    let backend = LlamaBackend::init().expect("initialize llama backend");
    let model = LlamaModel::load_from_file(
        &backend,
        model_path,
        &LlamaModelParams::default().with_n_gpu_layers(0),
    )
    .expect("load test model");
    let params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(128))
        .with_n_batch(64)
        .with_n_ubatch(64)
        .with_n_seq_max(2)
        .with_kv_unified(true)
        .with_offload_kqv(false)
        .with_op_offload(false)
        .with_flash_attention(FlashAttentionPolicy::Disabled);
    let mut context = model.new_context(&backend, params).expect("create context");
    assert!(context.kv_pages_supported());

    let text = "radix prefix cache validation token stream ".repeat(24);
    let tokens = model
        .str_to_token(&text, AddBos::Always)
        .expect("tokenize fixture");
    assert!(tokens.len() > 32);

    let mut prefix = LlamaBatch::new(32, 1);
    for (position, token) in tokens[..32].iter().copied().enumerate() {
        prefix
            .add(token, position as i32, &[0], false)
            .expect("add prefix token");
    }
    context.decode(&mut prefix).expect("decode source prefix");

    let tail = tokens[32];
    let mut source_tail = LlamaBatch::new(1, 1);
    source_tail
        .add(tail, 32, &[0], true)
        .expect("add source tail");
    context
        .decode(&mut source_tail)
        .expect("decode source tail");
    context.synchronize();
    let source_logits = context.get_logits_ith(0).to_vec();

    let page = context.pin_kv_page(0, 0, 32).expect("pin source page");
    let blob = context.export_kv_page(page).expect("export source page");
    assert_eq!(context.kv_page_token_count(page), 32);
    assert_eq!(context.kv_page_stats().pinned_cells, 32);
    assert!(context
        .clear_kv_cache_seq(Some(0), None, None)
        .expect("clear source sequence"));
    assert_eq!(context.kv_page_stats().used_cells, 32);

    context
        .attach_kv_page(page, 1)
        .expect("attach page to independent destination");
    let mut attached_tail = LlamaBatch::new(1, 1);
    attached_tail
        .add(tail, 32, &[1], true)
        .expect("add attached tail");
    context
        .decode(&mut attached_tail)
        .expect("decode from attached prefix");
    context.synchronize();
    let attached_logits = context.get_logits_ith(0);

    let max_error = source_logits
        .iter()
        .zip(attached_logits)
        .map(|(source, attached)| (source - attached).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_error <= 1.0e-5,
        "attached logits diverged by {max_error}"
    );

    context.release_kv_page(page).expect("release page pin");
    assert_eq!(context.kv_page_stats().pinned_cells, 0);
    assert!(context
        .clear_kv_cache_seq(Some(1), None, None)
        .expect("clear destination sequence"));
    assert_eq!(context.kv_page_stats().used_cells, 0);

    let mut corrupt = blob.clone();
    corrupt[0] ^= 0xff;
    assert_eq!(
        context.import_kv_page(&corrupt).unwrap_err(),
        KvPageError::InvalidBlob
    );
    assert_eq!(
        context.import_kv_page(&blob[..blob.len() / 2]).unwrap_err(),
        KvPageError::InvalidBlob
    );
    assert_eq!(context.kv_page_stats().used_cells, 0);

    let imported = context.import_kv_page(&blob).expect("import page blob");
    assert_eq!(context.kv_page_token_count(imported), 32);
    context
        .attach_kv_page(imported, 0)
        .expect("attach imported page");
    let mut imported_tail = LlamaBatch::new(1, 1);
    imported_tail
        .add(tail, 32, &[0], true)
        .expect("add imported tail");
    context
        .decode(&mut imported_tail)
        .expect("decode from imported prefix");
    context.synchronize();
    let imported_logits = context.get_logits_ith(0);
    let imported_max_error = source_logits
        .iter()
        .zip(imported_logits)
        .map(|(source, imported)| (source - imported).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        imported_max_error <= 1.0e-5,
        "imported logits diverged by {imported_max_error}"
    );
    context
        .release_kv_page(imported)
        .expect("release imported page");
    assert!(context
        .clear_kv_cache_seq(Some(0), None, None)
        .expect("clear imported destination"));
    assert_eq!(context.kv_page_stats().used_cells, 0);
}
