# Magnitude llama-cpp-rs fork instructions

This repository is Magnitude's Rust compatibility layer over two upstreams:

- `https://github.com/utilityai/llama-cpp-rs` is the wrapper upstream.
- `llama-cpp-sys-2/llama.cpp` pins `https://github.com/ggml-org/llama.cpp`.

The containing Magnitude repository consumes an exact commit of this repository and records both
native revisions in `inference/native-pin.toml`.

Follow the containing repository's fork-approval policy. Do not change this fork merely for
convenience, and do not change the nested llama.cpp source unless the user explicitly approves it.

## Maintenance model

This is an intentional product fork. It may evolve incrementally and diverge from the wrapper
upstream when Magnitude needs capabilities that upstream does not provide. Approved binding work
does not need to wait for an upstream refresh, and useful Magnitude functionality must not be
removed merely to make the diff smaller.

There are two normal kinds of maintenance:

1. **Incremental development:** add, change, and fix narrowly scoped bindings on the current
   Magnitude generation as product work requires.
2. **Clean-slate refresh:** occasionally analyze the complete Magnitude divergence against the
   latest wrapper upstream, start a new generation from that upstream, adopt upstream replacements,
   reimplement only the adaptations that remain necessary, and update Magnitude consumers as
   needed.

Incremental work may add whatever binding functionality Magnitude requires, subject to the
containing repository's approval rules. During a refresh, evaluate each existing divergence
individually:

- If latest upstream now provides the required functionality, use its implementation and update
  Magnitude for any API differences.
- If Magnitude still needs functionality that upstream does not provide, reimplement the necessary
  adaptation against the latest upstream code with the smallest reasonable change.
- If the functionality is no longer required, remove it.

Do not preserve custom code merely because it already exists, and do not remove required
functionality merely to reduce the size of the fork.

## Clean-slate refreshes

Periodic upstream refreshes are clean-slate re-derivations, not merges of the previous Magnitude
delta. The goal is not zero divergence; it is a deliberately reviewed, minimal, current divergence
that provides all required Magnitude behavior. For a refresh:

1. Fetch the latest wrapper upstream. Before changing code, carefully analyze the current fork's
   divergence and turn it into a capability-level inventory. Review the diff and history, but
   describe what Magnitude requires rather than assuming each old line must survive:

   ```sh
   git remote get-url upstream >/dev/null 2>&1 || \
     git remote add upstream https://github.com/utilityai/llama-cpp-rs.git
   git fetch upstream --tags
   git diff --stat upstream/main...HEAD
   git diff upstream/main...HEAD
   git log --oneline upstream/main..HEAD
   ```

   For each substantial divergence, identify its Magnitude consumer and determine whether latest
   upstream now provides it fully, provides part of it, or still lacks it. Include behavioral and
   build adaptations that a compiler-only inventory may miss.

2. Create a new branch directly from the latest upstream `main`:

   ```sh
   git switch -c refresh/YYYY-MM-DD upstream/main
   git submodule update --init --recursive
   ```

3. Do **not** merge, rebase, or cherry-pick the old Magnitude bindings branch into the refresh
   branch. Do not apply its diff wholesale. The old fork is reference material only.

4. First build and test the unmodified wrapper at the llama.cpp revision selected by upstream. This
   separates an upstream baseline failure from a Magnitude adaptation failure.

5. If Magnitude needs a different llama.cpp revision, select an explicit commit. Never use a
   floating submodule revision as the submitted result. Build the otherwise-unmodified wrapper
   again after changing the gitlink.

6. Reconcile the capability inventory against latest upstream, then point the containing Magnitude
   checkout at the refresh worktree and compile its consumers. For every still-needed capability:

   - Search the current wrapper and pinned llama.cpp first.
   - If upstream now provides it, change Magnitude to use the upstream API.
   - If upstream provides part of it, add only the missing portion.
   - If upstream still lacks it, reimplement the smallest reasonable version that satisfies the
     current Magnitude requirement.
   - Add or identify a test that fails without the adaptation.

7. Restore capabilities independently. Do not copy whole files or old compatibility layers merely
   because they existed in the previous generation. When consulting old code, rewrite only the
   still-required behavior against the current upstream APIs.

8. Update Magnitude consumers as necessary when current upstream offers a better API or the minimal
   refreshed binding differs from the previous generation. Preserve Magnitude's documented behavior
   unless the user has approved a product-level change.

9. Audit the complete remaining fork delta:

   ```sh
   git diff --stat upstream/main...HEAD
   git diff upstream/main...HEAD
   ```

   Every material addition must have a concrete answer to: "Which Magnitude consumer, invariant,
   build target, or parity test fails if this is removed?" Remove additions without such evidence.

10. Publish the refresh commit to `magnitudedev/llama-cpp-rs`, then update the containing Magnitude
   repository's nested pointer and both revisions in `inference/native-pin.toml`. Never point
   Magnitude at an unpushed bindings commit.

Historical Magnitude pins must remain fetchable. Do not force-push away a commit referenced by a
released or shared Magnitude revision. A refresh branch or immutable tag may preserve each accepted
generation; the exact commit recorded by Magnitude is authoritative.

## Required validation

Run the wrapper checks relevant to the changed surfaces. The baseline feature matrix is:

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --features sampler,mtmd -- -D warnings
cargo test -p llama-cpp-2 --no-default-features
cargo test -p llama-cpp-2 --no-default-features --features common
cargo test -p llama-cpp-2 --no-default-features --features common,serde
cargo test -p llama-cpp-2 --no-default-features --features mtmd
cargo test -p llama-cpp-2 --no-default-features --features common,mtmd,sampler
```

Also build the hardware backends affected by the change. A CPU-only build does not validate CUDA,
Metal, Vulkan, OpenCL, ROCm, Android, Windows, or static/dynamic linkage changes.

From the containing Magnitude repository, run at least:

```sh
bun icn:check
bun icn:test
bun icn:parity:validate
bun icn:verify-native-pin
```

Run focused native parity for every volatile surface touched by the refresh, especially chat
templates, sampling, model/context defaults, device placement and fit, sequence state and memory,
MTP, and MTMD. Compilation alone is not evidence of behavioral compatibility.

## Incremental changes between refreshes

Approved functionality, fixes, and adaptations may be developed normally on the current Magnitude
generation without starting a clean-slate refresh. The fork is allowed to accumulate intentional
divergence between refreshes. Keep each change well reasoned, scoped to the requirement, tested, and
clear about why it belongs in the binding instead of Magnitude or upstream llama.cpp.

At the next refresh, carefully analyze these changes along with the rest of the fork. They receive
no automatic presumption of retention, but neither are they presumed obsolete: latest upstream
availability, current consumer needs, safety requirements, build support, and behavioral evidence
decide whether they are adopted from upstream, rewritten minimally, moved into Magnitude, or
removed.

## Review rules

- Prefer deletion when upstream has acquired equivalent behavior.
- Do not retain both an upstream implementation and a Magnitude compatibility implementation.
- Do not recreate llama.cpp algorithms or policy in Rust when a native API is authoritative.
- Keep C/C++ shims narrow and return owned, explicit data across unstable `common/` and MTMD
  boundaries where practical.
- Separate a llama.cpp gitlink bump from substantial adaptations when doing so makes review and
  failure attribution clearer.
- Treat the compiler as an API inventory and parity tests as the behavioral specification; do not
  treat the previous fork's source tree as the specification.
