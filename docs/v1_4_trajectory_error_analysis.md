# V1.4 GPT-5.4-mini Strict-Context Error Analysis

This document summarizes failure modes in the canonical `openai_gpt54mini_v2_strict_context` trajectory release. The examples are selected deterministically from trajectory details using metric-based rules, then rendered with query/gold/final context for inspection.

Source artifacts:

- `data/eval/v1_4/v1_3_all_openai_gpt54mini_v2_strict_context_trajectory_details.jsonl`
- `data/trajectory_runs/v1_4/v1_3_all_gpt54mini/traces_openai_gpt54mini_v2_strict_context/`
- `data/benchmark/v1_3/samples.jsonl`

## Error Overview

| Task | Samples | Exact misses | Partial hits | Full hits | File hit but line F1=0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| overall | 287 | 101 | 45 | 141 | 102 |
| code2test | 106 | 50 | 15 | 41 | 47 |
| comment2context | 80 | 47 | 17 | 16 | 19 |
| trace2code | 101 | 4 | 13 | 84 | 36 |

Definitions:

- Exact miss: `final_file_recall=0`.
- Partial hit: `0 < final_file_recall < 1`.
- Full hit: `final_file_recall=1`.
- File hit but line F1=0: the run selected at least one gold file but did not overlap labeled spans/blocks under the trajectory line metric.

## Patterns

### code2test source-to-test mapping misses

The model reads plausible implementation or nearby files but misses the exact regression/integration test targets.

- `120f8ebe03b4feba2ac74c51` (code2test, `etcd-io/etcd`):
  - query: {"changed_file_summary": "1 implementation files and 1 existing test files changed within 2 total files.", "implementation_file_count": 1, "implementation_files": ["server/etcdmain/grpc_proxy.go"], "pr_body": "This adds ...
  - gold: `tests/e2e/etcd_grpcproxy_test.go`
  - final/read: `server/etcdmain/grpc_proxy.go`, `server/etcdmain/help.go`, `server/embed/config.go`
  - metrics: final_recall=0.0000, final_precision=0.0000, line_f1=n/a
  - suggested_unread: `tests/e2e/etcd_config_test.go`, `tests/framework/e2e/cluster_proxy.go`, `pkg/proxy/server_test.go`

- `176cf95c04cde79a4e1ef372` (code2test, `vitejs/vite`):
  - query: {"changed_file_summary": "5 implementation files and 1 existing test files changed within 7 total files.", "implementation_file_count": 5, "implementation_files": ["packages/vite/src/node/plugins/prepareOutDir.ts", "play...
  - gold: `playground/assets/__tests__/assets.spec.ts`
  - support: `playground/assets/index.html`
  - final/read: `packages/vite/src/node/plugins/prepareOutDir.ts`, `packages/vite/src/node/watch.ts`, `playground/legacy/vite.config-watch.js`
  - metrics: final_recall=0.0000, final_precision=0.0000, line_f1=0.0000
  - suggested_unread: `packages/vite/src/node/build.ts`, `packages/vite/src/node/server/index.ts`, `packages/vite/src/node/__tests__/resolve.spec.ts`, `playground/legacy/__tests__/client-and-ssr/serve.ts`, `playground/ssr-conditions/__tests__/serve.ts`

- `1bb413019cf24674eb1c59b1` (code2test, `tokio-rs/tokio`):
  - query: {"changed_file_summary": "1 implementation files and 1 existing test files changed within 2 total files.", "implementation_file_count": 1, "implementation_files": ["tokio/src/runtime/scheduler/multi_thread/worker.rs"], "...
  - gold: `tokio/tests/rt_threaded.rs`
  - final/read: `tokio/src/runtime/scheduler/multi_thread/worker.rs`, `tokio/tests/task_blocking.rs`, `tokio/src/runtime/scheduler/defer.rs`
  - metrics: final_recall=0.0000, final_precision=0.0000, line_f1=n/a
  - suggested_unread: `tokio/src/runtime/scheduler/current_thread/mod.rs`, `tokio/src/runtime/scheduler/block_in_place.rs`, `tokio/src/runtime/scheduler/mod.rs`

### comment2context support-near but gold-miss cases

The model finds useful supporting context but not the exact gold file; this separates broad context usefulness from exact target recovery.

- `01323034afde8983e978b8d2` (comment2context, `astral-sh/ruff`):
  - query: review_comment: Could we move the `return_callable_typevar_scope` at the top-level (the functions that are salsa cached) to avoid adding another method?; path: crates/ty_python_semantic/src/types/function.rs; given_file: crates/ty_python_semantic/src/types/function.rs
  - gold: `crates/ty_python_semantic/src/types/infer/builder.rs`
  - support: `crates/ty_python_semantic/src/types/infer/builder/function.rs`
  - final/read: `crates/ty_python_semantic/src/types/function.rs`, `crates/ty_python_semantic/src/types/infer/builder/function.rs`, `crates/ty_python_semantic/src/types/infer/builder/paramspec_validation.rs`
  - metrics: final_recall=0.0000, final_precision=0.0000, line_f1=n/a
  - suggested_unread: `crates/ty_python_semantic/src/types/infer/builder/post_inference/function.rs`, `crates/ty_python_core/src/scope.rs`, `crates/ruff_linter/src/rules/flake8_return/rules/function.rs`

- `2f39489dd27f283deb0f7fae` (comment2context, `microsoft/playwright`):
  - query: review_comment: Why don't we destroy stdin here as well?; path: packages/playwright-core/src/tools/cli-client/program.ts; given_file: packages/playwright-core/src/tools/cli-client/program.ts
  - gold: `packages/playwright-core/src/tools/dashboard/dashboardApp.ts`
  - support: `packages/playwright-core/src/tools/cli-client/session.ts`
  - final/read: `packages/playwright-core/src/tools/cli-client/program.ts`, `packages/playwright-core/src/tools/cli-client/session.ts`, `packages/playwright-core/src/tools/cli-daemon/program.ts`
  - metrics: final_recall=0.0000, final_precision=0.0000, line_f1=0.0000
  - suggested_unread: `packages/playwright-core/src/outofprocess.ts`

- `e1280404f66f39671f1939e5` (comment2context, `etcd-io/etcd`):
  - query: review_comment: Ticker doesn't work as you think. Please read https://pkg.go.dev/time#NewTicker `The ticker will adjust the time interval or drop ticks to make up for slow receivers`; path: cache/cache.go; given_file: cache/cache.go
  - gold: `cache/cache_test.go`, `cache/demux_test.go`, `tests/integration/cache_test.go`
  - support: `cache/config.go`, `cache/demux.go`
  - final/read: `cache/demux.go`, `cache/progress_requestor.go`, `cache/config.go`
  - metrics: final_recall=0.0000, final_precision=0.0000, line_f1=n/a
  - suggested_unread: `cache/cache.go`, `cache/progress_requestor_test.go`, `cache/fake_clock_test.go`, `client/v3/watch.go`

### comment2context hard additional-context misses

The model does not recover gold or supporting context, usually because the review comment is a weak anchor for a cross-module policy/config/test dependency.

- `0a166358bf317488838726c8` (comment2context, `huggingface/transformers`):
  - query: review_comment: We could use `hasarttr` but it's not very nice to read. vLLM has a very similar utility called `getattr_iter` for checking multiple names in configs.; path: src/transformers/integrations/finegrained_fp8.py; given_file: src/transformers/integrations/finegrained_fp8.py
  - gold: `src/transformers/conversion_mapping.py`
  - final/read: `src/transformers/integrations/finegrained_fp8.py`, `src/transformers/quantizers/quantizer_finegrained_fp8.py`, `src/transformers/utils/generic.py`
  - metrics: final_recall=0.0000, final_precision=0.0000, line_f1=0.0000
  - suggested_unread: `src/transformers/integrations/hub_kernels.py`, `src/transformers/utils/import_utils.py`, `tests/quantization/finegrained_fp8/test_fp8.py`

- `106d7add9c91db6d52f8153a` (comment2context, `pydantic/pydantic`):
  - query: review_comment: @davidhewitt I see `UnionChoices` is also used for the `UnionSerializer`. Can we confirm that this PR won't affect the current (non tagged) union serialization behavior?; path: pydantic-core/src/serializers/type_serializers/union.rs; given_file: pydantic-core/src/serializers/type_serializers/union.rs
  - gold: `tests/test_discriminated_union.py`
  - support: `pydantic-core/tests/serializers/test_union.py`
  - final/read: `pydantic-core/src/serializers/type_serializers/union.rs`, `pydantic-core/src/common/union.rs`, `pydantic-core/src/serializers/type_serializers/function.rs`, `pydantic-core/src/validators/union.rs`
  - metrics: final_recall=0.0000, final_precision=0.0000, line_f1=0.0000
  - suggested_unread: `pydantic-core/tests/serializers/test_union.py`, `tests/test_discriminated_union.py`

- `128f9e22e10889d2d819753c` (comment2context, `scrapy/scrapy`):
  - query: review_comment: maybe keep the system time here too? since its a public attribute. although this class is deprecated, so idk; path: scrapy/core/downloader/webclient.py; given_file: scrapy/core/downloader/webclient.py
  - gold: `scrapy/extensions/corestats.py`
  - support: `scrapy/commands/check.py`, `scrapy/core/engine.py`, `scrapy/utils/engine.py`
  - final/read: `scrapy/core/downloader/webclient.py`, `scrapy/core/downloader/handlers/http10.py`, `tests/test_webclient.py`
  - metrics: final_recall=0.0000, final_precision=0.0000, line_f1=0.0000
  - suggested_unread: `scrapy/core/downloader/handlers/http11.py`, `scrapy/core/downloader/handlers/http2.py`, `scrapy/downloadermiddlewares/downloadtimeout.py`

### trace2code file-level hit but line-level miss

The model identifies at least one gold source file, but its file-level read window is too broad or not aligned with labeled spans/blocks.

- `035c3b77d35f10df09605eff` (trace2code, `gin-gonic/gin`):
  - query: failure_excerpt: $ go test ./. FAIL github.com/gin-gonic/gin [build failed] FAIL # github.com/gin-gonic/gin [github.com/gin-gonic/gin.test] ./logger_test.go:325:12: p.LatencyColor unde...
  - gold: `logger.go`
  - final/read: `logger.go`, `logger_test.go`, `gin.go`
  - metrics: final_recall=1.0000, final_precision=0.3333, line_f1=n/a
  - suggested_unread: `docs/doc.md`, `gin_integration_test.go`

- `0765b7d2a08dd5736ac91857` (trace2code, `gin-gonic/gin`):
  - query: failure_excerpt: $ go test ./binding --- FAIL: TestMappingMultipleDefaultWithCollectionFormat (0.00s) form_mapping_test.go:368: Error Trace: /Users/eyuansu62/llm_judge/a-good-project/d...
  - gold: `binding/form_mapping.go`
  - final/read: `binding/form_mapping_test.go`, `binding/form_mapping.go`, `binding/query.go`
  - metrics: final_recall=1.0000, final_precision=0.3333, line_f1=n/a
  - suggested_unread: `binding/form.go`, `binding/binding_test.go`, `binding/multipart_form_mapping.go`

- `12951fd6d477e49ce706a14b` (trace2code, `gin-gonic/gin`):
  - query: failure_excerpt: $ go test ./binding --- FAIL: TestMappingBaseTypes (0.00s) form_mapping_test.go:58: Error Trace: /Users/eyuansu62/llm_judge/a-good-project/data/repro_worktrees/gin-gon...
  - gold: `binding/form_mapping.go`
  - final/read: `binding/form_mapping_test.go`, `binding/form_mapping.go`, `binding/multipart_form_mapping.go`, `binding/multipart_form_mapping_test.go`
  - metrics: final_recall=1.0000, final_precision=0.2500, line_f1=n/a
  - suggested_unread: `binding/form.go`, `binding/binding_test.go`

### trace2code source-file misses

The model reads plausible files from the failure neighborhood but misses the root-cause source file.

- `05041faae6e19b6882e3074a` (trace2code, `gin-gonic/gin`):
  - query: failure_excerpt: $ go test ./. 2026/05/06 22:52:39 The AppEngine flag is going to be deprecated. Please check issues #2723 and #2739 and use 'TrustedPlatform: gin.PlatformGoogleAppEngi...
  - gold: `gin.go`
  - final/read: `routes_test.go`, `path.go`, `tree.go`
  - metrics: final_recall=0.0000, final_precision=0.0000, line_f1=n/a
  - suggested_unread: `gin.go`, `context.go`, `render/redirect.go`, `tree_test.go`

- `1132f037ebb462f13652adff` (trace2code, `tokio-rs/tokio`):
  - query: failure_excerpt: $ cargo test --features full,test-util -p tokio --test sync_mpsc ... ... ok test test_rx_unbounded_is_closed_when_there_are_no_senders_and_there_are_messages ... ok te...
  - gold: `tokio/src/sync/mpsc/block.rs`
  - final/read: `tokio/tests/sync_mpsc.rs`, `tokio/src/sync/mpsc/bounded.rs`, `tokio/src/sync/mpsc/chan.rs`, `tokio/src/sync/mpsc/list.rs`
  - metrics: final_recall=0.0000, final_precision=0.0000, line_f1=0.0000
  - suggested_unread: `tokio/src/sync/mpsc/unbounded.rs`, `tokio/src/sync/mpsc/block.rs`

- `954f4ea51031f7c08c603b53` (trace2code, `tokio-rs/tokio`):
  - query: failure_excerpt: $ cargo test -p tokio --test sync_semaphore running 15 tests test add_max_amount_permits ... ok test add_more_than_max_amount_permits2 - should panic ... ok test add_p...
  - gold: `tokio/src/sync/batch_semaphore.rs`
  - final/read: `tokio/tests/sync_semaphore.rs`, `tokio/tests/sync_semaphore_owned.rs`, `tokio/src/sync/semaphore.rs`
  - metrics: final_recall=0.0000, final_precision=0.0000, line_f1=0.0000
  - suggested_unread: `tokio/src/sync/batch_semaphore.rs`, `tokio/src/sync/tests/semaphore_batch.rs`

## Interpretation

- `code2test` failures mostly reflect repository-specific source-to-test mapping. The agent can read plausible implementation context and still miss the exact regression test file.
- `comment2context` failures split into two groups: near misses where supporting context is useful but not exact gold, and hard misses where the review comment does not lexically point to the needed policy/config/test dependency.
- `trace2code` shows the strongest file-level retrieval, but many successes still have weak line-level overlap. This is evidence that future trajectory evaluation should reward not only which file entered context but also which span entered context.
- The strict-context runner makes these errors easier to trust: every `final_files` entry was actually read, while unread recommendations are isolated in `suggested_unread_files`.

## Next Analysis Hooks

- Add a span-aware read action that records smaller line windows when the model cites a file section.
- Compare this run against a strict-context GPT-4.1-mini or stronger model to separate model weakness from benchmark difficulty.
- Add a hybrid retrieval baseline that combines RepoMap candidates with model-driven final context selection.
