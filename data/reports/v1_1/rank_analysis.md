# Rank and Error Analysis

- Generated at: `2026-05-28T07:56:37+00:00`
- Eval dir: `data/eval/v1_1`
- Samples: `data/benchmark/v1_1/samples.jsonl`
- Candidate filter: `all_files`
- Models: `7`

## Overall First-Gold CDF

| Model | Mode | Any@1 | Any@5 | Any@10 | Any@20 | Any@50 | Median Hit Rank | Misses | MRR | Gold@8k |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-Embedding-4B | `embedding` | 0.0976 | 0.3589 | 0.4948 | 0.6864 | 0.8537 | 11 | 0 | 0.2296 | 0.2516 |
| Qwen3-Embedding-8B | `embedding` | 0.0732 | 0.4077 | 0.6272 | 0.7944 | 0.8955 | 7 | 0 | 0.2272 | 0.1533 |
| pplx-embed-v1-4b | `embedding` | 0.0801 | 0.3310 | 0.5017 | 0.6794 | 0.9024 | 10 | 0 | 0.2143 | 0.1359 |
| aider-style-repomap | `repomap` | 0.0627 | 0.3728 | 0.5436 | 0.7247 | 0.8711 | 9 | 0 | 0.2125 | 0.0627 |
| jina-code-embeddings-0.5b | `embedding` | 0.0801 | 0.2683 | 0.3624 | 0.5226 | 0.7770 | 19 | 0 | 0.1813 | 0.1568 |
| nomic-embed-code | `embedding` | 0.0627 | 0.3101 | 0.4042 | 0.5784 | 0.8258 | 15 | 0 | 0.1810 | 0.0832 |
| lexical | `corpus` | 0.0174 | 0.2474 | 0.3833 | 0.5470 | 0.7700 | 17 | 0 | 0.1392 | 0.0720 |

## Task First-Gold CDF

### code2test

| Model | Mode | Any@1 | Any@5 | Any@10 | Any@20 | Any@50 | Median Hit Rank | Misses | MRR | Gold@8k |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-Embedding-4B | `embedding` | 0.1415 | 0.5472 | 0.6698 | 0.7830 | 0.8585 | 5 | 0 | 0.3225 | 0.3887 |
| pplx-embed-v1-4b | `embedding` | 0.1038 | 0.4245 | 0.5472 | 0.7264 | 0.9151 | 9 | 0 | 0.2516 | 0.1934 |
| Qwen3-Embedding-8B | `embedding` | 0.0849 | 0.4528 | 0.6509 | 0.7453 | 0.8679 | 6 | 0 | 0.2406 | 0.1651 |
| nomic-embed-code | `embedding` | 0.0660 | 0.3491 | 0.5189 | 0.7453 | 0.8585 | 10 | 0 | 0.2066 | 0.0553 |
| jina-code-embeddings-0.5b | `embedding` | 0.0943 | 0.3113 | 0.4340 | 0.6038 | 0.7642 | 13 | 0 | 0.2033 | 0.2060 |
| aider-style-repomap | `repomap` | 0.0660 | 0.3113 | 0.4528 | 0.6321 | 0.8208 | 12 | 0 | 0.1962 | 0.0786 |
| lexical | `corpus` | 0.0000 | 0.0943 | 0.1887 | 0.3113 | 0.6509 | 29 | 0 | 0.0663 | 0.0299 |

### comment2context

| Model | Mode | Any@1 | Any@5 | Any@10 | Any@20 | Any@50 | Median Hit Rank | Misses | MRR | Gold@8k |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| jina-code-embeddings-0.5b | `embedding` | 0.1625 | 0.4625 | 0.5625 | 0.7125 | 0.8250 | 7 | 0 | 0.3043 | 0.2521 |
| Qwen3-Embedding-4B | `embedding` | 0.1500 | 0.4625 | 0.6375 | 0.7500 | 0.8000 | 7 | 0 | 0.2920 | 0.3250 |
| Qwen3-Embedding-8B | `embedding` | 0.1125 | 0.5125 | 0.6625 | 0.7750 | 0.8250 | 5 | 0 | 0.2874 | 0.2354 |
| nomic-embed-code | `embedding` | 0.1250 | 0.4500 | 0.5375 | 0.6000 | 0.7625 | 8 | 0 | 0.2657 | 0.1875 |
| pplx-embed-v1-4b | `embedding` | 0.1125 | 0.4000 | 0.6500 | 0.7250 | 0.8750 | 7 | 0 | 0.2623 | 0.1625 |
| aider-style-repomap | `repomap` | 0.0500 | 0.2625 | 0.3750 | 0.6125 | 0.7875 | 15 | 0 | 0.1558 | 0.0333 |
| lexical | `corpus` | 0.0125 | 0.2750 | 0.4500 | 0.6000 | 0.7500 | 11 | 0 | 0.1495 | 0.0563 |

### trace2code

| Model | Mode | Any@1 | Any@5 | Any@10 | Any@20 | Any@50 | Median Hit Rank | Misses | MRR | Gold@8k |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| aider-style-repomap | `repomap` | 0.0693 | 0.5248 | 0.7723 | 0.9109 | 0.9901 | 5 | 0 | 0.2745 | 0.0693 |
| lexical | `corpus` | 0.0396 | 0.3861 | 0.5347 | 0.7525 | 0.9109 | 8 | 0 | 0.2075 | 0.1287 |
| Qwen3-Embedding-8B | `embedding` | 0.0297 | 0.2772 | 0.5743 | 0.8614 | 0.9802 | 9 | 0 | 0.1654 | 0.0759 |
| pplx-embed-v1-4b | `embedding` | 0.0297 | 0.1782 | 0.3366 | 0.5941 | 0.9109 | 18 | 0 | 0.1372 | 0.0545 |
| nomic-embed-code | `embedding` | 0.0099 | 0.1584 | 0.1782 | 0.3861 | 0.8416 | 25 | 0 | 0.0871 | 0.0297 |
| Qwen3-Embedding-4B | `embedding` | 0.0099 | 0.0792 | 0.1980 | 0.5347 | 0.8911 | 20 | 0 | 0.0827 | 0.0495 |
| jina-code-embeddings-0.5b | `embedding` | 0.0000 | 0.0693 | 0.1287 | 0.2871 | 0.7525 | 29 | 0 | 0.0607 | 0.0297 |


## Cross-Model Coverage

| Task | Samples | Any Model Hit@20 | All Models Miss@20 | Hit Model Count Distribution |
| --- | ---: | ---: | ---: | --- |
| overall | 287 | 265 | 22 | 0 models: 22, 1 models: 10, 2 models: 20, 3 models: 29, 4 models: 41, 5 models: 45, 6 models: 65, 7 models: 55 |
| code2test | 106 | 96 | 10 | 0 models: 10, 1 models: 5, 2 models: 5, 3 models: 11, 4 models: 11, 5 models: 15, 6 models: 28, 7 models: 21 |
| comment2context | 80 | 71 | 9 | 0 models: 9, 1 models: 4, 2 models: 4, 3 models: 3, 4 models: 7, 5 models: 10, 6 models: 18, 7 models: 25 |
| trace2code | 101 | 98 | 3 | 0 models: 3, 1 models: 1, 2 models: 11, 3 models: 15, 4 models: 23, 5 models: 20, 6 models: 19, 7 models: 9 |

## Representative Error Slices

### All Models Miss@20: code2test

| Sample | Repo | Gold | Focus Model Results | Query Excerpt |
| --- | --- | --- | --- | --- |
| 6338b75970bc7489e13952b4 | etcd-io/etcd | tests/e2e/ctl_v3_test.go<br>tests/framework/e2e/cluster.go | aider-style-repomap: rank=26, R@20=0.0000, top=server/etcdmain/config_test.go, server/embed/config.go, server/etcdserver/server.go, server/storage/mvcc/metrics.go, server/storage/mvcc/kvstore_test.go | {"changed_file_summary": "9 implementation files and 2 existing test files changed within 13 total files.", "implementation_file_count": 9, "implementation_files": ["pkg/flags/uint32.go", "server/config/config.go", "server/embed/config.go", "server/embed/etcd.go", "server/embed/s ...[truncated] |
| 210933147c231644eeebdaae | huggingface/diffusers | tests/others/test_utils.py | nomic-embed-code: rank=139, R@20=0.0000, top=src/diffusers/utils/torch_utils.py, .github/workflows/pr_tests.yml, .github/workflows/pr_tests_gpu.yml, examples/community/kohya_hires_fix.py, docs/source/en/optimization/fp16.md | {"changed_file": "src/diffusers/utils/torch_utils.py", "pr_body": "## What does this PR do?\n\nCloses #12504.\n\n`fourier_filter` (the FFT helper used by `enable_freeu`) already upcasts\n`bfloat16` inputs to `float32` before calling `torch.fft.fftn`, because\nPyTorch's FFT does n ...[truncated] |
| c8318e728513c1a4c1bd4c9d | huggingface/transformers | tests/utils/test_generic.py | pplx-embed-v1-4b: rank=31, R@20=0.0000, top=src/transformers/pytorch_utils.py, tests/utils/test_model_output.py, src/transformers/utils/output_capturing.py, src/transformers/utils/generic.py, tests/generation/test_utils.py | {"changed_file_summary": "1 implementation files and 1 existing test files changed within 2 total files.", "implementation_file_count": 1, "implementation_files": ["src/transformers/utils/generic.py"], "pr_body": "`_register_model_output_pytree_node` was calling set.__contains__ ...[truncated] |
| 94f11ff5d8076850f94d6f04 | mockito/mockito | mockito-integration-tests/programmatic-tests/src/test/java/org/mockito/ProgrammaticMockMakerTest.java | pplx-embed-v1-4b: rank=50, R@20=0.0000, top=mockito-core/src/test/java/org/mockitousage/bugs/creation/MockClassWithMissingStaticDepTest.java, mockito-core/src/main/java/org/mockito/internal/PremainAttach.java, mockito-core/src/main/java/org/mockito/internal/creation/bytebuddy/InlineDelegateByteBuddyMockMaker.java, mockito-core/src/main/java/org/mockito/internal/PremainAttachAccess.java, doc/release-notes/official.md | {"changed_file_summary": "3 implementation files and 1 existing test files changed within 8 total files.", "implementation_file_count": 3, "implementation_files": ["mockito-core/src/main/java/org/mockito/Mockito.java", "mockito-core/src/main/java/org/mockito/internal/PremainAttac ...[truncated] |
| 2177ce0889655fd1979b99c1 | spring-projects/spring-boot | module/spring-boot-webflux/src/test/java/org/springframework/boot/webflux/autoconfigure/HttpHandlerAutoConfigurationTests.java | pplx-embed-v1-4b: rank=25, R@20=0.0000, top=module/spring-boot-webflux/src/test/java/org/springframework/boot/webflux/autoconfigure/error/DefaultErrorWebExceptionHandlerIntegrationTests.java, module/spring-boot-webflux/src/main/java/org/springframework/boot/webflux/autoconfigure/WebFluxAutoConfiguration.java, module/spring-boot-webflux/src/main/java/org/springframework/boot/webflux/autoconfigure/WebFluxProperties.java, module/spring-boot-webflux/src/main/java/org/springframework/boot/webflux/autoconfigure/error/ErrorWebFluxAutoConfiguration.java, documentation/spring-boot-docs/src/docs/antora/modules/reference/pages/web/reactive.adoc | {"changed_file_summary": "2 implementation files and 1 existing test files changed within 3 total files.", "implementation_file_count": 2, "implementation_files": ["module/spring-boot-webflux/src/main/java/org/springframework/boot/webflux/autoconfigure/HttpHandlerAutoConfiguratio ...[truncated] |

### All Models Miss@20: comment2context

| Sample | Repo | Gold | Focus Model Results | Query Excerpt |
| --- | --- | --- | --- | --- |
| 413524fdc2e0151ec121653c | astral-sh/ruff | crates/ty_server/tests/e2e/configuration.rs | aider-style-repomap: rank=35, R@20=0.0000, top=crates/ty_python_core/src/db.rs, crates/ty_server/src/db.rs, crates/ty_project/src/db/changes.rs, crates/ty_python_semantic/src/db.rs, crates/ruff_graph/src/db.rs | {"diff_hunk_context": "@@ -294,11 +312,12 @@ impl ProjectDatabase {\n return result;\n } else if result.custom_stdlib_changed {\n match project.metadata(self).to_program_settings(\n self.system(),\n self.vendored(),\ ...[truncated] |
| 449280b38ce700056be47fe | eslint/eslint | eslint.config.js | pplx-embed-v1-4b: rank=109, R@20=0.0000, top=docs/src/rules/no-param-reassign.md, lib/rules/no-param-reassign.js, lib/config/default-config.js, tests/lib/rule-tester/rule-tester.js, packages/js/src/configs/eslint-recommended.js | {"diff_hunk_context": "@@ -61,6 +61,8 @@ module.exports = {\n \t\t\t},\n \t\t],\n \n+\t\tdefaultOptions: [{ props: false }],", "given_file": "lib/rules/no-param-reassign.js", "line": 64, "path": "lib/rules/no-param-reassign.js", "pr_title": "refactor: add `meta.defaultOptions` to ...[truncated] |
| 8c8c2eb1ece3b9098d999ec6 | huggingface/diffusers | tests/pipelines/qwenimage/test_qwenimage.py<br>tests/pipelines/qwenimage/test_qwenimage_controlnet.py<br>tests/pipelines/qwenimage/test_qwenimage_edit.py | aider-style-repomap: rank=135, R@20=0.0000, top=src/diffusers/pipelines/qwenimage/pipeline_output.py, src/diffusers/pipelines/qwenimage/__init__.py, src/diffusers/modular_pipelines/qwenimage/modular_blocks_qwenimage.py, src/diffusers/pipelines/qwenimage/pipeline_qwenimage_controlnet_inpaint.py, src/diffusers/modular_pipelines/qwenimage/modular_blocks_qwenimage_edit.py | {"diff_hunk_context": "@@ -353,15 +355,6 @@ def check_inputs(\n f\" {negative_prompt_embeds}. Please make sure to only forward one of the two.\"\n )\n ", "given_file": "src/diffusers/pipelines/qwenimage/pipeline_qwenimage_controlnet_inpaint.py", "line" ...[truncated] |
| a17b0c140c1465ce72d6380f | huggingface/diffusers | tests/pipelines/qwenimage/test_qwenimage.py<br>tests/pipelines/qwenimage/test_qwenimage_controlnet.py<br>tests/pipelines/qwenimage/test_qwenimage_edit.py | nomic-embed-code: rank=58, R@20=0.0000, top=src/diffusers/modular_pipelines/qwenimage/before_denoise.py, src/diffusers/modular_pipelines/qwenimage/modular_pipeline.py, src/diffusers/modular_pipelines/qwenimage/denoise.py, src/diffusers/modular_pipelines/qwenimage/inputs.py, src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit.py | {"diff_hunk_context": "@@ -584,9 +584,7 @@ def __call__(\n \n device = self._execution_device\n ", "given_file": "src/diffusers/pipelines/qwenimage/pipeline_qwenimage.py", "line": 603, "path": "src/diffusers/pipelines/qwenimage/pipeline_qwenimage.py", "pr_title": "fix(qwe ...[truncated] |
| 0a166358bf317488838726c8 | huggingface/transformers | src/transformers/conversion_mapping.py | nomic-embed-code: rank=234, R@20=0.0000, top=src/transformers/integrations/finegrained_fp8.py, src/transformers/models/diffllama/modular_diffllama.py, src/transformers/utils/kernel_config.py, src/transformers/masking_utils.py, src/transformers/integrations/deepspeed.py | {"diff_hunk_context": "@@ -600,9 +617,9 @@ def __init__(\n self.block_size = block_size\n self.hidden_dim = config.hidden_size\n self.activation_scheme = activation_scheme", "given_file": "src/transformers/integrations/finegrained_fp8.py", "line": 612, "pa ...[truncated] |

### All Models Miss@20: trace2code

| Sample | Repo | Gold | Focus Model Results | Query Excerpt |
| --- | --- | --- | --- | --- |
| b389823a6435d1ada9d7d2b9 | etcd-io/etcd | server/etcdserver/util.go<br>server/etcdserver/v3_server.go<br>server/features/etcd_features.go | pplx-embed-v1-4b: rank=25, R@20=0.0000, top=server/go.mod, tests/go.mod, go.mod, client/v3/go.mod, etcdutl/go.mod | {"command": "go test ./server/etcdserver", "failure_excerpt": "$ go test ./server/etcdserver\nFAIL\tgo.etcd.io/etcd/server/v3/etcdserver [build failed]\nFAIL\n\ngo: downloading github.com/gogo/protobuf v1.3.2\ngo: downloading go.etcd.io/raft/v3 v3.6.0-beta.0.0.20260116184858-6d94 ...[truncated] |
| aac6e0174bcf8f6defe4c774 | gin-gonic/gin | binding/form_mapping.go | aider-style-repomap: rank=32, R@20=0.0000, top=gin_test.go, context_test.go, binding/binding_test.go, gin.go, render/render_test.go | {"command": "go test ./.", "failure_excerpt": "$ go test ./.\n2026/05/06 22:07:07 The AppEngine flag is going to be deprecated. Please check issues #2723 and #2739 and use 'TrustedPlatform: gin.PlatformGoogleAppEngine' instead.\n[GIN] 2026/05/06 - 22:07:07 \| 200 \| 3.167µs \| ...[truncated] |
| 1132f037ebb462f13652adff | tokio-rs/tokio | tokio/src/sync/mpsc/block.rs | Qwen3-Embedding-4B: rank=22, R@20=0.0000, top=tokio/tests/sync_mpsc.rs, tokio/src/sync/tests/loom_mpsc.rs, tokio-util/tests/mpsc.rs, tokio/tests/sync_errors.rs, tokio/tests/rt_common.rs | {"command": "cargo test --features full,test-util -p tokio --test sync_mpsc", "failure_excerpt": "$ cargo test --features full,test-util -p tokio --test sync_mpsc\n...\n... ok\ntest test_rx_unbounded_is_closed_when_there_are_no_senders_and_there_are_messages ... ok\ntest try_recv ...[truncated] |

### Trace2Code RepoMap Hits, Embeddings Miss@20

| Sample | Repo | Gold | Focus Model Results | Query Excerpt |
| --- | --- | --- | --- | --- |
| 30fb9ae30b972db986c4365d | gin-gonic/gin | response_writer.go | aider-style-repomap: rank=6, R@20=1.0000, top=response_writer_test.go, gin_test.go, render/render_test.go, context_test.go, binding/binding_test.go<br>Qwen3-Embedding-4B: rank=48, R@20=0.0000, top=gin_integration_test.go, .github/ISSUE_TEMPLATE.md, README.md, gin_test.go, doc.go<br>Qwen3-Embedding-8B: rank=27, R@20=0.0000, top=CHANGELOG.md, .github/ISSUE_TEMPLATE.md, go.mod, BENCHMARKS.md, .github/workflows/gin.yml | {"command": "go test ./.", "failure_excerpt": "$ go test ./.\nFAIL\tgithub.com/gin-gonic/gin [build failed]\nFAIL\n\ngo: downloading golang.org/x/net v0.41.0\ngo: downloading golang.org/x/crypto v0.39.0\ngo: downloading golang.org/x/text v0.26.0\n# github.com/gin-gonic/gin [githu ...[truncated] |
| 9e63d7a07d201c84d7d18546 | gin-gonic/gin | errors.go | aider-style-repomap: rank=9, R@20=1.0000, top=gin_test.go, errors_test.go, gin.go, context_test.go, go.mod<br>Qwen3-Embedding-4B: rank=37, R@20=0.0000, top=errors_test.go, gin_integration_test.go, doc.go, .github/ISSUE_TEMPLATE.md, README.md<br>Qwen3-Embedding-8B: rank=42, R@20=0.0000, top=CHANGELOG.md, errors_test.go, BENCHMARKS.md, gin.go, .github/ISSUE_TEMPLATE.md | {"command": "go test ./.", "failure_excerpt": "$ go test ./.\n2026/05/06 22:26:07 The AppEngine flag is going to be deprecated. Please check issues #2723 and #2739 and use 'TrustedPlatform: gin.PlatformGoogleAppEngine' instead.\n[GIN] 2026/05/06 - 22:26:07 \| 200 \| 2.042µs \| ...[truncated] |
| a9f6b8adbd0f71a58c99835e | gin-gonic/gin | binding/binding.go<br>binding/binding_nomsgpack.go<br>render/yaml.go | aider-style-repomap: rank=3, R@20=0.3333, top=context_test.go, render/render_test.go, render/yaml.go, context.go, binding/binding_test.go<br>Qwen3-Embedding-4B: rank=21, R@20=0.0000, top=render/render_test.go, context_test.go, CHANGELOG.md, gin_integration_test.go, .github/ISSUE_TEMPLATE.md<br>Qwen3-Embedding-8B: rank=22, R@20=0.0000, top=CHANGELOG.md, render/render_test.go, context_test.go, BENCHMARKS.md, docs/doc.md | {"command": "go test ./.", "failure_excerpt": "$ go test ./.\n--- FAIL: TestContextRenderYAML (0.00s)\n context_test.go:1072: \n \tError Trace:\t/Users/eyuansu62/llm_judge/a-good-project/data/repro_worktrees/gin-gonic__gin/f0fbdd683d906d138b55fc2b/context_test.go:1072\n ...[truncated] |

### Trace2Code Embedding Hits, RepoMap Miss@20

| Sample | Repo | Gold | Focus Model Results | Query Excerpt |
| --- | --- | --- | --- | --- |
| d4149b3f568dfe275250b62e | caddyserver/caddy | modules/caddyhttp/caddyauth/caddyauth.go | pplx-embed-v1-4b: rank=1, R@20=1.0000, top=modules/caddyhttp/caddyauth/caddyauth.go, go.mod, modules/caddytls/connpolicy_test.go, AGENTS.md, caddytest/caddytest.go<br>aider-style-repomap: rank=81, R@20=0.0000, top=go.mod, modules/caddyhttp/server.go, caddy.go, cmd/main.go, caddyconfig/httpcaddyfile/directives.go | {"command": "go test ./modules/caddyhttp/caddyauth", "failure_excerpt": "$ go test ./modules/caddyhttp/caddyauth\n--- FAIL: TestAuthenticationSetsUserPlaceholdersOnUnauthorized (0.00s)\n caddyauth_test.go:67: expected http.auth.user.id to be alice, got \"\" (ok=false)\nFAIL\nF ...[truncated] |
| de059c1c2e3ffc63820ce0e4 | caddyserver/caddy | caddyconfig/httpcaddyfile/builtins.go<br>caddyconfig/httpcaddyfile/tlsapp.go | Qwen3-Embedding-4B: rank=4, R@20=0.5000, top=go.mod, README.md, caddytest/a.caddy.localhost.crt, caddyconfig/httpcaddyfile/builtins.go, caddytest/integration/caddyfile_test.go<br>aider-style-repomap: rank=21, R@20=0.0000, top=go.mod, caddyconfig/httpcaddyfile/options_test.go, caddyconfig/httpcaddyfile/testdata/import_variadic.txt, caddyconfig/httpcaddyfile/tlsapp_test.go, caddyconfig/httpcaddyfile/testdata/import_variadic_with_import.txt | {"command": "go test ./caddyconfig/httpcaddyfile", "failure_excerpt": "$ go test ./caddyconfig/httpcaddyfile\nFAIL\tgithub.com/caddyserver/caddy/v2/caddyconfig/httpcaddyfile [build failed]\nFAIL\n\ngo: downloading github.com/google/cel-go v0.27.0\ngo: downloading github.com/DeRui ...[truncated] |
| 0f4d4723a06149a5172fed0a | gin-gonic/gin | context.go | pplx-embed-v1-4b: rank=10, R@20=1.0000, top=go.mod, debug_test.go, context_test.go, doc.go, ginS/README.md<br>aider-style-repomap: rank=29, R@20=0.0000, top=context_test.go, context_file_test.go, fs_test.go, middleware_test.go, binding/binding_test.go | {"command": "go test ./.", "failure_excerpt": "$ go test ./.\nFAIL\tgithub.com/gin-gonic/gin [build failed]\nFAIL\n\n# github.com/gin-gonic/gin [github.com/gin-gonic/gin.test]\n./context_test.go:1644:21: undefined: MIMEPROTOBUF", "run_strategy": "go_test_package", "source_type": ...[truncated] |
| 0f88458fc1fe4acce078ce3f | gin-gonic/gin | debug.go | pplx-embed-v1-4b: rank=3, R@20=1.0000, top=debug_test.go, .github/workflows/gin.yml, debug.go, go.mod, mode_test.go<br>aider-style-repomap: rank=23, R@20=0.0000, top=debug_test.go, gin_test.go, render/render_test.go, middleware_test.go, context_file_test.go | {"command": "go test ./.", "failure_excerpt": "$ go test ./.\nFAIL\tgithub.com/gin-gonic/gin [build failed]\nFAIL\n\n# github.com/gin-gonic/gin [github.com/gin-gonic/gin.test]\n./debug_test.go:118:2: undefined: runtimeVersion", "run_strategy": "go_test_package", "source_type": "l ...[truncated] |
| 1e7cac8cfdead734e7e51ad6 | gin-gonic/gin | recovery.go | jina-code-embeddings-0.5b: rank=10, R@20=1.0000, top=recovery_test.go, context_test.go, gin_test.go, gin_integration_test.go, render/render_test.go<br>aider-style-repomap: rank=32, R@20=0.0000, top=recovery_test.go, context_test.go, context_file_test.go, fs_test.go, render/render_test.go | {"command": "go test ./.", "failure_excerpt": "$ go test ./.\nFAIL\tgithub.com/gin-gonic/gin [build failed]\nFAIL\n\n# github.com/gin-gonic/gin [github.com/gin-gonic/gin.test]\n./recovery_test.go:300:14: undefined: secureRequestDump", "run_strategy": "go_test_package", "source_ty ...[truncated] |

### Comment2Context Given-File Top1 Traps

| Sample | Repo | Gold | Focus Model Results | Query Excerpt |
| --- | --- | --- | --- | --- |
| bf47308900fafdff5ca96ecd | HypothesisWorks/hypothesis | hypothesis-python/src/hypothesis/core.py<br>hypothesis-python/src/hypothesis/internal/conjecture/engine.py | aider-style-repomap: rank=28, R@20=0.0000, top=hypothesis-python/tests/cover/test_seed_printing.py, hypothesis-python/tests/cover/test_stateful.py, hypothesis-python/tests/cover/test_deadline.py, hypothesis-python/tests/cover/test_reporting.py, hypothesis-python/tests/cover/test_testdecorators.py | {"diff_hunk_context": "@@ -108,3 +116,30 @@ def test(i):\n test()\n \n assert \"@seed\" in o.getvalue()", "given_file": "hypothesis-python/tests/cover/test_seed_printing.py", "line": 145, "path": "hypothesis-python/tests/cover/test_seed_printing.py", "pr_title": "Prin ...[truncated] |
| 413524fdc2e0151ec121653c | astral-sh/ruff | crates/ty_server/tests/e2e/configuration.rs | lexical: rank=98, R@20=0.0000, top=crates/ty_project/src/db/changes.rs, crates/ty_wasm/src/lib.rs, crates/ty_project/src/db.rs, crates/ty_project/src/metadata/options.rs, crates/ty/tests/file_watching.rs | {"diff_hunk_context": "@@ -294,11 +312,12 @@ impl ProjectDatabase {\n return result;\n } else if result.custom_stdlib_changed {\n match project.metadata(self).to_program_settings(\n self.system(),\n self.vendored(),\ ...[truncated] |
| 538d376414b12c2b7cc08d73 | astral-sh/ruff | crates/ty_server/tests/e2e/configuration.rs<br>crates/ruff_db/src/diagnostic/mod.rs | Qwen3-Embedding-8B: rank=116, R@20=0.0000, top=crates/ty_project/src/db.rs, crates/ty_python_core/src/program.rs, crates/ty_wasm/src/lib.rs, crates/ty_project/src/lib.rs, crates/ty_project/src/db/changes.rs | {"diff_hunk_context": "@@ -106,17 +106,25 @@ impl ProjectDatabase {\n // we may want to have a dedicated method for this?\n \n // Initialize the `Program` singleton", "given_file": "crates/ty_project/src/db.rs", "line": 110, "path": "crates/ty_project/src/db.rs" ...[truncated] |
| 8e6bf4a9d26a7468cab5a5da | astral-sh/ruff | crates/ty_python_semantic/src/types/enums.rs | lexical: rank=124, R@20=0.0000, top=crates/ty_python_semantic/resources/mdtest/enums.md, crates/ty_python_semantic/resources/mdtest/annotations/new_types.md, crates/ty_python_semantic/resources/mdtest/type_of/dynamic.md, crates/ty_python_semantic/resources/mdtest/comparison/tuples.md, crates/ty_python_semantic/resources/mdtest/unpacking.md | {"diff_hunk_context": "@@ -441,6 +441,48 @@ reveal_type(Planet2.MERCURY.value) # revealed: Any\n reveal_type(Planet2.MERCURY._value_) # revealed: Any\n ```\n ", "given_file": "crates/ty_python_semantic/resources/mdtest/enums.md", "line": 444, "path": "crates/ty_python_semantic/ ...[truncated] |
| f9c42afe1a93c3f2286bd0aa | astral-sh/ruff | crates/ty_server/tests/e2e/configuration.rs<br>crates/ruff_db/src/diagnostic/mod.rs | Qwen3-Embedding-4B: rank=28, R@20=0.0000, top=crates/ty_project/src/db.rs, crates/ty_project/src/db/changes.rs, crates/ty_project/src/metadata/options.rs, crates/ty_wasm/src/lib.rs, crates/ty_project/src/lib.rs | {"diff_hunk_context": "@@ -106,17 +106,25 @@ impl ProjectDatabase {\n // we may want to have a dedicated method for this?\n \n // Initialize the `Program` singleton\n Program::from_settings(&db, program_settings);\n \n \|error\| anyhow::anyhow! ...[truncated] |
