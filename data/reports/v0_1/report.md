# Benchmark V0.1 Diagnostic Report

Generated at: `2026-04-27T04:53:09+00:00`

## Executive Conclusions
- V0.1 can be used as a closed-loop smoke benchmark, but not yet as a final model-ranking benchmark.
- Keep code2test as the main hard slice for V0.2; lexical Recall@20 is low enough to expose retrieval weakness.
- Keep comment2context, but downweight or separately report direct-hint samples because lexical retrieval is near ceiling.
- Treat trace2code as smoke-only until it has enough validated samples for stable task-level metrics.
- Use the failure table to inspect code2test weak labels before scaling V0.2.
- Keep testlog2code excluded until the audited valid rate reaches the planned threshold.

## Dataset Distribution

| Task | Samples |
| --- | ---: |
| `code2test` | 17 |
| `comment2context` | 16 |
| `trace2code` | 2 |

| Repo | Samples |
| --- | ---: |
| `clap-rs/clap` | 4 |
| `etcd-io/etcd` | 3 |
| `fastapi/fastapi` | 3 |
| `gin-gonic/gin` | 3 |
| `huggingface/diffusers` | 5 |
| `huggingface/transformers` | 3 |
| `mockito/mockito` | 4 |
| `pytest-dev/pytest` | 2 |
| `spring-projects/spring-boot` | 3 |
| `tokio-rs/tokio` | 2 |
| `vitejs/vite` | 2 |
| `vuejs/core` | 1 |

## Baseline Metrics

| Task | Samples | Recall@5 | Recall@10 | Recall@20 | MRR | gold_coverage@8k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `code2test` | 17 | 0.2353 | 0.2941 | 0.3676 | 0.1418 | 0.1176 |
| `comment2context` | 16 | 1.0000 | 1.0000 | 1.0000 | 0.9688 | 1.0000 |
| `overall` | 35 | 0.6286 | 0.6571 | 0.6929 | 0.5546 | 0.5714 |
| `trace2code` | 2 | 1.0000 | 1.0000 | 1.0000 | 0.7500 | 1.0000 |

## Corpus And Query Hint Checks

- Gold fully present in corpus: `31/35`.
- Samples with missing gold files: `4`.
- Samples with direct gold path hint in query: `19`.
- Samples with direct gold basename hint in query: `0`.

## Hard/Easy Buckets

| Bucket | Samples |
| --- | ---: |
| `easy_lexical` | 2 |
| `hard_lexical_miss` | 7 |
| `invalid_missing_gold` | 4 |
| `medium_lexical` | 2 |
| `partial_lexical` | 1 |
| `too_easy_direct_hint` | 19 |

## Sample Recommendations

| Recommendation | Samples |
| --- | ---: |
| `downweight_direct_hint` | 17 |
| `drop_missing_gold` | 4 |
| `keep` | 4 |
| `keep_hard_code2test` | 8 |
| `smoke_only` | 2 |

| Task | Gold path hints | Gold basename hints |
| --- | ---: | ---: |
| `code2test` | 1 | 0 |
| `comment2context` | 16 | 0 |
| `trace2code` | 2 | 0 |

- Drop candidates due to missing gold in corpus: `85ee292597f83f279a729958`, `9c61dfc0e244c9c32a56732a`, `6fe2bb7411520bb6314b0c5b`, `8fbb91cb2863b0c380f029a2`.

- Downweight candidates due to direct query hints: `f8df5943a927293c464bb24f`, `80f523585ea462f220bb3c2f`, `9848546c7c3e269971e40803`, `198cc5d51ea221dd713772f2`, `2e5e97b1057bd10b662d93b1`, `b5bc7d2215ff825d9965c1d4`, `5df695f154a448e2bff792c7`, `eca7fbe00fa92d50048e50df`, `33535ecd016f08e8792cb808`, `f85fb833471feeba260c0508`, `09f9dc342e5eb5622c342711`, `0a283cf6638a4587c5a6ebe6`, `cef62c2bbecf0069df3e775c`, `33624b6818d771c07d44ea12`, `c8c2b30d1862688717b12fce`, `7de0316f76bcb43bff686bfa`, `1511b1ae71dc7a16959309b7`.

## Failure Samples

| Sample | Task | Repo | Recall@20 | MRR | Bucket | Recommendation | Gold | Top 5 |
| --- | --- | --- | ---: | ---: | --- | --- | --- | --- |
| `dd57d697481176a1c52bd8d0` | `code2test` | `fastapi/fastapi` | 0.0000 | 0.0009 | `hard_lexical_miss` | `keep_hard_code2test` | `tests/test_jsonable_encoder.py` | `fastapi/_compat/v2.py`<br>`docs/en/docs/tutorial/schema-extra-example.md`<br>`docs/en/docs/how-to/migrate-from-pydantic-v1-to-pydantic-v2.md`<br>`fastapi/routing.py`<br>`docs/en/docs/how-to/separate-openapi-schemas.md` |
| `fb1fc240d8def44d045f2426` | `code2test` | `huggingface/diffusers` | 0.0000 | 0.0013 | `hard_lexical_miss` | `keep_hard_code2test` | `tests/pipelines/qwenimage/test_qwenimage.py`<br>`tests/pipelines/qwenimage/test_qwenimage_controlnet.py`<br>`tests/pipelines/qwenimage/test_qwenimage_edit.py`<br>`tests/pipelines/qwenimage/test_qwenimage_edit_plus.py`<br>`tests/pipelines/qwenimage/test_qwenimage_img2img.py`<br>`tests/pipelines/qwenimage/test_qwenimage_inpaint.py` | `.github/PULL_REQUEST_TEMPLATE.md`<br>`src/diffusers/pipelines/qwenimage/pipeline_qwenimage_controlnet_inpaint.py`<br>`.github/ISSUE_TEMPLATE/translate.md`<br>`docs/README.md`<br>`src/diffusers/pipelines/pipeline_loading_utils.py` |
| `210bcbf1622401f32d9f8c8e` | `code2test` | `huggingface/transformers` | 0.0000 | 0.0037 | `hard_lexical_miss` | `keep_hard_code2test` | `tests/models/d_fine/test_modeling_d_fine.py` | `src/transformers/loss/loss_d_fine.py`<br>`src/transformers/loss/loss_rt_detr.py`<br>`src/transformers/models/d_fine/modeling_d_fine.py`<br>`src/transformers/models/rt_detr/modeling_rt_detr.py`<br>`src/transformers/models/rt_detr/modular_rt_detr.py` |
| `6fe2bb7411520bb6314b0c5b` | `code2test` | `mockito/mockito` | 0.0000 | 0.0082 | `invalid_missing_gold` | `drop_missing_gold` | `mockito-core/src/test/java/org/mockito/internal/PremainAttachTest.java`<br>`mockito-core/src/test/java/org/mockito/internal/creation/bytebuddy/ClinitSuppressionTransformerTest.java`<br>`mockito-integration-tests/inline-mocks-tests/src/test/java/org/mockitoinline/PremainClinitSuppressionTest.java`<br>`mockito-integration-tests/programmatic-tests/src/test/java/org/mockito/ProgrammaticMockMakerTest.java` | `mockito-core/src/main/java/org/mockito/internal/PremainAttach.java`<br>`mockito-integration-tests/java-21-tests/build.gradle.kts`<br>`mockito-core/src/main/java/org/mockito/internal/creation/bytebuddy/InlineDelegateByteBuddyMockMaker.java`<br>`mockito-core/src/main/java/org/mockito/internal/PremainAttachAccess.java`<br>`mockito-core/src/test/java/org/mockitousage/misuse/InvalidUsageTest.java` |
| `8fbb91cb2863b0c380f029a2` | `code2test` | `mockito/mockito` | 0.0000 | 0.0116 | `invalid_missing_gold` | `drop_missing_gold` | `mockito-core/src/test/java/org/mockito/internal/PremainAttachTest.java`<br>`mockito-core/src/test/java/org/mockito/internal/creation/bytebuddy/ClinitSuppressionTransformerTest.java`<br>`mockito-integration-tests/inline-mocks-tests/src/test/java/org/mockitoinline/PremainClinitSuppressionTest.java`<br>`mockito-integration-tests/programmatic-tests/src/test/java/org/mockito/ProgrammaticMockMakerTest.java` | `mockito-integration-tests/java-21-tests/build.gradle.kts`<br>`mockito-core/src/main/java/org/mockito/internal/creation/bytebuddy/InlineDelegateByteBuddyMockMaker.java`<br>`mockito-core/src/main/java/org/mockito/internal/PremainAttach.java`<br>`mockito-core/src/test/java/org/mockitousage/misuse/InvalidUsageTest.java`<br>`mockito-core/src/main/java/org/mockito/internal/PremainAttachAccess.java` |
| `13e634ffbf14f5502ea30a90` | `code2test` | `gin-gonic/gin` | 0.0000 | 0.0123 | `hard_lexical_miss` | `keep_hard_code2test` | `binding/form_mapping_test.go` | `.github/PULL_REQUEST_TEMPLATE.md`<br>`CONTRIBUTING.md`<br>`binding/form_mapping.go`<br>`.github/workflows/codeql.yml`<br>`docs/doc.md` |
| `1dd2bdfdcf7092e1c69f50d6` | `code2test` | `etcd-io/etcd` | 0.0000 | 0.0244 | `hard_lexical_miss` | `keep_hard_code2test` | `server/etcdserver/txn/range_bench_test.go`<br>`server/etcdserver/txn/txn_test.go`<br>`server/storage/mvcc/index_test.go`<br>`server/storage/mvcc/kv_test.go`<br>`server/storage/mvcc/kvstore_test.go` | `server/etcdserver/v3_server.go`<br>`server/storage/mvcc/kvstore_txn.go`<br>`tools/benchmark/cmd/range.go`<br>`client/v3/op.go`<br>`server/storage/mvcc/index.go` |
| `a395d9eb9a2c1e5a6a079c41` | `code2test` | `tokio-rs/tokio` | 0.0000 | 0.0312 | `hard_lexical_miss` | `keep_hard_code2test` | `tokio/tests/rt_common.rs` | `tokio/src/runtime/mod.rs`<br>`tokio/src/runtime/thread_pool/worker.rs`<br>`tokio/src/runtime/park.rs`<br>`tokio/src/task/mod.rs`<br>`tokio/src/io/mod.rs` |
| `18842b45a0c6a9dd08e87de4` | `code2test` | `tokio-rs/tokio` | 0.0000 | 0.0323 | `hard_lexical_miss` | `keep_hard_code2test` | `tokio/tests/rt_common.rs` | `tokio/src/runtime/builder.rs`<br>`tokio/src/runtime/thread_pool/worker.rs`<br>`tokio/src/runtime/park.rs`<br>`tokio/src/runtime/mod.rs`<br>`tokio/tests/io_driver.rs` |
| `9c61dfc0e244c9c32a56732a` | `code2test` | `huggingface/diffusers` | 0.2500 | 0.0714 | `invalid_missing_gold` | `drop_missing_gold` | `tests/models/testing_utils/__init__.py`<br>`tests/models/testing_utils/parallelism.py`<br>`tests/models/testing_utils/utils.py`<br>`tests/models/transformers/test_models_transformer_flux.py` | `src/diffusers/models/attention_dispatch.py`<br>`src/diffusers/models/transformers/transformer_prx.py`<br>`docs/source/en/optimization/attention_backends.md`<br>`examples/research_projects/anytext/anytext.py`<br>`src/diffusers/models/transformers/cogvideox_transformer_3d.py` |
| `85ee292597f83f279a729958` | `code2test` | `etcd-io/etcd` | 0.5000 | 0.5000 | `invalid_missing_gold` | `drop_missing_gold` | `server/proxy/grpcproxy/adapter/chan_stream_test.go`<br>`tests/integration/clientv3/maintenance_test.go` | `server/proxy/grpcproxy/adapter/chan_stream.go`<br>`tests/integration/clientv3/maintenance_test.go`<br>`server/proxy/grpcproxy/maintenance.go`<br>`code-of-conduct.md`<br>`.github/PULL_REQUEST_TEMPLATE.md` |
| `e8ef6b1c59b7afde52d0cded` | `code2test` | `spring-projects/spring-boot` | 0.5000 | 0.5000 | `partial_lexical` | `keep_hard_code2test` | `spring-boot-project/spring-boot-autoconfigure/src/test/java/org/springframework/boot/autoconfigure/web/reactive/error/DefaultErrorWebExceptionHandlerIntegrationTests.java`<br>`spring-boot-project/spring-boot-autoconfigure/src/test/java/org/springframework/boot/autoconfigure/web/servlet/error/ErrorMvcAutoConfigurationTests.java` | `spring-boot-project/spring-boot-autoconfigure/src/main/java/org/springframework/boot/autoconfigure/web/reactive/error/AbstractErrorWebExceptionHandler.java`<br>`spring-boot-project/spring-boot-autoconfigure/src/test/java/org/springframework/boot/autoconfigure/web/reactive/error/DefaultErrorWebExceptionHandlerIntegrationTests.java`<br>`spring-boot-project/spring-boot/src/main/java/org/springframework/boot/web/reactive/error/ErrorWebExceptionHandler.java`<br>`spring-boot-project/spring-boot-autoconfigure/src/main/java/org/springframework/boot/autoconfigure/web/servlet/error/ErrorMvcAutoConfiguration.java`<br>`spring-boot-project/spring-boot-autoconfigure/src/main/java/org/springframework/boot/autoconfigure/web/ErrorProperties.java` |

## V0.2 Decisions

- Prioritize expanding and auditing `code2test`; it is the only V0.1 slice with enough hard lexical misses.
- Keep `comment2context`, but report direct-hint and no-hint subsets separately to avoid ceiling-effect metrics.
- Keep `trace2code` as a smoke slice until the validated count is large enough for stable metrics.
- Do not add `testlog2code` to V0.2 until the cleaned audit valid rate reaches at least 50%.

## Output Files

- `diagnostic_summary.json` contains aggregate counts, metrics, buckets, and conclusions.
- `sample_diagnostics.jsonl` contains one row per evaluated sample with hints, corpus coverage, bucket, and recommendation.
