# Data and Corpus Licensing Notes

Agent Retrieval Bench has two different licensing surfaces:

1. **Evaluator code, benchmark metadata, reports, and documentation** are released under the MIT License in this repository.
2. **Redistributed repository corpus files** are snapshots or derived chunks from upstream open-source repositories at recorded base commits. Those files remain governed by the licenses of their original upstream projects.

The Hugging Face release bundle includes benchmark JSONL files, evaluation reports, and prebuilt corpus chunks for reproducibility. The corpus chunks are provided only to make the benchmark easy to run; they do not change or replace upstream licenses.

## Upstream repositories represented in V1

V1 corpus chunks are derived from commits in these repositories:

- `caddyserver/caddy`
- `clap-rs/clap`
- `etcd-io/etcd`
- `fastapi/fastapi`
- `gin-gonic/gin`
- `huggingface/diffusers`
- `huggingface/transformers`
- `mockito/mockito`
- `pytest-dev/pytest`
- `spring-projects/spring-boot`
- `tokio-rs/tokio`
- `vitejs/vite`
- `vuejs/core`

Users who redistribute, modify, or use the corpus chunks beyond benchmark evaluation should review and comply with the corresponding upstream project licenses.

## Citation and attribution

If you use the benchmark, cite Agent Retrieval Bench and retain attribution to the upstream repositories represented in the corpus.
