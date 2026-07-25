# V1.1 Related Work Draft

This document collects citation-backed related work for the V1.1 paper draft. It is written as paper-facing prose first, with a companion BibTeX draft for later LaTeX conversion.

BibTeX draft: `docs/paper_references_v1_1.bib`.

## Positioning Summary

Agent Retrieval Bench is closest to code search, repository-level code completion, bug localization, and end-to-end coding-agent benchmarks, but it does not fit cleanly into any one of them. ARB evaluates file-level repository context retrieval from workflow signals. The goal is not to retrieve semantically similar snippets, complete a next line of code, localize a bug from a natural-language report, or solve an issue end to end. The goal is to isolate whether a coding agent can find the files it needs before patch generation begins.

This distinction matters because ARB's three tasks use different query and gold semantics:

- `code2test`: implementation or PR intent must retrieve related tests.
- `comment2context`: a review comment and given file must retrieve additional context files.
- `trace2code`: reproduced failure output must retrieve root-cause source files, not merely visible tests or stack frames.

## Code Search and Code Intelligence Benchmarks

CodeSearchNet evaluates semantic code search over function-level natural-language/code pairs and is one of the standard reference points for neural code retrieval. CodeXGLUE broadens the scope to code understanding and generation tasks, including clone detection, code search, code summarization, and code generation. These benchmarks are important because they establish retrieval and representation learning tasks for code, but they are not designed around coding-agent context construction.

ARB differs in three ways. First, ARB evaluates file-level retrieval because coding agents consume repository files as context, even when scoring uses chunks internally. Second, ARB queries come from workflow signals rather than natural-language descriptions paired directly with code snippets. Third, ARB gold files can be indirect: tests for implementation changes, additional context beyond a reviewed file, or root-cause source files for failures.

Paper-ready wording:

> Code search benchmarks such as CodeSearchNet and CodeXGLUE evaluate important semantic matching and code-intelligence capabilities, but ARB targets a different retrieval unit and query distribution. ARB asks whether a retriever can find repository files needed by a coding agent from PR intent, review comments, and failure traces, where the relevant file may not be semantically similar to the query text.

## Repository-Level Code Completion

RepoBench evaluates repository-level code auto-completion and emphasizes cross-file context retrieval for code completion. This is closer to ARB than single-function code search because it recognizes that repository context matters. However, the task target is still completion-oriented: retrieve or use context to predict code at a location.

ARB instead evaluates workflow context retrieval before editing. The query may be a review comment, failure trace, or PR summary, and the target is a file an agent should read. This makes ARB complementary to repository-level completion benchmarks: RepoBench asks whether models can use repository context to complete code, while ARB asks whether agents can retrieve the right repository context in the first place.

Paper-ready wording:

> Repository-level completion benchmarks such as RepoBench show that cross-file context is necessary for code generation, but they still evaluate completion around a code location. ARB moves the retrieval problem upstream: the agent may not know which file location matters and must infer it from workflow evidence.

## End-to-End Coding-Agent Benchmarks

SWE-bench evaluates whether systems can resolve real GitHub issues by producing patches. SWE-agent studies agent-computer interfaces for solving software engineering tasks. These benchmarks measure end-to-end agent performance and are therefore highly relevant to coding agents. Their strength is realism: the system must retrieve, reason, edit, and validate.

ARB isolates a narrower stage. It does not ask whether an agent can generate the patch. Instead, it measures whether a retriever finds the files the agent needs to read. This makes failures easier to interpret. If an end-to-end agent fails, retrieval, reasoning, editing, or test execution may be responsible. In ARB, the upstream retrieval result is measured directly.

Paper-ready wording:

> End-to-end benchmarks such as SWE-bench are essential for measuring full agent performance, but their final patch metric hides intermediate failures. ARB isolates the context-retrieval stage, making it possible to ask whether an agent had the right repository evidence before reasoning and patch generation.

## Bug Localization and Fault Localization

Bug localization and fault localization are adjacent to `trace2code`. Defects4J provides a widely used benchmark of real Java bugs. IR-based bug localization work studies retrieving buggy source files from bug reports. These lines of work motivate the idea that failure evidence can be mapped to source locations.

ARB's `trace2code` differs in scope and query construction. The query comes from reproduced failure output, including commands, traces, compile errors, assertions, panics, and test failures. The gold is the root-cause source file an agent needs to inspect. Tests can be visible in the trace but are auxiliary unless manually audited as root cause. This is why `trace2code` is not simply trace-frame lookup.

Paper-ready wording:

> `trace2code` is related to bug and fault localization, but ARB frames the task as agent context retrieval rather than bug ranking alone. The query is a reproduced failure signal, and the output is the set of source files an agent should read before editing. This allows tests and stack frames to serve as evidence without making them the target.

## RAG, Repository Maps, and Agent Context Construction

Retrieval-augmented generation introduced the general pattern of retrieving external evidence before generation. Coding agents instantiate this idea over repositories: the retrieved evidence is source files, tests, configurations, and documentation. Aider's repository map is a practical example of non-vector context construction for coding agents, using repository structure and symbols to summarize relevant files.

ARB's results support this direction. RepoMap is strongest on `trace2code`, while embeddings lead aggregate MRR and perform well on `code2test` and `comment2context`. This suggests that agent context construction should not rely on a single retrieval family. Hybrid systems should combine semantic vectors, lexical/path matching, repository graphs, source-test relations, and task-aware reranking.

Paper-ready wording:

> ARB provides an evaluation target for repository-level RAG in coding agents. The results show that semantic retrieval and structure-aware retrieval are complementary: embeddings perform strongly on broad intent and review-context tasks, while RepoMap-style structure is especially valuable for failure-trace root-cause retrieval.

## Embedding Retrieval Evaluation

Embedding models are central baselines in ARB, but V1.1 shows why an embedding leaderboard alone is incomplete. Qwen3-Embedding-4B has the highest overall MRR, while Qwen3-Embedding-8B has broader first-gold top-20 coverage. Jina leads `comment2context` by MRR. None of these embedding models leads `trace2code`; RepoMap does.

This result should be positioned carefully. ARB does not show that embeddings are weak. It shows that semantic embeddings alone do not capture all agentic retrieval signals. In particular, failure traces often require path, symbol, and source-test structure that vectorless methods can exploit.

Paper-ready wording:

> Embedding models remain strong baselines, but ARB makes their limitations visible. A single embedding score can rank semantically related files, but agentic retrieval also needs repository topology and task semantics, especially when the query is an indirect failure signal.

## Related Work Section for Paper Draft

The final paper can use the following compact section.

### Related Work

**Code search and code intelligence.** CodeSearchNet [@husain2019codesearchnet] and CodeXGLUE [@lu2021codexglue] established influential benchmarks for semantic code search and broader code-intelligence tasks. These benchmarks evaluate representation learning and matching between natural language and code, often at the function or snippet level. ARB differs in both retrieval unit and query distribution: it evaluates file-level repository context retrieval from coding workflow signals, where the relevant file may be a test, a cross-module context file, or a root-cause source file that is not semantically close to the query.

**Repository-level code context.** RepoBench evaluates repository-level code auto-completion and highlights the importance of cross-file context [@liu2023repobench]. ARB is complementary: instead of completing code at a known location, it asks which repository files an agent should inspect before editing. This shifts the problem from using context to finding context.

**End-to-end coding-agent evaluation.** SWE-bench [@jimenez2023swebench] and SWE-agent [@yang2024sweagent] evaluate realistic software engineering agents on full issue-resolution workflows. ARB targets a narrower but important intermediate stage. End-to-end patch success can fail because of retrieval, reasoning, editing, or validation. ARB isolates retrieval, making it possible to diagnose whether the agent found the necessary files before patch generation.

**Bug and fault localization.** Bug-localization benchmarks and datasets such as Defects4J study mapping bug reports or failures to source locations [@just2014defects4j], and IR-based bug localization work studies source-file ranking from textual bug evidence [@akbar2020irbuglocalization]. ARB's `trace2code` task is adjacent, but it is framed around agent context retrieval from reproduced failure output. The gold files are root-cause source files needed for editing; visible tests and stack frames are evidence rather than automatically counted targets.

**RAG and repository maps.** Retrieval-augmented generation motivates retrieving evidence before generation [@lewis2020rag], and coding agents apply this idea to repository context. Practical systems such as Aider's repo map use structure and symbols rather than only vector similarity [@aiderrepomap]. ARB provides an evaluation target for this setting and shows that semantic embeddings and structure-aware retrieval are complementary.

## Citation Candidates

- CodeSearchNet: Hamel Husain et al., "CodeSearchNet Challenge: Evaluating the State of Semantic Code Search." arXiv:1909.09436. https://arxiv.org/abs/1909.09436
- CodeXGLUE: Shuai Lu et al., "CodeXGLUE: A Machine Learning Benchmark Dataset for Code Understanding and Generation." arXiv:2102.04664. https://arxiv.org/abs/2102.04664
- RepoBench: Tianyang Liu et al., "RepoBench: Benchmarking Repository-Level Code Auto-Completion Systems." arXiv:2306.03091. https://arxiv.org/abs/2306.03091
- SWE-bench: Carlos E. Jimenez et al., "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?" arXiv:2310.06770. https://arxiv.org/abs/2310.06770
- SWE-agent: John Yang et al., "SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering." arXiv:2405.15793. https://arxiv.org/abs/2405.15793
- Defects4J: Rene Just et al., "Defects4J: A Database of Existing Faults to Enable Controlled Testing Studies for Java Programs." ISSTA 2014. https://homes.cs.washington.edu/~mernst/pubs/bug-database-issta2014-abstract.html
- IR-based bug localization comparison: "A Large-Scale Comparative Evaluation of IR-Based Tools for Bug Localization." MSR 2020. https://2020.msrconf.org/details/msr-2020-papers/35/A-Large-Scale-Comparative-Evaluation-of-IR-Based-Tools-for-Bug-Localization
- Retrieval-Augmented Generation: Patrick Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." arXiv:2005.11401. https://arxiv.org/abs/2005.11401
- Aider repo map documentation. https://aider.chat/docs/repomap.html
