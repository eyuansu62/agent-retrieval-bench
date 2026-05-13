# Agent Retrieval Bench V1：面向 Coding Agent 的代码检索基准

> 工作草稿。本文介绍 Agent Retrieval Bench V1 的动机、任务设计、数据构建方式、首批 baseline 结果，以及下一阶段 V1.1 的方向。

## 背景：coding agent 的瓶颈不只在生成代码

过去两年，代码模型和 coding agent 的评测重点大多集中在“最终能不能写出正确 patch”。这当然重要，但在真实软件工程任务里，patch 生成通常不是第一步。一个 agent 在动手修改之前，必须先完成一系列上下文定位工作：理解仓库结构、找到相关实现、找到已有测试、判断失败日志对应的 root cause、识别 review comment 背后的设计约束。

换句话说，coding agent 的能力链条至少包含三个阶段：

1. **检索上下文**：从一个大仓库里找到需要阅读的文件。
2. **推理和规划**：理解这些文件之间的关系，判断该怎么改。
3. **生成和验证 patch**：修改代码、运行测试、迭代修复。

现有很多 benchmark 主要评估第三步，或者把第一步隐含在端到端任务里。这样会带来一个问题：当 agent 失败时，我们很难区分失败原因到底是“不会写 patch”，还是“一开始就没找到该看的文件”。在真实仓库中，后者非常常见。一个模型即使具备很强的代码生成能力，如果上下文检索错了，也会产生看似合理但实际无效的修改。

Agent Retrieval Bench 的出发点就是把这个前置能力单独拿出来测：**给定一个真实 coding workflow 中的任务信号，retriever 能不能把 agent 带到正确的文件附近？**

## 现有代码检索评测缺了什么？

传统代码检索通常被建模为一个相对直接的信息检索问题：给定自然语言 query，找语义相近的代码片段；或者给定代码片段，找相似实现。这类设定对搜索 API、代码问答、snippet retrieval 很有价值，但它和 coding agent 的实际上下文需求并不完全一致。

真实 agentic workflow 中的 query 往往不是“我要找实现 X”。它可能是：

- 一个 PR title/body，描述的是行为变化，而不是测试文件名。
- 一个 review comment，只指出局部问题，但真正需要看的约束在别的模块。
- 一段失败日志，包含测试名、panic、assertion、stack trace，但 root cause 可能不在 trace frame 里。
- 一个迁移或兼容性任务，影响范围跨配置、runtime、测试和文档。

这些场景有几个共同特点。

第一，**gold 文件不一定和 query 语义相近**。例如 `trace2code` 中，失败日志可能只出现测试函数名，但真正需要找的是 source file 中的行为实现。Embedding 模型如果只依赖语义相似度，可能会优先返回测试文件或日志中直接出现的路径。

第二，**文件关系比单文件语义更重要**。Source/test 映射、package/module 结构、symbol reference、reviewed file 与上下文文件之间的依赖关系，经常决定什么文件应该被找回。这类信号更接近 repo graph，而不是单纯文本相似度。

第三，**泄漏很容易发生**。如果 query 中直接包含 gold path、test basename、fix diff 或后续 patch 内容，检索任务会退化成路径匹配。这样的样本会高估系统能力，尤其会让 lexical baseline 看起来异常强。

第四，**端到端 agent 成功率无法解释检索质量**。一个 agent 可能因为工具、测试环境、patch 生成、依赖安装等原因失败；也可能靠很强的模型推理弥补检索错误。我们需要一个更小、更可诊断的 benchmark，专门衡量 retrieval 这一环。

因此，我们把问题重新定义为 **agentic retrieval**：不是找“语义最像的代码”，而是找“agent 为了完成任务必须阅读的仓库文件”。

## 三个代表性例子

为了说明这个问题不是人为构造出来的，我们可以看三个简化后的任务形态。它们不是完整 PR 复盘，而是从 V1 数据构建过程中反复出现的模式中抽象出来的 representative examples。

### 例子一：PR intent 不直接指向测试文件

Agent 看到的信号可能是一个 PR 描述：

> Add support for optional values in shell completion.

这个 query 很自然地会命中 completion engine 的实现文件，因为 “optional values” 和 “completion” 都是实现侧概念。但如果 agent 的任务是补测试或验证行为，它真正需要找的可能是不同 shell backend 的 completion tests，例如 bash、fish、PowerShell 或 elvish 的测试快照。

容易犯的错误是：retriever 只返回实现文件，或者返回文档和 changelog。这样的结果对理解实现有帮助，但对“应该改哪些测试”帮助有限。

这个例子说明 `code2test` 不是简单的文本相似检索。测试文件名不一定出现在 PR 描述里，甚至测试文件可能和实现文件不在同一目录。retriever 需要学到 source→test 映射、模块边界和行为覆盖关系。

### 例子二：review comment 所在文件不是完整上下文

Review comment 往往落在某个具体 hunk 上，例如 reviewer 在一个 endpoint 或 serializer 的实现附近问：

> Should this be sanitized consistently with the web response path?

如果把任务简化成“找 comment 所在文件”，这个样本会非常容易。但对 coding agent 来说，被评论文件只是已经给定的局部上下文。真正需要看的可能是：

- 另一个 web extension 中已有的 sanitize 逻辑；
- 相关配置属性；
- 已有 endpoint tests；
- 文档中定义的 masking/sanitization 行为。

容易犯的错误是：retriever 把 comment 所在文件排第一，然后认为任务完成。但 agent 实际上还缺少判断一致性的依据。

这就是 `comment2context` 的 motivation：我们把 comment 所在文件放入 `given_files`，不作为主指标 gold；模型必须找的是额外上下文。这样才能测出 review 场景中真正有价值的 retrieval 能力。

### 例子三：failure trace 经常指向测试，而不是 root cause

失败日志里最明显的路径往往是测试文件。例如一个 Go 或 Python 测试失败可能只显示：

```text
--- FAIL: TestOptionalCompletion
    completion_test.go:42: expected candidates [...], got [...]
```

或者一个 panic/traceback 的顶部 frame 在 test helper 中。直接路径匹配会优先返回 `completion_test.go`。但如果测试文件只是复现信号，真正要修改的可能是 completion engine、argument parser、runtime config 或某个 source module。

容易犯的错误是：把 trace 中出现的测试文件当成 root cause。对于人类工程师来说，测试文件告诉你“哪里失败了”，不一定告诉你“哪里错了”。Agent 也一样。

这就是 `trace2code` 的核心：gold 是人工审核过的 root-cause source files，related tests 只作为辅助信息。这个任务要求 retriever 从 failure text、repo structure、source/test 关系和局部符号信号中推断 root cause，而不是做 trace-frame lookup。

这三个例子共同说明：agentic retrieval 的难点不在于 query 没有信息，而在于 query 的信息经常是间接的。一个好的 retriever 需要把局部信号映射到完成任务所需的仓库上下文。

## 我们想回答的研究问题

Agent Retrieval Bench V1 不是为了证明某一种检索方法最好，而是为了提供一个能区分不同检索机制的诊断面。我们特别关心几个问题：

1. **语义 embedding 是否足够？**  
   如果任务信号来自 PR、review comment 或 failure trace，向量检索是否仍然稳定优于 lexical 和结构化方法？

2. **vectorless repo map 是否有独立价值？**  
   Aider-style RepoMap 这类方法不依赖 embedding，而是利用 symbol、path、reference 和 source/test 关系。它是否能在某些 agent 场景里超过 embedding？

3. **不同任务是否需要不同 retrieval bias？**  
   `code2test`、`comment2context`、`trace2code` 看似都是“找文件”，但它们依赖的信号完全不同。一个统一 retriever 是否会在任务之间表现不均衡？

4. **能否构造一个不会被路径泄漏和同名测试支配的 benchmark？**  
   如果 benchmark 太容易，lexical baseline 接近 ceiling，就很难判断高级 retriever 是否真的有用。V1 的设计目标之一，是让任务保持可解但不被简单路径匹配解决。

这些问题决定了 V1 的数据策略：我们不追求一开始就做成数千条样本，而是先做一个小而硬、人工审核、可复现、能暴露方法差异的版本。

## 设计原则

V1 的设计遵循几个原则。

**第一，评测对象是 file-level retrieval。**  
Coding agent 最终需要的是可读上下文文件，而不只是单个 chunk。我们仍然用 chunked corpus 支持 ranking 和 8k budget 统计，但主指标围绕 gold file 是否被找回。

**第二，候选 corpus 固定在 base commit。**  
每条样本都在任务发生前的仓库状态上评测。后续修复、PR final diff、人工 gold 证据都不能进入 query 或候选 corpus。这样可以避免把答案泄漏到索引里。

**第三，query 尽量模拟 agent 实际可见信息。**  
`code2test` 使用 PR 意图和实现变化描述；`comment2context` 使用 review comment 和给定文件；`trace2code` 使用本地复现产生的 failure excerpt。query 不包含 fix patch、raw diff 或 gold 路径。

**第四，gold 必须是“完成任务需要看的文件”。**  
尤其在 `comment2context` 中，comment 所在文件只是 given context，不参与主 Recall；在 `trace2code` 中，测试文件可以是 related tests，但主 gold 必须是 root-cause source file。

**第五，保留多种 baseline。**  
我们同时报告 lexical、RepoMap、Jina 和 Qwen，不把 benchmark 绑定到某一种检索范式。一个好的 agent retrieval benchmark 应该能显示不同归纳偏置的优缺点。

这也是 V1 的核心 motivation：构建一个足够接近 coding agent 工作流、又足够可控和可解释的检索评测层。

## V1 包含什么？

Agent Retrieval Bench V1 是当前冻结的主 benchmark。它包含 225 条人工 curated 样本，覆盖三类任务：

| 任务 | 样本数 | 评测目标 |
| --- | ---: | --- |
| `code2test` | 106 | 给定实现变化或 PR 意图，找相关测试文件 |
| `comment2context` | 51 | 给定 review comment 和被评论文件，找额外必须看的上下文文件 |
| `trace2code` | 68 | 给定真实复现的失败 trace，找 root-cause source files |
| **总计** | **225** |  |

每条样本都在 `repo_at_base_commit` 上评测。也就是说，候选 corpus 是修复前的仓库状态，后续 patch 不会进入索引。这样可以避免把答案泄漏给检索器。

V1 还附带了完整 corpus、baseline 输出和报告。用户可以通过一条命令下载：

```bash
arb download-benchmark --version v1 --local-dir data --force
```

这条命令会自动从 Hugging Face 下载 release bundle、校验 checksum，并解压出：

```text
benchmark/v1/
corpus/v1/
eval/v1/
reports/v1/
```

## 三个任务分别在测什么？

### 1. `code2test`：实现变化应该对应哪些测试？

这个任务模拟 agent 收到一个 PR 或实现变更意图后，需要判断应该看哪些测试文件。query 中不会直接给出测试路径或测试文件 basename。gold 是人工确认的相关测试文件。

它考察的是跨 source/test 的结构联想能力，而不是文件名匹配能力。

### 2. `comment2context`：review comment 之外还要看什么？

Review comment 通常落在某个具体文件上，但 reviewer 的真实意图经常依赖其他上下文：API 约束、配置行为、已有测试、另一个模块的实现等。

V1 中 `comment2context` 把 comment 所在文件视为 given context，不计入主指标。模型必须找的是额外上下文文件。这一点很重要，否则任务会退化成“找到 comment 所在文件”。

### 3. `trace2code`：失败 trace 的 root cause 在哪里？

`trace2code` 来自本地测试复现产生的 failure trace，包括 compile error、assertion failure、panic 等。主 gold 是人工审核过的 root-cause source files；相关测试文件只作为辅助信息，不作为主 gold。

这个任务故意不奖励“trace frame path lookup”。如果 trace 里只有测试文件，模型仍然需要借助仓库结构和 failure signal 推断真正相关的 source file。

## Baseline：我们测了哪些方法？

V1 附带四类 baseline：

1. **lexical**：传统词法检索。
2. **aider-style-repomap**：不使用向量的 RepoMap baseline，通过 file/symbol/reference graph 和 query-aware ranking 排文件。
3. **jina-code-embeddings-0.5b**：代码 embedding 模型。
4. **Qwen3-Embedding-4B**：通用/代码能力更强的 embedding 模型。

主排序指标是 overall `MRR`，同时报告 `Recall@5/10/20` 和 `gold_coverage@8k`。

## 当前结果

整体结果如下：

| Model | Samples | Recall@5 | Recall@10 | Recall@20 | MRR | Gold@8k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-Embedding-4B | 225 | 0.2883 | 0.4033 | 0.5828 | 0.2455 | 0.2542 |
| aider-style-repomap | 225 | 0.3089 | 0.4705 | 0.6299 | 0.2227 | 0.0704 |
| jina-code-embeddings-0.5b | 225 | 0.2230 | 0.3133 | 0.4492 | 0.1883 | 0.1556 |
| lexical | 225 | 0.1970 | 0.3267 | 0.4874 | 0.1450 | 0.0785 |

按任务拆开看，结论更有意思：

| 任务 | 最强 MRR 模型 | MRR | 最强 Recall@20 模型 | Recall@20 |
| --- | --- | ---: | --- | ---: |
| overall | Qwen3-Embedding-4B | 0.2455 | aider-style-repomap | 0.6299 |
| code2test | Qwen3-Embedding-4B | 0.3225 | Qwen3-Embedding-4B | 0.7230 |
| comment2context | jina-code-embeddings-0.5b | 0.3282 | lexical | 0.5752 |
| trace2code | aider-style-repomap | 0.2750 | aider-style-repomap | 0.8064 |

几个直接观察：

- Qwen3-Embedding-4B 在 overall MRR 和 `code2test` 上最强。
- Jina 在 `comment2context` 上 MRR 最强。
- RepoMap 在 `trace2code` 上明显强于 embedding，并且 overall Recall@20 最高。
- `trace2code` 对 embedding 不友好，这不是数据错误，而是一个有价值的信号：失败日志定位更依赖结构、路径、调用关系和 repo graph，而不只是语义相似度。

## 为什么 RepoMap 在 trace2code 上强？

这和任务本身有关。

Embedding 模型擅长把自然语言 query 和语义相近的代码放近。但 failure trace 经常包含：

- 文件路径片段；
- 函数名；
- package/module 名；
- 测试名；
- panic/assertion 的局部文本。

这些信号不一定和 root-cause source file 的语义描述相近，但它们在仓库结构图里很有用。RepoMap baseline 用 file/symbol/reference graph 做 query-aware ranking，因此更容易利用这些结构信号。

这也说明：agentic retrieval 不应该只有 vector baseline。对于某些 coding agent 场景，vectorless repo map、symbol graph、dependency graph 可能比 embedding 更契合。

## 数据构建原则

V1 最重要的原则是：宁可少，也不要把简单或泄漏样本混进去。

我们做了几件事：

- V1 样本全部人工 curated。
- 查询中避免直接出现 gold path、basename、raw patch、fix diff。
- `comment2context` 不把 comment 所在文件作为主 gold。
- `trace2code` 只把 root-cause source files 作为主 gold，测试文件只能做辅助信息。
- 所有样本都在 base commit corpus 上评测，避免用修复后的文件污染候选集。

最终审核结果中，V1 保留 225 条，另有 222 条被丢弃或标为不适合进入主 benchmark。

## 如何复现？

安装 evaluator：

```bash
pip install "git+https://github.com/eyuansu62/agent-retrieval-bench.git"
```

下载 V1：

```bash
arb download-benchmark --version v1 --local-dir data --force
```

验证样本：

```bash
arb validate data/benchmark/v1/*.jsonl
```

跑 lexical baseline：

```bash
arb eval-baseline \
  --derived data/benchmark/v1 \
  --corpus data/corpus/v1 \
  --out data/eval/v1/lexical_summary.json \
  --details data/eval/v1/lexical_details.jsonl \
  --no-keep-list
```

跑 RepoMap baseline：

```bash
arb eval-repomap \
  --derived data/benchmark/v1 \
  --corpus data/corpus/v1 \
  --out data/eval/v1/repomap_summary.json \
  --details data/eval/v1/repomap_details.jsonl \
  --candidate-filter all_files \
  --no-keep-list
```

生成 leaderboard：

```bash
arb report-models \
  --eval-dir data/eval/v1 \
  --out data/reports/v1/model_leaderboard.md \
  --json-out data/reports/v1/model_leaderboard.json
```

数据在 Hugging Face：`eyuansu71/agent_retrieval_bench`。代码在 GitHub：`eyuansu62/agent-retrieval-bench`。

## 下一步：V1.1

V1 已冻结，后续不会修改 `benchmark/v1`。下一步会做 V1.1，重点不是盲目扩大总量，而是定向补薄弱任务：

- `comment2context`：从 51 扩到 80–100，优先跨模块、无路径泄漏、非同目录上下文。
- `trace2code`：从 68 扩到 100+，增加非 Go repo、更多语言和更多 failure 类型。
- `code2test`：当前已有 106 条，暂不默认扩充。

长期看，我们希望 Agent Retrieval Bench 能覆盖更多真实 coding agent 场景，例如 issue/bug report → code、migration task → affected files、API usage → implementation 等。但 V1 的第一步，是先把 code review、test retrieval 和 trace root-cause retrieval 三个核心能力测清楚。

## 结语

Agent 的能力上限不只取决于模型会不会生成 patch，也取决于它能不能在大仓库里找到该看的文件。

Agent Retrieval Bench V1 想测的正是这个前置能力：面对 PR 意图、review comment、失败 trace，一个 retriever 能不能把 agent 带到正确的上下文附近。

从首批结果看，embedding、lexical 和 RepoMap 各有优势，没有一种方法在所有任务上统治。这说明 agentic retrieval 很可能需要混合路线：语义向量、结构图、符号索引和任务感知 ranking 需要一起工作。
