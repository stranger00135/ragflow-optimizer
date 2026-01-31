# RAG Ingestion Parameter Optimizer

> Automatically discover the best chunking parameters for each document type and improve your RAG retrieval quality.

**In one sentence**: Given a set of documents, this tool tests different chunking strategies (chunk_size, overlap, auto Q&A, etc.) and finds the configuration that yields the best retrieval performance.

---

## Why this tool?

When building RAG applications, chunking parameters have a huge impact on retrieval quality:

| Issue | Impact |
|-------|--------|
| chunk_size too small | Lost context, incomplete answers |
| chunk_size too large | Too much noise, relevant content diluted |
| Picking parameters by feel | No measurable evaluation, blind optimization |

**This tool runs automated experiments** to find the best configuration for each document type.

---

## Features

- **Fully automated optimization**: Two-phase strategy (preset screening + parameter fine-tuning)
- **Rigorous evaluation**: LLM-based relevance judgment with Precision@3, MRR, fail rate, and combined score
- **Noise-resilient testing**: Injects distractor documents to test chunking robustness
- **Per-folder optimization**: Different document types get different optimal configs
- **Efficient execution**: KB reuse—upload files once, re-chunk multiple times

---

## Quick start

### 1. Environment

```bash
git clone <repo-url>
cd ragflow_eval_v3
pip install -r requirements.txt
```

### 2. Configure credentials

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

Edit `.env`:

```env
# RAGFlow server and API token
RAGFLOW_DOMAIN="http://your-ragflow-server/"
RAGFLOW_TOKEN="ragflow-xxx-your-token"

# LLM API key (choose one according to config.yaml provider)
OPENAI_API_KEY="sk-xxx"
# or DASHSCOPE_API_KEY="sk-xxx"
# or DEEPSEEK_API_KEY="sk-xxx"
```

### 3. Prepare documents

Put documents under `source_docs/`. **Each leaf folder represents one document type** with similar structure; the tool finds an optimal config per type.

Example layout:

```
source_docs/
├── HR/
│   └── Employee_Handbook/     ← one type: policy docs
│       ├── overtime_policy.pdf
│       └── dress_code.docx
└── Quality/
    └── SOPs/                  ← another type: procedure docs
        └── coating_sop.pdf

Distractors/                   # Noise documents for robustness testing
└── random_news.pdf
```

### 4. Run optimization

```bash
# Test API connectivity
python main.py test-api

# Full optimization
python main.py run

# Single folder only
python main.py run --folder "HR/Employee_Handbook"
```

### 5. Results

Output is under `output/run_YYYYMMDD_HHMMSS/`:

- `summary_report.md` — human-readable summary  
- `summary_report.json` — machine-readable results  
- Per-folder `exp_*.json` — experiment details and retrieval results  

---

## How it works

- **Phase 1 (Preset selection)**: Compare presets (e.g. general, manual, QA). The best-performing preset wins.
- **Phase 2 (Fine-tuning)**: Tune parameters (chunk_token_num, auto_questions, auto_keywords, etc.) for the winning preset.
- **Tiebreaker**: If scores are equal, the config with **faster ingestion** wins.

Evaluation uses an LLM to judge relevance of retrieved chunks and computes **Fail Rate**, **Precision@3**, **MRR**, and a **combined score**.

---

## Configuration

Main config: `config/config.yaml`. Use `${VAR}` for values that come from `.env` (e.g. `${RAGFLOW_DOMAIN}`, `${OPENAI_API_KEY}`). Never commit `.env`; keep secrets in environment or `.env` only.

---

## Security (for public release)

- **Secrets**: No API keys or tokens are hardcoded. All credentials are read from environment variables via `.env` (and `.env` is in `.gitignore`). Use `.env.example` as a template only.
- **Path handling**: The `--folder` argument is validated so paths stay under the configured source-docs directory (no path traversal).
- **Dependencies**: `requirements.txt` pins versions; review and update periodically for security advisories.
- **Recommendations**: (1) Do not commit `.env`. (2) Add `.claude/` to `.gitignore` if you do not want to publish internal project rules. (3) Run the tool in a dedicated environment with minimal permissions.

---

## Command reference

```bash
python main.py run                           # Full optimization
python main.py run --folder "path/to/folder" # Single folder
python main.py run --keep-kbs                # Keep temp KBs (debug)
python main.py generate-questions            # Generate questions only
python main.py generate-questions --regenerate
python main.py cleanup                       # Remove temp KBs
python main.py cleanup --force               # Force cleanup all temp KBs
python main.py test-api                      # Test API connectivity
```

---

# RAG 摄取参数优化器（中文）

> 自动发现每个文档类型的最佳分块参数配置，让你的 RAG 检索质量显著提升

**一句话总结**：给定一堆文档，自动测试不同的分块策略（chunk_size、overlap、自动问答等），找出检索效果最好的配置。

---

## 为什么需要这个工具？

在构建 RAG 应用时，分块参数的选择对检索质量影响巨大：

| 问题 | 影响 |
|-----|------|
| chunk_size 太小 | 上下文丢失，答案不完整 |
| chunk_size 太大 | 噪音太多，相关内容被稀释 |
| 参数选择全凭感觉 | 无法量化评估，优化盲目 |

**本工具通过自动化实验**，科学地找到每类文档的最佳配置。

---

## 核心特性

- **全自动优化**：两阶段优化策略（预设筛选 + 参数微调）
- **科学评估**：基于 LLM 的相关性判断，计算 Precision@3、MRR、失败率等指标
- **抗噪测试**：注入干扰文档，测试分块策略的鲁棒性
- **按文件夹优化**：不同类型文档使用不同最优配置
- **高效执行**：KB 复用策略，上传一次文件，多次重新分块

---

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repo-url>
cd ragflow_eval_v3

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置凭证

复制 `.env.example` 为 `.env` 并填入你的 API 密钥：

```bash
cp .env.example .env
```

编辑 `.env`：

```env
# RAGFlow 服务地址和 API Token
RAGFLOW_DOMAIN="http://your-ragflow-server/"
RAGFLOW_TOKEN="ragflow-xxx-your-token"

# LLM API Key（根据 config.yaml 中的 provider 配置选择一个）
OPENAI_API_KEY="sk-xxx"
# 或 DASHSCOPE_API_KEY="sk-xxx"
# 或 DEEPSEEK_API_KEY="sk-xxx"
```

### 3. 准备文档

`source_docs/` 下的**每个底层文件夹代表一类具有相似特征的文档**，系统会为每类文档单独找出最优配置。

**文件夹组织示例：**

```
source_docs/
├── 总经办/
│   ├── 人力资源/
│   │   └── 员工手册/     ← 一类文档：包含加班制度、工作服管理等制度文件
│   │       ├── 加班管理制度.pdf
│   │       └── 工作服管理制度.docx
│   │
│   └── 安全环保/
│       └── 环保指标/     ← 另一类文档：包含排放数据表格
│           └── 2024年排放数据.xlsx
│
└── 质量管理/
    └── 操作规程/         ← 又一类：SOP 类流程文档
        └── 涂布工艺规程.pdf

Distractors/              # 干扰文档（用于测试抗噪能力）
└── 无关新闻文章.pdf
```

**为什么按文件夹组织？**
- 同一文件夹内的文档结构相似（如都是制度文档、都是表格、都是操作规程）
- 相似结构的文档适合使用相同的分块参数
- 系统会为每个底层文件夹独立运行优化，找到该类文档的最佳配置

### 4. 运行优化

```bash
# 测试 API 连通性
python main.py test-api

# 运行完整优化
python main.py run

# 仅优化单个文件夹
python main.py run --folder "人力资源/员工手册"
```

### 5. 查看结果

优化完成后，结果保存在 `output/run_YYYYMMDD_HHMMSS/` 目录：

```
output/run_20260117_143022/
├── summary_report.md      # 人类可读的汇总报告
├── summary_report.json    # 机器可读的结果
└── 人力资源_员工手册/
    ├── exp_001.json       # 实验详情（含每个问题的检索结果）
    ├── exp_002.json
    └── ...
```

---

## 工作原理

### 两阶段优化策略

```
阶段 1: 预设筛选（Tournament）
├── General（通用分块）  ──┐
├── Manual（层级分块）   ──┼── 选出得分最高的预设
└── QA（问答对分块）     ──┘

阶段 2: 参数微调（Fine-Tuning）
└── 基于获胜预设，逐个调优参数
    ├── chunk_token_num: [256, 512, 768, 1024]
    ├── auto_questions: [0, 3, 5]
    └── auto_keywords: [3, 5, 7]
```

**得分相同时的决胜规则**：当多个配置得分相同时，选择**摄取时间更短**的配置（更高效的配置优先）。

### 问题生成与缓存

系统使用 LLM 自动为每个文件夹生成评估问题：

1. **读取文档内容**：提取文件夹中所有文档的文本
2. **调用 LLM 生成问题**：根据文档内容生成 N 个有代表性的问题
3. **缓存问题**：保存到 `questions_cache/` 避免重复生成

```
questions_cache/
└── 总经办_人力资源__员工手册.json  # 缓存的问题列表
```

**缓存策略**：
- 相同文件夹的问题只生成一次
- 使用 `--regenerate` 可强制重新生成
- 问题缓存让多次运行实验时保持评估一致性

### 评估指标说明

| 指标 | 含义 | 计算方式 |
|-----|------|---------|
| **Fail Rate** | 完全检索失败率 | Top-3 中无任何相关文档的问题占比 |
| **Precision@3** | Top-3 精确率 | Top-3 中相关文档的比例 |
| **MRR** | 平均倒数排名 | 第一个相关文档排名倒数的平均值 |
| **综合得分** | 加权总分 | `(1-FailRate)×0.4 + P@3×0.3 + MRR×0.3` |

### 相关性判断

使用 LLM 对每个检索到的文本块进行相关性判断：

```
问题: 加班申请的流程是什么？
检索片段: [加班管理制度第三条...]

LLM 判断:
- relevant: true
- rationale: 该片段直接描述了加班申请的具体流程...
```

---

## 配置说明

主配置文件 `config/config.yaml`：

```yaml
# RAGFlow 服务配置
ragflow:
  base_url: "${RAGFLOW_DOMAIN}"
  api_key: "${RAGFLOW_TOKEN}"

# LLM 配置（用于生成问题和判断相关性）
llm:
  provider: "openai"  # openai / dashscope / deepseek
  model: "gpt-4o-mini"
  api_key: "${OPENAI_API_KEY}"

# 评估设置
evaluation:
  questions_per_folder: 5    # 每个文件夹生成的测试问题数
  top_k: 3                   # 检索返回的文档数

# 预设配置
presets:
  general:
    chunk_method: naive
    chunk_token_num: 512
    auto_questions: 5
    auto_keywords: 5

  manual:
    chunk_method: manual
    chunk_token_num: 800
    toc_enhance: true
    auto_keywords: 5

# 参数搜索空间
optimization_spaces:
  general:
    - name: chunk_token_num
      values: [256, 512, 768, 1024]
    - name: auto_questions
      values: [0, 3, 5]
```

---

## 命令行参考

```bash
# 运行完整优化
python main.py run

# 优化指定文件夹
python main.py run --folder "部门/子目录"

# 保留临时知识库（调试用）
python main.py run --keep-kbs

# 仅生成测试问题（不运行实验）
python main.py generate-questions
python main.py generate-questions --regenerate  # 强制重新生成

# 清理临时知识库
python main.py cleanup
python main.py cleanup --force  # 强制清理所有临时 KB

# 测试 API 连通性
python main.py test-api
```

---

## 输出示例

### 汇总报告 (summary_report.md)

```markdown
# Optimization Report
Run ID: run_20260117_143022

## 人力资源/员工手册

### Phase 1: Preset Selection

| Exp ID | Preset | Status | Files | Chunks | Time(s) | Fail Rate | P@3 | MRR | Score |
|--------|--------|--------|-------|--------|---------|-----------|-----|-----|-------|
| exp_001 | general | completed | 3/3 | 42 | 10.2 | 0.20 | 0.60 | 0.70 | 0.7100 |
| exp_002 | manual | completed | 3/3 | 38 | 8.5 | 0.10 | 0.75 | 0.80 | 0.7600 |
| exp_003 | qa | disqualified | 3/3 | 0 | 5.1 | - | - | - | - |

**Phase 1 Winner: manual (exp_002)**

### Phase 2: Fine-Tuning (manual)
...

### Recommended Configuration

{
  "chunk_method": "manual",
  "chunk_token_num": 1024,
  "auto_keywords": 5,
  "toc_enhance": true
}
```

### 实验详情 (exp_001.json)

```json
{
  "experiment_id": "exp_001",
  "folder_path": "人力资源/员工手册",
  "status": "completed",
  "config": {
    "chunk_method": "naive",
    "chunk_token_num": 512
  },
  "ingestion_details": {
    "total_files": 3,
    "ingested_files": 3,
    "chunk_count": 42,
    "ingestion_time_seconds": 10.2
  },
  "metrics": {
    "fail_rate": 0.20,
    "precision_at_k": 0.60,
    "mrr": 0.70,
    "combined_score": 0.71
  },
  "questions": [
    {
      "question": "加班申请的流程是什么？",
      "chunks_retrieved": [
        {
          "content": "加班管理制度...",
          "relevant": true,
          "rationale": "该片段直接描述了加班申请流程..."
        }
      ]
    }
  ]
}
```

---

## 项目结构

```
ragflow_eval_v3/
├── config/
│   └── config.yaml          # 主配置文件
│
├── source_docs/              # 待优化的文档
├── Distractors/              # 干扰文档
├── questions_cache/          # 问题缓存
├── output/                   # 输出结果
│
├── src/
│   ├── orchestrator.py       # 主控制器
│   ├── ingestion_engine.py   # 文档摄取引擎
│   ├── retrieval_engine.py   # 检索引擎
│   ├── relevance_judge.py    # 相关性判断
│   ├── question_generator.py # 问题生成器
│   ├── metrics.py            # 指标计算
│   ├── config_loader.py      # 配置加载
│   └── ragflow_client.py     # RAGFlow API 封装
│
├── main.py                   # CLI 入口
├── .env                      # 环境变量（不提交）
├── .env.example              # 环境变量模板
└── requirements.txt
```

---

## 常见问题

### Q: 为什么有些预设会被踢出比较？

当某个分块策略无法生成任何 chunks（如 QA 模式处理非问答格式文档），系统会自动将其标记为 disqualified 并跳过评估。

### Q: 如何处理中断的运行？

```bash
# 清理可能残留的临时知识库
python main.py cleanup --force
```

---

## 技术原理

### KB 复用策略

为提高效率，每个文件夹只创建一个临时 KB：

```
1. 创建临时 KB
2. 上传所有文件（只上传一次）
3. 对每个实验：
   a. 更新解析器配置
   b. 重新触发解析
   c. 等待解析完成
   d. 运行检索测试
4. 删除临时 KB
```

### 干扰文档注入

为测试分块策略的抗噪能力：

```python
# 检索时同时查询目标 KB 和干扰 KB
chunks = retrieve(
    question=question,
    dataset_ids=[target_kb_id, distractor_kb_id]
)

# 来自干扰 KB 的 chunks 自动标记为不相关
```
