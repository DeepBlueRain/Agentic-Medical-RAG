# Agentic Medical RAG

一个面向医学科普文章的 RAG 问答项目。它使用本地 embedding 和 Milvus Lite 做文档检索，使用在线 OpenAI-compatible 模型生成中文回答；LangGraph 负责把查询分析、检索、证据判断、回答和校验串成一个可观察的工作流。

## 工作流

```text
用户问题
  -> 路由与意图分析
  -> 复合问题拆分（必要时）
  -> 生成多个检索 query
  -> dense 检索 + 本地关键词检索
  -> Reciprocal Rank Fusion 合并排序
  -> 证据质量判断
  -> 证据不足时扩展 query 并补检索一次
  -> 基于证据生成回答
  -> 在线模型依据校验
  -> 校验失败时严格重生成一次
  -> 返回答案、证据和完整 trace
```

这里的 Agent 不是让模型随意调用工具，而是把每个能力做成 LangGraph 节点，通过状态和条件边控制执行路径。比如证据质量为 `weak` 或 `no_context` 时，图会回到 `retry_query`，最多补检索一次；答案依据不足时进入 `revise_answer`，最多严格重生成一次。

## 主要功能

- 使用 BeautifulSoup 解析并切分医学 HTML 文档。
- 使用 Sentence-Transformers 生成 384 维 embedding，Milvus Lite 保存向量并完成 dense 检索。
- 使用不依赖额外服务的本地 BM25 风格关键词检索，标题命中会加权。
- 对多个改写 query 的 dense/关键词结果按各自排名做 RRF 融合，并保留检索来源、排名、匹配 query 和分数。
- 根据证据距离、实体命中和关键词命中情况进行证据分级。
- 在线模型调用失败时有限重试并指数退避；仍失败时切换到只摘录召回证据的 fallback。
- Streamlit 展示工作流节点、工具调用、检索来源、证据分数和延迟。
- FastAPI `/ask` 接口返回答案、工作流事件和检索证据，方便调试与后续评测。

## 技术栈

Python、LangGraph、FastAPI、Streamlit、Milvus Lite、Sentence-Transformers、BeautifulSoup、OpenAI-compatible API。

## 目录说明

```text
agentic_rag.py       LangGraph 状态、节点和条件分支
lexical_retriever.py 本地关键词检索和多 query 结果合并
milvus_utils.py      Milvus Lite 初始化、建库和 dense 检索
llm_client.py        在线模型请求、重试和响应解析
rag_core.py          受证据约束的回答 prompt
app.py               Streamlit 演示页面
api.py               FastAPI 接口
preprocess.py        HTML 解析和文本切分
config.py            路径、检索和模型配置
```

## 配置

复制 `.env.example` 为 `.env`，填写在线模型配置：

```dotenv
ONLINE_LLM_BASE_URL=https://apihub.agnes-ai.com/v1
ONLINE_LLM_API_KEY=your-api-key
ONLINE_LLM_MODEL=agnes-2.5-flash
```

不要把 `.env` 或 API Key 提交到 Git。embedding 模型可以通过 `EMBEDDING_MODEL_NAME` 指向本地目录，例如：

```dotenv
EMBEDDING_MODEL_NAME=G:/shujuwajue/model/all-MiniLM-L6-v2
```

默认数据文件为 `data/processed_data.json`，Milvus Lite 数据库为 `data/medical_rag.db`。这两个文件都被 `.gitignore` 排除；公开仓库只保留代码和配置示例。

## 运行

在项目目录使用已经配置好依赖的 Python 环境执行：

```bash
streamlit run app.py
```

或者启动 API：

```bash
uvicorn api:app --host 127.0.0.1 --port 8000
```

健康检查：

```text
GET http://127.0.0.1:8000/health
```

请求示例：

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"白血病为什么会发烧？"}'
```

## 面试时可以说明的工程点

- 纯向量检索对医学实体和固定术语不一定稳定，因此增加关键词检索，并使用 RRF 融合两路结果。
- Agent 的核心在于状态流转和条件分支，而不是节点数量；证据不足会触发受控补检索，回答校验失败会触发一次严格重生成。
- 每一步都记录输入摘要、输出结果和耗时，页面可以直接查看 query 改写、检索来源、证据质量和最终回答。
- 在线模型只负责生成和校验，embedding、向量库和检索逻辑留在本地，便于复现和替换模型服务。

项目不承诺医学诊断能力，回答只用于展示基于已有科普资料的检索和生成流程。
