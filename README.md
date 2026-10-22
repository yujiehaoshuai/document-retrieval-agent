# 本地文档检索问答 Agent

Python / LangGraph / BM25 / Streamlit。适合个人学习及 AI 应用开发面试。

参考 MIT 项目 https://github.com/langchain-ai/retrieval-agent-template 的 retrieve → respond 工作流，在本项目重新实现轻量本地版本，并增加无匹配拒答分支。保留上游 LICENSE；没有声称实现原项目的云向量数据库能力。

## 改进内容

- UTF-8 TXT / Markdown 上传，限制单文件1MB、总字符数200万。
- SHA256 内容去重、固定窗口重叠分块、来源和字符偏移追溯。
- 英文词与中文二元字组分词，BM25 排序、Top-3 证据。
- LangGraph 条件分支：有匹配才回答，无匹配直接拒答。
- 离线模式返回摘录；可选模型基于证据生成回答。
- 展示检索分数和执行路径，模型发送资料前要求明确确认。

## 运行

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run main.py --server.address 127.0.0.1 --server.port 8502
```

无需密钥即可演示。真实模型模式通过环境变量 OPENAI_API_KEY、OPENAI_MODEL、可选 OPENAI_BASE_URL 配置。不要提交密钥。

演示：默认查询“发布失败怎么回滚”；再查“缓存穿透”；最后问“火星天气”演示拒答。上传自己的 Markdown 后会替换示例资料。

## 测试

`python -m unittest -v test_retrieval.py`；`python smoke_ui.py`。

测试覆盖排序、去重、重叠分块、引用、拒答、模型模拟路径、空查询和参数约束。真实模型效果需有密钥后单独验证。

## 可用于简历的表述

参考 LangGraph 开源检索模板实现本地文档问答工作流，增加内容去重、重叠分块和 BM25 检索；设计证据引用与无匹配拒答分支，提供离线演示及可选模型回答，并通过自动测试验证检索与界面流程。

## 面试边界

这是受控 RAG 工作流，非自主多 Agent 系统。BM25 是关键词检索，不是向量检索；中文二元字组可能误匹配，同义词召回不足。无匹配拒答不能保证所有回答正确；模型引用格式和语义尚未强制校验。索引每次内存重建，未做生产规模性能评测。上传内容只会在启用模型并确认后发送相关片段到配置的服务；本项目不提供登录和公网多租户能力。

可讲的取舍：为何先用可解释 BM25；分块大小与重叠的影响；内容去重与文件名去重的区别；为什么“搜到”不等于“能回答”；如何用真实问题集评测 Recall@k 和拒答质量。
