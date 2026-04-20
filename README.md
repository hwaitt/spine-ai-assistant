# 🏥 脊柱外科 AI 辅助诊断系统

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python" />
  <img src="https://img.shields.io/badge/Qwen2.5--7B-LoRA微调-orange" />
  <img src="https://img.shields.io/badge/vLLM-高性能推理-green" />
  <img src="https://img.shields.io/badge/LangGraph-多Agent协作-purple" />
  <img src="https://img.shields.io/badge/MCP-工具协议-red" />
  <img src="https://img.shields.io/badge/RAG-ChromaDB-yellow" />
  <img src="https://img.shields.io/badge/CUDA-12.1+-76B900?logo=nvidia" />
</p>

> 基于 **Qwen2.5-7B LoRA 微调** + **LangGraph 多 Agent 协作** + **双知识库 RAG** + **MCP 协议** + **Qwen-VL 多模态图像理解** 构建的脊柱内窥镜手术智能辅助平台，实现从图像理解 → 病历检索 → 文献查询 → 综合诊断建议的完整闭环。

---

## ✨ 核心亮点

| 亮点 | 说明 |
|------|------|
| 🧠 **领域微调模型** | 基于 6000+ 条脊柱外科专业问答数据集，对 Qwen2.5-7B-Instruct 进行 LoRA 微调，train_loss 1.42，无过拟合 |
| 🤖 **多 Agent 协作** | LangGraph 路由节点 + 病历Agent / 论文Agent 并行执行 + 汇总Agent 整合，按需激活避免资源浪费 |
| 🔌 **MCP 工具协议** | 工具服务解耦为独立进程，热插拔式扩展，预留接入医院 HIS 系统接口 |
| 📚 **双知识库 RAG** | 病历库 + 论文库，独立 ChromaDB，BGE-small-zh 语义检索，支持动态扩充 |
| 👁️ **多模态图像理解** | Qwen-VL 自动识别内窥镜图像中的黄韧带、硬膜外脂肪等组织，描述自动注入诊断上下文 |
| 💬 **患者维度记忆** | 每位患者独立会话 thread_id，切换患者自动加载历史，真正多轮上下文记忆 |

---

## 🏗️ 系统架构

```
医生提问 + 内窥镜图像上传
         │
         ├─── Qwen-VL API ──► 图像描述（组织结构识别）
         │                          │
         ▼                          ▼
     路由节点（LangGraph）◄──── 注入上下文
         │
    关键词判断问题类型
         │
    ┌────┴────┐
    ▼         ▼
病历 Agent   论文 Agent        ← 并行执行
│            │
├ 查患者病历  ├ 检索医学文献
├ 评估手术风险 └ 提供循证依据
└ 计算用药剂量
    │         │
    └────┬────┘
         ▼
     汇总 Agent
  （整合结果，简洁输出）
         │
         ▼
  Qwen2.5-7B 微调模型
     vLLM 推理服务
         │
         ▼
     Gradio 医生界面
```

**MCP 工具服务（独立进程）：**
```
LangGraph Agent
      │
   MCP Client
      ├── patient_server.py   ← 病历查询 / 过敏史 / 患者列表
      └── knowledge_server.py ← 知识库检索 / 剂量计算
```

---

## 🛠️ 技术栈

| 模块 | 技术 | 说明 |
|------|------|------|
| 大模型微调 | Qwen2.5-7B-Instruct + LoRA | LLaMA-Factory WebUI，rank=16，3 epochs |
| 高性能推理 | vLLM | OpenAI 兼容 API，支持工具调用 |
| 图像理解 | Qwen-VL API | 阿里云 DashScope，多模态组织识别 |
| 知识检索 | RAG + ChromaDB + BGE | 双知识库，MMR 检索策略 |
| Agent 框架 | LangGraph ReAct | 多 Agent 并行 + 路由调度 |
| 工具协议 | MCP | 工具服务解耦，独立进程部署 |
| 对话管理 | LangChain + MemorySaver | 患者维度会话隔离 |
| 后端接口 | FastAPI | REST API，预留 RK3588 边缘设备接口 |
| 前端界面 | Gradio | 患者切换、图像上传、多轮对话 |
| Embedding | BGE-small-zh-v1.5 | 本地部署，CPU 运行 |

---

## 🚀 快速启动

### 环境要求

- Python 3.12
- CUDA 12.1+
- RTX 4090（24GB）推荐
- 阿里云 DashScope API Key（Qwen-VL 图像理解）

### 安装依赖

```bash
pip install vllm langchain langchain-openai langchain-community
pip install langgraph chromadb sentence-transformers pypdf
pip install fastapi uvicorn gradio dashscope mcp langchain-mcp-adapters
```

### 第一步：启动 vLLM 推理服务

```bash
vllm serve /path/to/spine_qwen25_merged \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
```

### 第二步：设置环境变量

```bash
export DASHSCOPE_API_KEY="your_dashscope_api_key"
export OMP_NUM_THREADS=1
```

### 第三步：构建知识库

```bash
cd medical-ai/
python rebuild_db.py
```

### 第四步：启动服务

```bash
# 启动 FastAPI 后端
python server.py

# 启动多 Agent 主程序（新终端）
python multi_agent_app.py
```

### 第五步：访问界面

---

## 📋 新增病历 / 论文

### 新增患者病历

```bash
# 1. 按格式新建病历文件
nano docs/patient_xxx.txt
```

病历文件格式：
```
患者姓名：XXX
年龄：XX岁
主诉：XXXX
MRI报告：XXXX
既往史：XXXX
过敏史：XXXX
手术风险评估：XXXX
麻醉建议：XXXX
```

```bash
# 2. 重建知识库（自动更新患者列表）
python rebuild_db.py

# 3. 重启服务
python multi_agent_app.py
```

### 新增医学论文

```bash
# 将 PDF 放入 papers/ 目录
cp your_paper.pdf papers/

# 重建论文向量库
python rebuild_papers_db.py
```

---

## 📁 项目结构

```
medical-ai/
├── docs/                      # 患者病历文件（txt格式）
├── papers/                    # 医学论文（PDF格式）
├── mcp_servers/               # MCP 独立服务
│   ├── patient_server.py      # 病历查询服务
│   └── knowledge_server.py   # 知识库检索服务
├── chroma_db/                 # 病历向量数据库
├── chroma_papers/             # 论文向量数据库
├── multi_agent_app.py         # 多 Agent 主程序（推荐）
├── agent_app.py               # 单 Agent 版本
├── agent_mcp.py               # MCP Agent 版本
├── server.py                  # FastAPI 后端
├── rebuild_db.py              # 重建病历知识库
└── README.md
```

---

## 🤖 Agent 工具说明

| 工具 | 所属 Agent | 功能 |
|------|-----------|------|
| `search_patient_record` | 病历 Agent | 根据患者姓名查询完整病历 |
| `get_surgery_risk` | 病历 Agent | 从病历提取手术风险评估信息 |
| `calculate_drug_dosage` | 病历 Agent | 根据药物名称和体重计算用药剂量 |
| `search_medical_literature` | 论文 Agent | 检索医学论文知识库，标注文献来源 |

---

## 📊 模型训练信息

| 参数 | 值 |
|------|-----|
| 基座模型 | Qwen2.5-7B-Instruct |
| 微调方法 | LoRA（rank=16，alpha=32） |
| 训练框架 | LLaMA-Factory |
| 数据集 | 6000+ 条脊柱外科专业问答（JSONL） |
| 数据来源 | HuaTuo 关键词筛选 + Qwen-Plus API 蒸馏 |
| 训练轮数 | 3 epochs |
| train_loss | 1.4158 |
| eval_loss | 1.4296 |
| 训练硬件 | RTX 4090（24GB）|
| 训练时长 | 约 42 分钟 |

## 📄 License

MIT License