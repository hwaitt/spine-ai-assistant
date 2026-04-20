# 🏥 脊柱外科 AI 辅助系统

基于大模型 + RAG + Agent + MCP 的脊柱内窥镜手术智能辅助平台

## 系统架构
内窥镜图像 → Qwen-VL 图像理解
        ↓
LangGraph Agent
        ↓
MCP 协议调度
┌───────┴───────┐
病历MCP服务      知识库MCP服务
(患者病历)      (RAG+ChromaDB)
└───────┬───────┘
        ↓
Qwen-Plus 推理
        ↓
Gradio 医生界面

## 技术栈

| 模块 | 技术 |
|------|------|
| 大模型微调 | 基于LLaMA-Factory，Qwen-7B + LoRA |
| 高性能推理 | vLLM |
| 图像理解 | Qwen-VL |
| 知识检索 | RAG + ChromaDB + BGE |
| Agent框架 | LangGraph ReAct |
| 工具协议 | MCP (Model Context Protocol) |
| 对话管理 | LangChain |
| 后端接口 | FastAPI |
| 前端界面 | Gradio |

## 核心功能

- 📷 上传内窥镜图像，Qwen-VL 自动识别组织结构
- 📋 RAG 自动检索患者病历、MRI报告
- 🤖 Agent 自主决策：查病历/搜知识/算剂量/评估风险
- 🔌 MCP 协议连接各服务，支持扩展接入医院HIS系统
- 💬 多轮对话，记忆上下文

## 快速启动

### 环境要求
- Python 3.12
- CUDA 12.1+
- RTX 4090 (24GB)

### 启动步骤
cd /root/autodl-tmp/medical-ai/
```bash
# 1. 启动 vLLM 推理服务
vllm serve /root/autodl-tmp/spine_qwen25_merged \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
# 2.启动后端
python server.py
# 3. 设置环境变量
export DASHSCOPE_API_KEY="your_key"
export OMP_NUM_THREADS=1

# 4. 启动 Agent 系统
python agent_mcp.py
```

### 新增病历

```bash
# 1. 在 docs/ 目录新建 txt 文件
nano docs/patient_xxx.txt

# 2. 重建知识库
python rebuild_db.py

# 3. 重启服务
python agent_mcp.py
```

## 项目结构
medical-ai/
├── docs/                 # 患者病历文件
├── mcp_servers/          # MCP 服务
│   ├── patient_server.py # 病历查询服务
│   └── knowledge_server.py # 知识库服务
├── agent_mcp.py          # MCP Agent 主程序
├── agent_app.py          # 普通 Agent 版本
├── server.py             # FastAPI 后端
├── rebuild_db.py         # 重建知识库脚本
└── README.md
