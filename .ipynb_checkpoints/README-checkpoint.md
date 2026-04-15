# 🏥 脊柱外科 AI 辅助系统

基于大模型的脊柱内窥镜手术智能辅助平台，实现图像理解、病历检索、智能问答的完整闭环。

## 系统架构
内窥镜图像
↓
Qwen-VL 图像理解
↓
FastAPI 后端
├── RAG 检索患者病历（ChromaDB）
└── Qwen-7B 微调模型推理（vLLM）
↓
Gradio 医生交互界面
## 技术栈

| 模块 | 技术 |
|------|------|
| 大模型微调 | Qwen-7B + LoRA |
| 高性能推理 | vLLM |
| 图像理解 | Qwen-VL |
| 知识检索 | RAG + ChromaDB |
| 对话管理 | LangChain |
| 后端接口 | FastAPI |
| 前端界面 | Gradio |

## 核心功能

- 📷 上传内窥镜图像，AI自动识别组织结构
- 📋 自动检索患者病历、MRI报告
- 💬 多轮对话，结合图像+病历给出诊断建议
- 🔌 标准 REST API，支持与RK3588边缘设备对接

## 快速启动

### 1. 启动推理服务
```bash
vllm serve /root/autodl-tmp/qwen7b_medical_merged \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
```
cd /root/autodl-tmp/medical-ai/
添加完数据库后
python rebuild_db.py
### 2. 启动后端
```bash
python server.py
```

### 3. 启动前端
```bash
export OMP_NUM_THREADS=1
export DASHSCOPE_API_KEY="你的key"
python agent_app.py
```

### 4. 访问界面
浏览器打开 `http://localhost:6006`
