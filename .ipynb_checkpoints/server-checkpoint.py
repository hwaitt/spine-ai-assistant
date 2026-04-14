from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import uvicorn

app = FastAPI(title="脊柱外科AI辅助系统")

# ============ 启动时初始化所有模块 ============
print("正在初始化AI系统...")

llm = ChatOpenAI(
    api_key="not-needed",
    base_url="http://localhost:8000/v1",
    model="/root/autodl-tmp/qwen7b_medical_merged",
    temperature=0.7,
    max_tokens=256,
    model_kwargs={
        "extra_body": {"repetition_penalty": 1.3}
    }
)

loader = DirectoryLoader(
    "/root/autodl-tmp/medical-ai/docs",
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)
docs = loader.load()
chunks = RecursiveCharacterTextSplitter(
    chunk_size=300, chunk_overlap=50
).split_documents(docs)

embeddings = HuggingFaceEmbeddings(
    model_name="/root/autodl-tmp/models/AI-ModelScope/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

vectorstore = Chroma(
    persist_directory="/root/autodl-tmp/medical-ai/chroma_db",
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 每个会话独立维护对话历史
session_histories = {}

print("AI系统初始化完成！")

# ============ 请求/响应数据结构 ============
class ChatRequest(BaseModel):
    question: str
    cv_result: str = ""       # 来自RK3588的视觉检测结果，可选
    session_id: str = "default"  # 会话ID，用于多轮对话

class ChatResponse(BaseModel):
    answer: str
    context_used: str         # 检索到的病历片段，方便调试

# ============ 核心接口 ============
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # 1. 检索相关病历
    related_docs = retriever.invoke(req.question)
    context = "\n".join([d.page_content for d in related_docs])

    # 2. 构建 system prompt（融合CV结果）
    cv_info = f"\n【视觉检测结果】\n{req.cv_result}" if req.cv_result else ""
    system_msg = f"""你是脊柱外科AI助手。根据资料简洁回答，没有的信息说"资料未记录"。
禁止输出与医疗无关的内容，回答控制在150字以内。

【患者相关资料】
{context}{cv_info}"""

    # 3. 拼接历史对话（最近3轮）
    history = session_histories.get(req.session_id, [])
    messages = [{"role": "system", "content": system_msg}]
    for h in history[-3:]:
        messages.append({"role": "user", "content": h[0]})
        messages.append({"role": "assistant", "content": h[1]})
    messages.append({"role": "user", "content": req.question})

    # 4. 调用模型
    response = llm.invoke(messages)
    answer = response.content

    # 5. 保存历史
    history.append((req.question, answer))
    session_histories[req.session_id] = history

    return ChatResponse(answer=answer, context_used=context)

# ============ 健康检查接口 ============
@app.get("/health")
def health():
    return {"status": "ok", "sessions": len(session_histories)}

# ============ 清除会话历史 ============
@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    session_histories.pop(session_id, None)
    return {"status": "cleared"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)