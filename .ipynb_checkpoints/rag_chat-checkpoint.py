from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
import os

print("正在初始化，请稍候...")

# ============ 1. 连接 vLLM ============
llm = ChatOpenAI(
    api_key="not-needed",
    base_url="http://localhost:8000/v1",
    model="/root/autodl-tmp/qwen7b_medical_merged",
    temperature=0.7,
    max_tokens=512,
    model_kwargs={
        "extra_body": {"repetition_penalty": 1.15}
    }
)

# ============ 2. 加载文档 ============
print("加载文档...")
loader = DirectoryLoader(
    "/root/autodl-tmp/medical-ai/docs",
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)
docs = loader.load()

# ============ 3. 切分文档 ============
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)
chunks = splitter.split_documents(docs)
print(f"文档切分完成，共 {len(chunks)} 个片段")

# ============ 4. 向量化 + 存入数据库 ============
print("向量化中（首次运行较慢）...")
embeddings = HuggingFaceEmbeddings(
    model_name="/root/autodl-tmp/models/AI-ModelScope/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

vectorstore = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="/root/autodl-tmp/medical-ai/chroma_db"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("知识库构建完成！")

# ============ 5. 定义 Prompt ============
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一名专业的脊柱外科AI辅助助手。
请根据以下检索到的患者资料和医学知识回答问题。
如果资料中有相关信息，优先基于资料回答。

【检索到的相关资料】
{context}
"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# ============ 6. 多轮对话记忆 ============
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# ============ 7. 完整 RAG 链 ============
def get_context(inputs):
    query = inputs["input"]
    docs = retriever.invoke(query)
    return "\n\n".join([d.page_content for d in docs])

rag_chain = (
    RunnablePassthrough.assign(context=get_context)
    | prompt
    | llm
)

chat_with_history = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# ============ 8. 对话主循环 ============
print("\n" + "=" * 50)
print("  脊柱外科 AI 辅助助手（RAG版）已启动")
print("  模型已接入患者病历知识库")
print("  输入 quit 退出")
print("=" * 50)

session_id = "doctor_001"

while True:
    user_input = input("\n医生: ").strip()

    if user_input.lower() in ["quit", "exit", "退出"]:
        print("助手: 再见！")
        break
    if not user_input:
        continue

    print("助手: ", end="", flush=True)

    try:
        response = chat_with_history.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        print(response.content)
    except Exception as e:
        print(f"\n[错误] {e}")