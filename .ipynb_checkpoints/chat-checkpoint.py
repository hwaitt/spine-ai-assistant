from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 修复 surrogate 编码问题
def clean_text(text: str) -> str:
    return text.encode('utf-8', errors='ignore').decode('utf-8')

llm = ChatOpenAI(
    api_key="not-needed",
    base_url="http://localhost:8000/v1",
    model="/root/autodl-tmp/qwen7b_medical_merged",
    temperature=0.7,
    max_tokens=512,
    model_kwargs={
        "extra_body": {
            "repetition_penalty": 1.15
        }
    }
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一名专业的脊柱外科AI辅助助手，专注于脊柱内窥镜手术领域。
你具备以下专业知识：
- 脊柱解剖结构（黄韧带、硬膜外脂肪、神经根等）
- 脊柱内窥镜手术操作要点
- 组织学特征识别与鉴别诊断
请基于专业知识给出准确、简洁的回答。"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chain = prompt | llm

chat_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

print("=" * 50)
print("  脊柱外科 AI 辅助助手已启动")
print("  输入 quit 退出")
print("=" * 50)

session_id = "doctor_session_001"

while True:
    user_input = input("\n医生: ").strip()
    user_input = clean_text(user_input)

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
        print(clean_text(response.content))
    except Exception as e:
        print(f"\n[错误] {e}")
        print("请重新输入")