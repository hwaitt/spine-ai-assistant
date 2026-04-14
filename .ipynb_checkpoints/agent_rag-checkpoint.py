import os
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# ============ 1. 连接阿里云 Qwen（负责 Agent 决策）============
llm = ChatTongyi(
    dashscope_api_key=os.environ["DASHSCOPE_API_KEY"],
    model="qwen-plus",
    temperature=0.7,
)

# ============ 2. 加载已有的 RAG 知识库 ============
print("加载知识库...")
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
print("知识库加载完成！")

# ============ 3. 定义工具 ============
@tool
def search_patient_record(patient_name: str) -> str:
    """根据患者姓名查询病历，包括既往史、过敏史、MRI报告、手术风险"""
    records = {
        "张某某": """
        年龄：58岁，主诉：腰痛伴右下肢放射痛3年
        MRI：L4-L5黄韧带肥厚8mm，硬膜囊受压
        既往史：高血压10年，口服氨氯地平
        过敏史：青霉素过敏
        手术风险：中等风险，需控制血压
        麻醉建议：全身麻醉，避免青霉素类抗生素
        """,
        "李某某": """
        年龄：45岁，主诉：颈椎病，上肢麻木
        MRI：C5-C6椎间盘突出
        既往史：糖尿病5年
        过敏史：无
        手术风险：中等风险，需控制血糖
        麻醉建议：全身麻醉，术前血糖控制在8mmol/L以下
        """
    }
    return records.get(patient_name, "未找到该患者病历，请确认姓名")

@tool
def search_medical_knowledge(query: str) -> str:
    """搜索脊柱外科医学知识库，包括解剖知识、手术指南、组织特征等"""
    docs = retriever.invoke(query)
    if not docs:
        return "知识库中未找到相关内容"
    return "\n\n".join([d.page_content for d in docs])

@tool
def calculate_drug_dosage(drug: str, weight_kg: float) -> str:
    """根据药物名称和患者体重计算用药剂量"""
    dosages = {
        "布洛芬": f"{weight_kg * 5:.0f}mg，每日3次，饭后服用",
        "氨氯地平": "5-10mg，每日1次",
        "地塞米松": f"{weight_kg * 0.1:.1f}mg，遵医嘱",
        "克林霉素": f"{weight_kg * 8:.0f}mg，每日3次",
    }
    return dosages.get(drug, f"暂无{drug}剂量规则，请查阅药典")

@tool
def get_surgery_risk(patient_name: str) -> str:
    """评估患者手术风险等级和注意事项"""
    risks = {
        "张某某": "中等风险：高血压需术前控制，青霉素过敏需备用抗生素方案",
        "李某某": "中等风险：糖尿病需术前血糖控制在8mmol/L以下"
    }
    return risks.get(patient_name, "未找到该患者风险评估")

# ============ 4. 创建带记忆的 Agent ============
tools = [
    search_patient_record,
    search_medical_knowledge,
    calculate_drug_dosage,
    get_surgery_risk
]

# 关键：加入 MemorySaver 实现多轮记忆
memory = MemorySaver()

agent = create_react_agent(
    model=llm,
    tools=tools,
    checkpointer=memory,  # 记忆模块
    prompt="""你是脊柱外科AI助手，拥有以下工具：
- search_patient_record: 查询患者病历
- search_medical_knowledge: 搜索医学知识库
- calculate_drug_dosage: 计算用药剂量
- get_surgery_risk: 评估手术风险
根据问题自己判断需要调用哪些工具，综合信息后给出专业回答。"""
)

# ============ 5. 对话主循环 ============
print("\n" + "=" * 50)
print("  脊柱外科 AI Agent 已启动（支持多轮记忆）")
print("  输入 quit 退出")
print("=" * 50)

# 每个会话用固定 thread_id，Agent 自动维护历史
config = {"configurable": {"thread_id": "doctor_session_001"}}

while True:
    user_input = input("\n医生: ").strip()
    if user_input.lower() in ["quit", "exit", "退出"]:
        print("助手: 再见！")
        break
    if not user_input:
        continue

    print("助手: ", end="", flush=True)

    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config  # 通过 thread_id 自动关联历史
        )
        answer = result["messages"][-1].content
        print(answer)

    except Exception as e:
        print(f"\n[错误] {e}")