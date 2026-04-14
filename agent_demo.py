from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
import os

# ============ 1. 连接你的 vLLM ============
from langchain_community.chat_models.tongyi import ChatTongyi

llm = ChatTongyi(
    dashscope_api_key=os.environ["DASHSCOPE_API_KEY"],
    model="qwen-plus",
    temperature=0.7,
)

# ============ 2. 定义工具 ============
@tool
def search_patient_record(patient_name: str) -> str:
    """根据患者姓名查询病历信息，包括既往史、过敏史、手术风险等"""
    # 模拟病历数据库
    records = {
        "张某某": """
        年龄：58岁
        主诉：腰痛伴右下肢放射痛3年
        MRI：L4-L5黄韧带肥厚8mm
        既往史：高血压10年
        过敏史：青霉素过敏
        麻醉建议：全身麻醉，避免青霉素
        """,
        "李某某": """
        年龄：45岁
        主诉：颈椎病，上肢麻木
        MRI：C5-C6椎间盘突出
        既往史：糖尿病5年
        过敏史：无
        麻醉建议：全身麻醉，注意血糖监测
        """
    }
    return records.get(patient_name, "未找到该患者病历")

@tool
def calculate_drug_dosage(drug: str, weight_kg: float) -> str:
    """根据药物名称和患者体重计算用药剂量"""
    dosages = {
        "布洛芬": f"推荐剂量：{weight_kg * 5:.0f}mg，每日3次，饭后服用",
        "氨氯地平": f"推荐剂量：5-10mg，每日1次",
        "地塞米松": f"推荐剂量：{weight_kg * 0.1:.1f}mg，遵医嘱"
    }
    return dosages.get(drug, f"暂无{drug}的剂量计算规则，请查阅药典")

@tool
def get_surgery_risk(patient_name: str) -> str:
    """评估患者手术风险等级"""
    risks = {
        "张某某": "中等风险：高血压需术前控制血压，青霉素过敏需备用抗生素方案",
        "李某某": "中等风险：糖尿病需术前控制血糖在8mmol/L以下"
    }
    return risks.get(patient_name, "未找到该患者风险评估")

# ============ 3. 创建 Agent ============
tools = [search_patient_record, calculate_drug_dosage, get_surgery_risk]

agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt="""你是脊柱外科AI助手。你有以下工具：
- search_patient_record: 查询患者病历
- calculate_drug_dosage: 计算用药剂量
- get_surgery_risk: 评估手术风险
根据问题自己判断需要调用哪些工具，然后给出专业回答。"""
)

# ============ 4. 测试 ============
print("=" * 50)
print("Agent 思考过程")
print("=" * 50)

questions = [
    "张某某这个患者手术前需要注意什么？",
    "张某某体重65kg，需要用布洛芬止痛，剂量是多少？",
    "李某某的手术风险怎么样？"
]

for q in questions:
    print(f"\n❓ 问题：{q}")
    print("-" * 40)
    result = agent.invoke({"messages": [{"role": "user", "content": q}]})
    final = result["messages"][-1].content
    print(f"✅ 最终回答：{final}")
    print("=" * 50)