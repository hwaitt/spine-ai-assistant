import gradio as gr
import os
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
import dashscope
import base64
from dashscope import MultiModalConversation

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")

# ============ 1. 初始化模型和知识库 ============
print("初始化中...")

llm = ChatTongyi(
    dashscope_api_key=DASHSCOPE_API_KEY,
    model="qwen-plus",
    temperature=0.7,
)

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

# ============ 2. 定义工具 ============
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
    return records.get(patient_name, "未找到该患者病历")

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

# ============ 3. 创建 Agent ============
tools = [
    search_patient_record,
    search_medical_knowledge,
    calculate_drug_dosage,
    get_surgery_risk
]

memory = MemorySaver()

agent = create_react_agent(
    model=llm,
    tools=tools,
    checkpointer=memory,
    prompt="""你是脊柱外科AI助手，拥有以下工具：
- search_patient_record: 查询患者病历
- search_medical_knowledge: 搜索医学知识库
- calculate_drug_dosage: 计算用药剂量
- get_surgery_risk: 评估手术风险
根据问题自己判断需要调用哪些工具，综合信息后给出专业回答。"""
)

print("Agent 初始化完成！")

# ============ 4. 图像分析函数 ============
def analyze_image(image_path):
    if image_path is None:
        return "请先上传图片"
    try:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()
        ext = image_path.split(".")[-1].lower()
        mime = "image/jpeg" if ext in ["jpg", "jpeg"] else "image/png"
        response = MultiModalConversation.call(
            api_key=DASHSCOPE_API_KEY,
            model="qwen-vl-plus",
            messages=[{
                "role": "user",
                "content": [
                    {"image": f"data:{mime};base64,{image_data}"},
                    {"text": "这是脊柱内窥镜手术图像，请简洁描述图中组织结构，重点关注组织类型、颜色形态、是否有黄韧带或硬膜外脂肪。"}
                ]
            }]
        )
        return response.output.choices[0].message.content[0]["text"]
    except Exception as e:
        return f"图像分析失败: {e}"

# ============ 5. Agent 对话函数 ============
def agent_chat(question, image_desc, patient_name, history, session_id):
    if not question.strip():
        return history, history, ""

    # 把图像描述和患者信息拼入问题
    full_question = question
    if patient_name:
        full_question = f"【当前患者：{patient_name}】{question}"
    if image_desc and image_desc.strip():
        full_question += f"\n【内窥镜图像描述】{image_desc}"

    config = {"configurable": {"thread_id": session_id}}

    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": full_question}]},
            config=config
        )
        answer = result["messages"][-1].content
    except Exception as e:
        answer = f"Agent 错误: {e}"

    history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer}
    ]
    return history, history, ""

def switch_patient():
    return [], []

# ============ 6. Gradio 界面 ============
PATIENTS = ["张某某", "李某某", "王某某"]

with gr.Blocks(title="脊柱外科AI Agent系统") as demo:
    gr.Markdown("# 🏥 脊柱外科 AI Agent 系统")
    gr.Markdown("Agent 自动判断：查病历 / 搜知识库 / 算剂量 / 评估风险")

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Agent 对话", height=450)
            question = gr.Textbox(
                label="医生提问",
                placeholder="例如：张某某术前需要注意什么？体重65kg用克林霉素剂量多少？",
                lines=2
            )
            with gr.Row():
                submit = gr.Button("发送", variant="primary")
                clear = gr.Button("清除对话")

        with gr.Column(scale=1):
            gr.Markdown("### 👤 当前患者")
            patient_name = gr.Dropdown(
                choices=PATIENTS,
                value="张某某",
                label="选择患者"
            )

            gr.Markdown("### 📷 内窥镜图像")
            image_input = gr.Image(
                label="上传手术图像",
                type="filepath"
            )
            analyze_btn = gr.Button("🔍 AI分析图像", variant="secondary")
            image_desc = gr.Textbox(
                label="图像描述（自动填入）",
                placeholder="上传图片后点击分析自动填入",
                lines=3
            )

            gr.Markdown("### 🔧 Agent 工具箱")
            gr.Markdown("""
- 🗂️ 查询患者病历
- 📚 搜索医学知识库
- 💊 计算用药剂量
- ⚠️ 评估手术风险

**Agent 自动决定调用哪些工具**
            """)

    session_id = gr.State("session_001")
    history_state = gr.State([])

    analyze_btn.click(
        analyze_image,
        inputs=[image_input],
        outputs=[image_desc]
    )
    submit.click(
        agent_chat,
        inputs=[question, image_desc, patient_name, history_state, session_id],
        outputs=[chatbot, history_state, question]
    )
    question.submit(
        agent_chat,
        inputs=[question, image_desc, patient_name, history_state, session_id],
        outputs=[chatbot, history_state, question]
    )
    clear.click(lambda: ([], []), outputs=[chatbot, history_state])
    patient_name.change(
        lambda: ([], []),
        outputs=[chatbot, history_state]
    )

demo.launch(server_name="0.0.0.0", server_port=6006, share=False)