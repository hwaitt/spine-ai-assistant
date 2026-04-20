import gradio as gr
import os
import re
import base64
from typing import Annotated
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict
from dashscope import MultiModalConversation
import uuid

def reset_session(patient_name):
    history = patient_histories.get(patient_name, [])
    return history, str(uuid.uuid4())
patient_histories = {}

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
DOCS_DIR = "/root/autodl-tmp/medical-ai/docs"
CHROMA_DIR = "/root/autodl-tmp/medical-ai/chroma_db"
CHROMA_PAPERS_DIR = "/root/autodl-tmp/medical-ai/chroma_papers"
EMBEDDING_MODEL = "/root/autodl-tmp/models/AI-ModelScope/bge-small-zh-v1.5"

print("初始化中...")

# ============ 1. 初始化 LLM ============
llm = ChatOpenAI(
    api_key="not-needed",
    base_url="http://localhost:8000/v1",
    model="/root/autodl-tmp/spine_qwen25_merged",
    temperature=0.7,
    max_tokens=512,
    model_kwargs={"extra_body": {"repetition_penalty": 1.15}}
)

# ============ 2. 初始化两个知识库 ============
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# 病历库
record_store = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)
record_retriever = record_store.as_retriever(search_kwargs={"k": 3})

# 论文库
paper_store = Chroma(
    persist_directory=CHROMA_PAPERS_DIR,
    embedding_function=embeddings
)
paper_retriever = paper_store.as_retriever(search_kwargs={"k": 3})

# ============ 3. 病历 Agent 的工具 ============
@tool
def search_patient_record(patient_name: str) -> str:
    """根据患者姓名查询病历，包括既往史、过敏史、MRI报告、手术风险"""
    docs = record_retriever.invoke(patient_name)
    if not docs:
        return f"未找到{patient_name}的病历"
    return "\n\n".join([d.page_content for d in docs])

@tool
def get_surgery_risk(patient_name: str) -> str:
    """评估患者手术风险，从病历中提取风险相关信息"""
    docs = record_retriever.invoke(f"{patient_name} 手术风险 麻醉")
    if not docs:
        return f"未找到{patient_name}的风险评估"
    return "\n\n".join([d.page_content for d in docs])

@tool
def calculate_drug_dosage(drug: str, weight_kg: float) -> str:
    """根据药物名称和患者体重计算用药剂量"""
    dosages = {
        "布洛芬": f"{weight_kg * 5:.0f}mg，每日3次，饭后服用",
        "氨氯地平": "5-10mg，每日1次",
        "地塞米松": f"{weight_kg * 0.1:.1f}mg，遵医嘱",
        "克林霉素": f"{weight_kg * 8:.0f}mg，每日3次",
        "万古霉素": f"{weight_kg * 15:.0f}mg，每日2次，静脉滴注",
    }
    return dosages.get(drug, f"暂无{drug}剂量规则，请查阅药典")

# ============ 4. 论文 Agent 的工具 ============
@tool
def search_medical_literature(query: str) -> str:
    """搜索医学论文知识库，获取最新研究证据和手术指南"""
    docs = paper_retriever.invoke(query)
    if not docs:
        return "论文库中未找到相关内容"
    results = []
    for doc in docs:
        source = os.path.basename(doc.metadata.get("source", "未知来源"))
        results.append(f"【来源：{source}】\n{doc.page_content}")
    return "\n\n".join(results)

# ============ 5. 创建子 Agent ============
record_agent = create_react_agent(
    model=llm,
    tools=[search_patient_record, get_surgery_risk, calculate_drug_dosage],
    prompt="""你是病历查询专家,拥有以下工具：
- search_patient_record: 查询患者病历
- calculate_drug_dosage: 计算用药剂量
- get_surgery_risk: 评估手术风险
根据问题自己判断需要调用哪些工具，综合信息后给出专业回答。回答简洁专业，不要重复。"""
)

paper_agent = create_react_agent(
    model=llm,
    tools=[search_medical_literature],
    prompt="你是医学文献专家。专门负责从论文知识库中检索最新研究证据和手术指南。回答简洁准确，注明文献来源。"
)

# ============ 6. 定义多 Agent 状态 ============
class MultiAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    record_result: str
    paper_result: str
    final_answer: str
    need_record: bool
    need_paper: bool

# ============ 7. 定义各节点函数 ============
def record_agent_node(state: MultiAgentState):
    """病历 Agent 节点"""
    question = state["question"]
    result = record_agent.invoke({
        "messages": [HumanMessage(content=question)]
    })
    answer = result["messages"][-1].content
    return {"record_result": answer}

def paper_agent_node(state: MultiAgentState):
    """论文 Agent 节点"""
    question = state["question"]
    result = paper_agent.invoke({
        "messages": [HumanMessage(content=question)]
    })
    answer = result["messages"][-1].content
    return {"paper_result": answer}

def synthesizer_node(state: MultiAgentState):
    """汇总 Agent：整合两个子 Agent 的结果"""
    question = state["question"]
    record_result = state.get("record_result", "无病历信息")
    paper_result = state.get("paper_result", "无论文信息")

    synthesis_prompt = f"""你是脊柱外科AI助手。根据以下信息简洁回答医生问题
要求：3-5句话，直接给结论，不要展开分析，不要重复。

【医生问题】{question}

【病历信息】
{record_result}

【论文证据】
{paper_result}

请给出综合、专业的回答，结合患者具体情况和最新研究证据："""

    response = llm.invoke(synthesis_prompt)
    return {"final_answer": response.content}

# ============ 8. 构建多 Agent 图 ============
def router_node(state: MultiAgentState):
    """路由节点：判断问题需要调用哪些Agent"""
    question = state["question"]
    
    # 判断是否需要论文Agent
    paper_keywords = ["研究", "论文", "文献", "指南", "最新", "证据", "方案", "进展", "综述", "手术技术"]
    need_paper = any(kw in question for kw in paper_keywords)
    
    # 判断是否需要病历Agent
    record_keywords = ["患者", "病历", "过敏", "风险", "麻醉", "剂量", "用药", "既往"]
    need_record = any(kw in question for kw in record_keywords) or "【当前患者" in question
    
    # 默认至少调用病历Agent
    if not need_record and not need_paper:
        need_record = True

    return {
        "need_paper": need_paper,
        "need_record": need_record
    }

def build_graph():
    graph = StateGraph(MultiAgentState)

    graph.add_node("路由", router_node)
    graph.add_node("病历Agent", record_agent_node)
    graph.add_node("论文Agent", paper_agent_node)
    graph.add_node("汇总Agent", synthesizer_node)

    graph.add_edge(START, "路由")

    def route_to_agents(state):
        targets = []
        if state.get("need_record"):
            targets.append("病历Agent")
        if state.get("need_paper"):
            targets.append("论文Agent")
        return targets

    graph.add_conditional_edges("路由", route_to_agents)
    graph.add_edge("病历Agent", "汇总Agent")
    graph.add_edge("论文Agent", "汇总Agent")
    graph.add_edge("汇总Agent", END)

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)

print("构建多 Agent 图...")
multi_agent = build_graph()
print("多 Agent 系统初始化完成！")

# ============ 9. 自动读取患者列表 ============
def get_patient_list():
    patients = []
    for filename in os.listdir(DOCS_DIR):
        if filename.endswith(".txt"):
            with open(os.path.join(DOCS_DIR, filename), "r", encoding="utf-8") as f:
                content = f.read()
            match = re.search(r"患者姓名[：:]\s*(\S+)", content)
            if match:
                patients.append(match.group(1))
    return patients if patients else ["未找到患者"]

PATIENTS = get_patient_list()
print(f"已加载患者：{PATIENTS}")

# ============ 10. 图像分析（Qwen-VL）============
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
                    {"text": "这是脊柱内窥镜手术图像，请简洁描述图中组织结构，重点关注组织类型、颜色形态、是否有黄韧带或硬膜外脂肪、组织边界是否清晰。"}
                ]
            }]
        )
        return response.output.choices[0].message.content[0]["text"]
    except Exception as e:
        return f"图像分析失败: {e}"

# ============ 11. 多 Agent 对话函数 ============
def multi_agent_chat(question, image_desc, patient_name, session_id):
    if not question.strip():
        history = patient_histories.get(patient_name, [])
        return history, ""

    full_question = question
    if patient_name:
        full_question = f"【当前患者：{patient_name}】{question}"
    if image_desc and image_desc.strip():
        full_question += f"\n【内窥镜图像描述】{image_desc}"

    config = {"configurable": {"thread_id": f"{session_id}_{patient_name}"}}

    try:
        result = multi_agent.invoke(
            {
                "question": full_question,
                "messages": [HumanMessage(content=full_question)],
                "record_result": "",
                "paper_result": "",
                "final_answer": "",
                "need_record": False,
                "need_paper": False,
            },
            config=config
        )
        answer = result["final_answer"]
    except Exception as e:
        answer = f"错误: {e}"

    history = patient_histories.get(patient_name, [])
    history = history + [[question, answer]]
    patient_histories[patient_name] = history
    return history, ""

# ============ 12. Gradio 界面 ============
with gr.Blocks(title="脊柱外科多Agent AI系统") as demo:
    gr.Markdown("# 🏥 脊柱外科多 Agent AI 系统")
    gr.Markdown("**病历Agent** + **论文Agent** → **汇总Agent** 协同工作")

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="多Agent对话", height=450)
            question = gr.Textbox(
                label="医生提问",
                placeholder="例如：张某某术前需要注意什么？黄韧带肥厚最新手术方案？",
                lines=2
            )
            with gr.Row():
                submit = gr.Button("发送", variant="primary")
                clear = gr.Button("清除对话")

        with gr.Column(scale=1):
            gr.Markdown("### 👤 当前患者")
            patient_name = gr.Dropdown(
                choices=PATIENTS,
                value=PATIENTS[0],
                label="选择患者"
            )

            gr.Markdown("### 📷 内窥镜图像")
            image_input = gr.Image(label="上传手术图像", type="filepath")
            analyze_btn = gr.Button("🔍 AI分析图像", variant="secondary")
            image_desc = gr.Textbox(
                label="图像描述",
                placeholder="上传图片后点击分析",
                lines=3
            )

            gr.Markdown("### 🤖 Agent 架构")
            gr.Markdown("""
- 🗂️ **病历Agent**：查病历/风险/剂量
- 📚 **论文Agent**：查最新研究证据
- 🧠 **汇总Agent**：综合两者给出建议

**两个子Agent并行运行，汇总Agent整合结果**
            """)

    session_id = gr.State(str(uuid.uuid4()))

    analyze_btn.click(analyze_image, inputs=[image_input], outputs=[image_desc])
    submit.click(
        multi_agent_chat,
        inputs=[question, image_desc, patient_name, session_id],
        outputs=[chatbot, question]
    )
    question.submit(
        multi_agent_chat,
        inputs=[question, image_desc, patient_name, session_id],
        outputs=[chatbot, question]
    )
    clear.click(
        lambda p: (patient_histories.update({p: []}) or []),
        inputs=[patient_name],
        outputs=[chatbot]
    )
    patient_name.change(
        reset_session,
        inputs=[patient_name],
        outputs=[chatbot, session_id]
    )

demo.launch(server_name="0.0.0.0", server_port=6006, share=False)