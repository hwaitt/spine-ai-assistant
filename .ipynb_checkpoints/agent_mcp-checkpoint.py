import asyncio
import os
import base64
import gradio as gr
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from dashscope import MultiModalConversation
import re

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
DOCS_DIR = "/root/autodl-tmp/medical-ai/docs"

# ============ 1. 初始化 LLM ============
llm = ChatTongyi(
    dashscope_api_key=DASHSCOPE_API_KEY,
    model="qwen-plus",
    temperature=0.7,
)

# ============ 2. 自动读取患者列表 ============
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

# ============ 3. 图像分析 ============
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

# ============ 4. MCP Agent 对话（异步）============
async def run_agent(question, image_desc, patient_name, session_id):
    # 1. 直接实例化 client，移除 async with
    client = MultiServerMCPClient({
        "病历服务": {
            "command": "python",
            "args": ["/root/autodl-tmp/medical-ai/mcp_servers/patient_server.py"],
            "transport": "stdio"
        },
        "知识库服务": {
            "command": "python",
            "args": ["/root/autodl-tmp/medical-ai/mcp_servers/knowledge_server.py"],
            "transport": "stdio"
        }
    })
    
    # 2. 异步获取工具（注意必须加 await）
    tools = await client.get_tools()
    
    # 3. 正常创建 Agent
    memory = MemorySaver()
    agent = create_react_agent(
        model=llm,
        tools=tools,
        checkpointer=memory,
        prompt="""你是脊柱外科AI助手。
使用可用工具查询患者病历和医学知识，综合信息给出专业回答。
回答简洁专业，不要重复。"""
    )

    full_question = f"【当前患者：{patient_name}】{question}"
    if image_desc and image_desc.strip():
        full_question += f"\n【内窥镜图像描述】{image_desc}"

    config = {"configurable": {"thread_id": session_id}}
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": full_question}]},
        config=config
    )
    return result["messages"][-1].content

def agent_chat(question, image_desc, patient_name, history, session_id):
    if not question.strip():
        return history, history, ""
    try:
        answer = asyncio.run(
            run_agent(question, image_desc, patient_name, session_id)
        )
    except Exception as e:
        answer = f"错误: {e}"

    history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer}
    ]
    return history, history, ""

# ============ 5. Gradio 界面 ============
PATIENTS = get_patient_list()
print(f"已加载患者：{PATIENTS}")

with gr.Blocks(title="脊柱外科AI Agent（MCP版）") as demo:
    gr.Markdown("# 🏥 脊柱外科 AI Agent 系统（MCP版）")
    gr.Markdown("通过 MCP 协议连接病历服务 + 知识库服务")

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Agent 对话", height=450)
            question = gr.Textbox(
                label="医生提问",
                placeholder="例如：张某某术前需要注意什么？",
                lines=2
            )
            with gr.Row():
                submit = gr.Button("发送", variant="primary")
                clear = gr.Button("清除对话")

        with gr.Column(scale=1):
            gr.Markdown("### 👤 当前患者")
            patient_name = gr.Dropdown(
                choices=PATIENTS,
                value=PATIENTS[0] if PATIENTS else None,
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

            gr.Markdown("### 🔌 MCP 服务状态")
            gr.Markdown("""
- 🟢 病历服务（patient_server）
- 🟢 知识库服务（knowledge_server）

**工具通过 MCP 协议动态加载**
            """)

    session_id = gr.State("mcp_session_001")
    history_state = gr.State([])

    # ======= 事件绑定区域 =======
    analyze_btn.click(analyze_image, inputs=[image_input], outputs=[image_desc])
    
    # 【修改的代码】：使用 change 事件。无论是清空图片还是上传新图，都会自动清空描述框
    image_input.change(lambda: "", outputs=[image_desc])

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
    patient_name.change(lambda: ([], []), outputs=[chatbot, history_state])

demo.launch(server_name="0.0.0.0", server_port=6006, share=False)