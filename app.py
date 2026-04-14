import gradio as gr
import requests
import base64
import os
from dashscope import MultiModalConversation

API_URL = "http://localhost:8001/chat"
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")

PATIENTS = ["张某某", "李某某", "王某某"]

def analyze_image(image_path):
    """上传图片 → Qwen-VL 分析 → 返回描述"""
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
                    {"text": "这是一张脊柱内窥镜手术图像，请简洁描述图中组织结构，重点关注：组织类型、颜色形态、是否有黄韧带或硬膜外脂肪、组织边界是否清晰。"}
                ]
            }]
        )
        return response.output.choices[0].message.content[0]["text"]
    except Exception as e:
        return f"图像分析失败: {e}"

def chat(question, cv_result, patient_name, history):
    if not question.strip():
        return history, history, ""

    session_id = f"patient_{patient_name}"
    payload = {
        "question": question,
        "cv_result": cv_result,
        "session_id": session_id
    }
    try:
        resp = requests.post(API_URL, json=payload, timeout=60)
        answer = resp.json()["answer"]
    except Exception as e:
        answer = f"请求失败: {e}"

    history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer}
    ]
    return history, history, ""

def switch_patient(patient_name):
    return [], []

with gr.Blocks(title="脊柱外科AI辅助系统") as demo:
    gr.Markdown("# 🏥 脊柱外科 AI 辅助系统")
    gr.Markdown("基于患者病历 + 内窥镜视觉检测的智能诊断助手")

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="对话记录", height=400)
            question = gr.Textbox(
                label="医生提问",
                placeholder="例如：这个患者麻醉要注意什么？",
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
                label="选择患者",
                info="切换患者自动切换病历"
            )

            gr.Markdown("### 📷 内窥镜图像分析")
            image_input = gr.Image(
                label="上传手术图像",
                type="filepath"
            )
            analyze_btn = gr.Button("🔍 AI分析图像", variant="secondary")
            cv_result = gr.Textbox(
                label="视觉检测结果（可手动修改）",
                placeholder="上传图片后自动填入，也可手动输入",
                lines=4
            )

            gr.Markdown("### 📋 使用说明")
            gr.Markdown("""
1. 选择患者
2. 上传内窥镜图像，点击分析
3. 确认检测结果后提问
4. 系统自动结合病历回答
            """)

    history_state = gr.State([])

    analyze_btn.click(
        analyze_image,
        inputs=[image_input],
        outputs=[cv_result]
    )
    submit.click(
        chat,
        inputs=[question, cv_result, patient_name, history_state],
        outputs=[chatbot, history_state, question]
    )
    question.submit(
        chat,
        inputs=[question, cv_result, patient_name, history_state],
        outputs=[chatbot, history_state, question]
    )
    clear.click(lambda: ([], []), outputs=[chatbot, history_state])
    patient_name.change(
        switch_patient,
        inputs=[patient_name],
        outputs=[chatbot, history_state]
    )

demo.launch(server_name="0.0.0.0", server_port=6006, share=False)