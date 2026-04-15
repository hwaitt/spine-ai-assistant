import os
import re
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("病历服务")

DOCS_DIR = "/root/autodl-tmp/medical-ai/docs"

@mcp.tool()
def search_patient_record(patient_name: str) -> str:
    """根据患者姓名查询病历信息"""
    for filename in os.listdir(DOCS_DIR):
        if not filename.endswith(".txt"):
            continue
        filepath = os.path.join(DOCS_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        if patient_name in content:
            return content
    return f"未找到{patient_name}的病历"

@mcp.tool()
def list_all_patients() -> str:
    """列出所有患者姓名"""
    patients = []
    for filename in os.listdir(DOCS_DIR):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(DOCS_DIR, filename), "r", encoding="utf-8") as f:
            content = f.read()
        match = re.search(r"患者姓名[：:]\s*(\S+)", content)
        if match:
            patients.append(match.group(1))
    return f"当前患者列表：{', '.join(patients)}"

@mcp.tool()
def get_patient_allergy(patient_name: str) -> str:
    """专门查询患者过敏史"""
    for filename in os.listdir(DOCS_DIR):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(DOCS_DIR, filename), "r", encoding="utf-8") as f:
            content = f.read()
        if patient_name in content:
            match = re.search(r"过敏史[：:]\s*(.+)", content)
            if match:
                return f"{patient_name}过敏史：{match.group(1).strip()}"
    return f"未找到{patient_name}的过敏史"

if __name__ == "__main__":
    mcp.run(transport="stdio")