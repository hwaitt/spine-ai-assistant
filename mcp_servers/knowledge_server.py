from mcp.server.fastmcp import FastMCP
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

mcp = FastMCP("知识库服务")

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
print("知识库加载完成")

@mcp.tool()
def search_medical_knowledge(query: str) -> str:
    """搜索脊柱外科医学知识库，包括解剖、手术指南、组织特征"""
    docs = retriever.invoke(query)
    if not docs:
        return "知识库中未找到相关内容"
    return "\n\n".join([d.page_content for d in docs])

@mcp.tool()
def calculate_drug_dosage(drug: str, weight_kg: float) -> str:
    """根据药物和体重计算剂量"""
    dosages = {
        "布洛芬": f"{weight_kg * 5:.0f}mg，每日3次，饭后服用",
        "克林霉素": f"{weight_kg * 8:.0f}mg，每日3次",
        "万古霉素": f"{weight_kg * 15:.0f}mg，每日2次，静脉滴注",
        "氨氯地平": "5-10mg，每日1次",
        "地塞米松": f"{weight_kg * 0.1:.1f}mg，遵医嘱",
    }
    return dosages.get(drug, f"暂无{drug}剂量规则，请查阅药典")

if __name__ == "__main__":
    mcp.run(transport="stdio")