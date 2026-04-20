from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

PAPERS_DIR = "/root/autodl-tmp/medical-ai/papers"
CHROMA_PAPERS_DIR = "/root/autodl-tmp/medical-ai/chroma_papers"
EMBEDDING_MODEL = "/root/autodl-tmp/models/AI-ModelScope/bge-small-zh-v1.5"

print("加载论文...")
docs = []
for filename in os.listdir(PAPERS_DIR):
    if filename.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(PAPERS_DIR, filename))
        pages = loader.load()
        docs.extend(pages)
        print(f"  已加载: {filename} ({len(pages)}页)")

print(f"\n共加载 {len(docs)} 页")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks = splitter.split_documents(docs)
print(f"切分为 {len(chunks)} 个片段")

print("\n向量化中...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

vectorstore = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory=CHROMA_PAPERS_DIR
)
print(f"✅ 论文知识库构建完成！共 {len(chunks)} 个片段")