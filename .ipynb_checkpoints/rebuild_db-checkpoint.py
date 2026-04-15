from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import shutil, os

print("重建病历知识库...")
shutil.rmtree("/root/autodl-tmp/medical-ai/chroma_db", ignore_errors=True)

loader = DirectoryLoader(
    "/root/autodl-tmp/medical-ai/docs",
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)
docs = loader.load()
chunks = RecursiveCharacterTextSplitter(
    chunk_size=300, chunk_overlap=50
).split_documents(docs)

embeddings = HuggingFaceEmbeddings(
    model_name="/root/autodl-tmp/models/AI-ModelScope/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

Chroma.from_documents(
    chunks, embeddings,
    persist_directory="/root/autodl-tmp/medical-ai/chroma_db"
)

print(f"✅ 完成！共 {len(docs)} 份病历，{len(chunks)} 个片段")
for doc in docs:
    print(f"   - {os.path.basename(doc.metadata['source'])}")
