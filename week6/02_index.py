from pathlib import Path
import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
client = OpenAI()
ROOT = Path(__file__).resolve().parent.parent
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50, separators=["\n## ", "\n\n", "\n", " "])

ids, docs, metas = [], [], []
for p in sorted((ROOT / "data/policies").glob("*.md")):
    for i, chunk in enumerate(splitter.split_text(p.read_text(encoding="utf-8"))):
        ids.append(f"{p.stem}-{i}"); docs.append(chunk); metas.append({"source": p.name})

emb = [d.embedding for d in client.embeddings.create(model="text-embedding-3-small", input=docs).data]
col = chromadb.PersistentClient(str(ROOT / "chroma_db")).get_or_create_collection("policies", metadata={"hnsw:space": "cosine"})
col.upsert(ids=ids, documents=docs, embeddings=emb, metadatas=metas)
print("indexed", len(ids))
