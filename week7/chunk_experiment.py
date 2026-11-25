# chunk_size를 바꿔 가며 week7/questions.json의 기대 문서가 top-3에 들어오는지 확인합니다.
# 사용법: python week7/chunk_experiment.py 150 300 600
import json, sys
from pathlib import Path
import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
client = OpenAI()
ROOT = Path(__file__).resolve().parent.parent
QS = json.loads((ROOT / "week7/questions.json").read_text(encoding="utf-8"))

def embed(t): return [d.embedding for d in client.embeddings.create(model="text-embedding-3-small", input=t).data]

for size in map(int, sys.argv[1:] or ["150", "300", "600"]):
    sp = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=size // 6)
    ids, docs = [], []
    for p in sorted((ROOT / "data/policies").glob("*.md")):
        for i, c in enumerate(sp.split_text(p.read_text(encoding="utf-8"))):
            ids.append(f"{p.stem}-{i}"); docs.append(c)
    col = chromadb.Client().create_collection(f"exp{size}", metadata={"hnsw:space": "cosine"})
    col.add(ids=ids, documents=docs, embeddings=embed(docs))
    hit = 0
    for q in QS:
        got = col.query(query_embeddings=embed([q["question"]]), n_results=3)["ids"][0]
        hit += any(g.startswith(q["expected_source"]) for g in got)
    print(f"chunk_size={size:<4} hit@3={hit}/{len(QS)}")
