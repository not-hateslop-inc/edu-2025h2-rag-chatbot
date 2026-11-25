import sys
from pathlib import Path
import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()
ROOT = Path(__file__).resolve().parent.parent
col = chromadb.PersistentClient(str(ROOT / "chroma_db")).get_collection("policies")

q = sys.argv[1] if len(sys.argv) > 1 else "재택 근무는 일주일에 몇 번?"
e = client.embeddings.create(model="text-embedding-3-small", input=[q]).data[0].embedding
r = col.query(query_embeddings=[e], n_results=3)
for i, d, doc in zip(r["ids"][0], r["distances"][0], r["documents"][0]):
    print(f"{i:<20} {d:.3f}  {doc[:50]!r}")
