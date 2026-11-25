from pathlib import Path
import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()
ROOT = Path(__file__).resolve().parent.parent
col = chromadb.PersistentClient(str(ROOT / "chroma_db")).get_collection("policies")
NOT_FOUND = "규정에서 찾을 수 없습니다. 운영·기획부에 문의해 주세요."

def answer(q, k=3, max_dist=0.6):
    e = client.embeddings.create(model="text-embedding-3-small", input=[q]).data[0].embedding
    r = col.query(query_embeddings=[e], n_results=k)
    hits = [(i, doc) for i, d, doc in zip(r["ids"][0], r["distances"][0], r["documents"][0]) if d <= max_dist]
    if not hits:
        return NOT_FOUND, []
    ctx = "\n\n".join(f"[{n}] {doc}" for n, (_, doc) in enumerate(hits, 1))
    resp = client.chat.completions.create(model="gpt-4o-mini", temperature=0, messages=[
        {"role": "system", "content": f"사내 규정 안내 도우미다. 참고 문서만 근거로 답하고 문장 끝에 [번호]를 단다. 근거가 없으면 '{NOT_FOUND}'라고만 답한다."},
        {"role": "user", "content": f"참고 문서:\n{ctx}\n\n질문: {q}"},
    ])
    return resp.choices[0].message.content, [i for i, _ in hits]

if __name__ == "__main__":
    while (q := input("질문> ").strip()) not in ("", "q"):
        a, src = answer(q)
        print(a, "\n출처:", src)
