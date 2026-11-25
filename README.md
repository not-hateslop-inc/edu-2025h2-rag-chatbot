# edu-2025h2-rag-chatbot

2025 하반기 6·7회차 **RAG** 실습 저장소입니다.

- EDU-2025-H2-06 RAG 기초: Embedding과 Vector DB (2025-10-16)
- EDU-2025-H2-07 RAG 실습: 사내 문서 Q&A 챗봇 (2025-10-23)

> 이 저장소의 예전 이름은 `edu-2025-h2-rag`였습니다. 7회차부터 `edu-2025h2-rag-chatbot`으로 바꿨습니다. 예전 주소로 Clone 했다면 `git remote set-url origin https://github.com/not-hateslop-inc/edu-2025h2-rag-chatbot.git`을 실행하세요.

## 데이터

`data/policies/` 사내 규정 4종(가상). 조항 번호(제N조)가 붙어 있습니다.

## 파일

| 파일 | 회차 | 내용 |
| --- | --- | --- |
| `week6/01_embedding.py` | 6 | 임베딩과 코사인 유사도 |
| `week6/02_index.py` | 6 | Chunk → Chroma 저장 |
| `week6/03_search.py` | 6 | 질문으로 검색 |
| `week7/chatbot.py` | 7 | 검색 + 답변 + 출처 |
| `week7/chunk_experiment.py` | 7 | chunk_size 비교 |
| `week7/questions.json` | 7 | 고정 질문 |

## 실행

```bash
pip install -r requirements.txt
python week6/02_index.py
python week7/chatbot.py
```
