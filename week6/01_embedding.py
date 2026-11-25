import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()
texts = ["연차는 3일 전까지 신청한다", "휴가 신청은 사흘 전에 해야 한다", "야근 식대는 1만 2천 원이다"]
vecs = np.array([d.embedding for d in client.embeddings.create(model="text-embedding-3-small", input=texts).data])
vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
print(np.round(vecs @ vecs.T, 3))
