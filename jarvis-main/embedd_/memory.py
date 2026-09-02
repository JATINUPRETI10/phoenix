import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class MemoryStore:

    def __init__(self):
        self.memories = []
        self.embeddings = []

    def create_embedding(self, text):
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )

        return response.data[0].embedding

    def add_memory(self, text):
        embedding = self.create_embedding(text)

        self.memories.append(text)
        self.embeddings.append(embedding)

        print(f"Memory added: {text}")

    def similarity(self, vector1, vector2):
        vector1 = np.array(vector1)
        vector2 = np.array(vector2)

        return np.dot(vector1, vector2) / (
            np.linalg.norm(vector1) * np.linalg.norm(vector2)
        )

    def search(self, query, top_k=3):

        if not self.memories:
            return []

        query_embedding = self.create_embedding(query)

        results = []

        for memory, embedding in zip(
            self.memories,
            self.embeddings
        ):
            score = self.similarity(
                query_embedding,
                embedding
            )

            results.append(
                {
                    "memory": memory,
                    "score": score
                }
            )

        results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return results[:top_k]