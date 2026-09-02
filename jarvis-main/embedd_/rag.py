from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class RAG:

    def __init__(self, memory_store):
        self.memory_store = memory_store

    def retrieve(self, query, top_k=3):

        results = self.memory_store.search(
            query,
            top_k
        )

        return results

    def build_context(self, results):

        context = ""

        for i, result in enumerate(results):

            context += (
                f"Context {i + 1}:\n"
                f"{result['memory']}\n\n"
            )

        return context

    def generate_answer(self, query, context):

        prompt = f"""
You are a helpful AI assistant.

Answer the user's question using the provided context.

If the answer cannot be found in the context,
say that you don't have enough information.

Context:
{context}

User Question:
{query}

Answer:
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        return response.choices[0].message.content

    def ask(self, query, top_k=3):

        results = self.retrieve(
            query,
            top_k
        )

        context = self.build_context(results)

        answer = self.generate_answer(
            query,
            context
        )

        return {
            "query": query,
            "retrieved": results,
            "context": context,
            "answer": answer
        }