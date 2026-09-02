from memory import MemoryStore
from rag import RAG


def main():

    # --------------------------------
    # 1. Create memory store
    # --------------------------------

    memory = MemoryStore()


    # --------------------------------
    # 2. Add knowledge/memories
    # --------------------------------

    memory.add_memory(
        "Python is a high-level programming language "
        "used for web development, AI, automation and data science."
    )

    memory.add_memory(
        "FastAPI is a Python framework used to build "
        "modern and high-performance APIs."
    )

    memory.add_memory(
        "RAG stands for Retrieval-Augmented Generation. "
        "It retrieves relevant information and provides it "
        "to an LLM before generating an answer."
    )

    memory.add_memory(
        "Embeddings convert text into numerical vectors "
        "that capture semantic meaning."
    )

    memory.add_memory(
        "Cosine similarity measures how similar two vectors "
        "are based on the angle between them."
    )

    memory.add_memory(
        "Vector databases are commonly used to store "
        "and search embedding vectors efficiently."
    )


    # --------------------------------
    # 3. Create RAG system
    # --------------------------------

    rag = RAG(memory)


    # --------------------------------
    # 4. Ask question
    # --------------------------------

    query = input("\nAsk a question: ")

    result = rag.ask(
        query,
        top_k=3
    )


    # --------------------------------
    # 5. Show retrieved memories
    # --------------------------------

    print("\n========== RETRIEVED ==========\n")

    for item in result["retrieved"]:

        print(
            f"Score: {item['score']:.4f}"
        )

        print(
            f"Memory: {item['memory']}\n"
        )


    # --------------------------------
    # 6. Show injected context
    # --------------------------------

    print("\n========== CONTEXT INJECTED ==========\n")

    print(result["context"])


    # --------------------------------
    # 7. Show final answer
    # --------------------------------

    print("\n========== FINAL ANSWER ==========\n")

    print(result["answer"])


if __name__ == "__main__":
    main()