import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

MODEL_NAME = "llama-3.3-70b-versatile"

PROMPT_TEMPLATE = """You are answering a question using ONLY the context below.
If the answer isn't in the context, say "I don't have that in my notes" --
do not use outside knowledge.

Context:
{context}

Question: {question}

Answer:"""


def generate_answer(question: str, retrieved_chunks: list[dict]) -> str:
    context = "\n\n".join(c["text"] for c in retrieved_chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    from src.retrieve import retrieve

    question = "What is a encapsulation?"
    chunks = retrieve(question, k=3)
    answer = generate_answer(question, chunks)
    print("Q:", question)
    print("A:", answer)
