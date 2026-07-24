"""
Step 11: Hallucination guard
After the generator produces an answer, ask a SEPARATE, narrow Groq call to
judge whether every claim in the answer is actually supported by the given
context. This catches the most common RAG failure: the model quietly answering
from its own pretrained knowledge instead of the retrieved notes.
"""

import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

MODEL_NAME = "llama-3.3-70b-versatile"

GUARD_PROMPT_TEMPLATE = """You are a strict fact-checker. Given a CONTEXT and an ANSWER,
determine if the answer is fully supported by the context.

Respond ONLY with valid JSON, no other text, in this exact format:
{{"grounded": true or false, "unsupported_claims": ["claim1", "claim2"]}}

If everything in the answer is supported by the context, unsupported_claims should be
an empty list.

CONTEXT:
{context}

ANSWER:
{answer}

JSON:"""


def check_grounding(answer: str, retrieved_chunks: list[dict]) -> dict:
    context = "\n\n".join(c["text"] for c in retrieved_chunks)
    prompt = GUARD_PROMPT_TEMPLATE.format(context=context, answer=answer)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # Fail safe: if the judge's output isn't parseable, flag it rather
        # than silently trusting the answer.
        result = {"grounded": False, "unsupported_claims": ["Guard response could not be parsed"]}

    return result


if __name__ == "__main__":
    from src.retrieve import retrieve
    from src.generate import generate_answer

    question = "What is a encapsulation?"
    chunks = retrieve(question, k=3)
    answer = generate_answer(question, chunks)
    verdict = check_grounding(answer, chunks)

    print("Q:", question)
    print("A:", answer)
    print("Guard verdict:", verdict)
