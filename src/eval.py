import json
import os
from src.retrieve import retrieve

EVAL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "eval_dataset.json")


def is_correct_chunk(chunk: dict, item: dict) -> bool:
    same_source = chunk["source"] == item["ground_truth_source"]
    has_keyword = item["ground_truth_keyword"].lower() in chunk["text"].lower()
    return same_source and has_keyword


def evaluate(k: int = 5) -> dict:
    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        eval_items = json.load(f)

    hits = 0
    reciprocal_ranks = []
    per_question_log = []

    for item in eval_items:
        results = retrieve(item["question"], k=k)

        rank_of_first_correct = None
        for rank, chunk in enumerate(results, start=1):
            if is_correct_chunk(chunk, item):
                rank_of_first_correct = rank
                break

        if rank_of_first_correct is not None:
            hits += 1
            reciprocal_ranks.append(1.0 / rank_of_first_correct)
        else:
            reciprocal_ranks.append(0.0)

        per_question_log.append({
            "question": item["question"],
            "rank_of_first_correct": rank_of_first_correct,
        })

    hit_rate = hits / len(eval_items)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

    return {
        "k": k,
        "num_questions": len(eval_items),
        "hit_rate": round(hit_rate, 3),
        "mrr": round(mrr, 3),
        "per_question": per_question_log,
    }


if __name__ == "__main__":
    report = evaluate(k=10)
    print(f"Questions evaluated : {report['num_questions']}")
    print(f"Hit Rate@{report['k']}        : {report['hit_rate']}")
    print(f"MRR               : {report['mrr']}")
    print("\nPer-question detail:")
    for q in report["per_question"]:
        rank = q["rank_of_first_correct"] if q["rank_of_first_correct"] else "MISS"
        print(f"  rank={rank}\t{q['question']}")
