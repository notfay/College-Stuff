import json
import pandas as pd
import time
from pathlib import Path
from google import genai
import re
from dotenv import load_dotenv

load_dotenv()

client = genai.Client()


# === CONFIG ===
INPUT_PATH = Path("New_N1N3.json")
CSV_OUT = Path("Results.csv")
XLSX_OUT = Path("Test1_results_parsed.xlsx")

# === Your LLM API call placeholder ===
def review_with_llm(needle: str, question: str, answer: str) -> str:
    """
    Replace this function with your actual API call.
    It receives Needle, Question, and Answer, and should return a review string.
    """
    # Example payload if you’re using a REST API:
    prompt = f"You are a answer checker in which you check if the answer of the question is correct given the needle. Needle: {needle} Question: {question} Answer: {answer}. You are to answer strictly Yes or No and a score 1-10 on how accurate it is to the needle. (example: Yes 6)"

    # --- Example structure (commented out) ---
    # response = requests.post("https://api.yourllm.com/v1/review", json=payload)
    # review_text = response.json().get("review", "")
    # return review_text
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    # For now, just simulate it:
    return response.text

# === Helper to parse LLM output ===
def parse_review(review_text: str):
    """
    Parse something like:
      'Yes 7', 'No 0', 'YES:10', 'no - 3', etc.
    into ('Yes', 7)
    """
    if not isinstance(review_text, str):
        return None, None

    # Try to find yes/no
    match_resp = re.search(r"\b(yes|no)\b", review_text, re.IGNORECASE)
    response = match_resp.group(1).capitalize() if match_resp else None

    # Try to find numeric score
    match_score = re.search(r"(-?\d+(?:\.\d+)?)", review_text)
    score = float(match_score.group(1)) if match_score else None

    return response, score

# === MAIN SCRIPT ===
def main():
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for idx, item in enumerate(data, start=1):
        cfg = item.get("config", {})
        needle = cfg.get("needle", {})
        question = cfg.get("question", {})

        pos = needle.get("custom_position_percent", None)
        haystack_sim = needle.get("haystack_similarity", None)
        q_sim = question.get("question_similarity", None)
        needle_text = needle.get("text", "")
        question_text = question.get("text", "")
        answer_text = item.get("response", "")
        latency = item.get("latency_seconds", item.get("latency", None))

        # === Send to your LLM review API ===
        try:
            review = review_with_llm(needle_text, question_text, answer_text)
        except Exception as e:
            review = f"ERROR: {e}"

        # Parse into response/score
        resp, score = parse_review(review)

        rows.append({
            "Needle Position": pos,
            "Needle-Haystack Similarity": haystack_sim,
            "Needle Question Similarity": q_sim,
            "Needle": needle_text,
            "Question": question_text,
            "Answer": answer_text,
            "Latency": latency,
            "Review": resp,
            "Score": score,
        })

        
        print(f"[{idx}/{len(data)}] → Review: {review} → waiting 6s...")
        time.sleep(6)

    # === Save results ===
    df = pd.DataFrame(rows, columns=[
        "Needle Position",
        "Needle-Haystack Similarity",
        "Needle Question Similarity",
        "Needle",
        "Question",
        "Answer",
        "Latency",
        "Review",
        "Score",
    ])

    df.to_csv(CSV_OUT, index=False, encoding="utf-8", quoting=1)
    df.to_excel(XLSX_OUT, index=False)

    print(f"\n✅ Done! Saved:\n- {CSV_OUT}\n- {XLSX_OUT}")


if __name__ == "__main__":
    main()
