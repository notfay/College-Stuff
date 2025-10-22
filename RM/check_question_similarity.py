"""
Script to verify that questions match their intended similarity scores with their needles.
Uses TF-IDF cosine similarity to measure actual question-needle similarity.
"""

import csv
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate cosine similarity between two texts using TF-IDF."""
    vectorizer = TfidfVectorizer()
    try:
        vectors = vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
        return similarity
    except:
        return 0.0

def check_question_similarities(csv_file: str):
    """
    Read CSV and calculate actual similarity between each question and its needle.
    Compare with the expected question_similarity value.
    """
    # Read CSV
    df = pd.read_csv(csv_file)
    
    print("=" * 100)
    print("QUESTION-NEEDLE SIMILARITY ANALYSIS")
    print("=" * 100)
    
    results = []
    output_rows = []  # For CSV export matching original format
    
    for idx, row in df.iterrows():
        needle_id = row['needle_id']
        haystack_similarity = row['haystack_similarity']
        needle_text = row['needle_text']
        question_text = row['question_text']
        expected_q_sim = row['question_similarity']
        
        # Calculate actual similarity between question and needle
        actual_q_sim = calculate_similarity(question_text, needle_text)
        
        # Calculate difference
        difference = actual_q_sim - expected_q_sim
        
        # Determine if it's a good match (within ±0.05 tolerance)
        status = "✓ GOOD" if abs(difference) <= 0.05 else "✗ OFF"
        
        results.append({
            'row': idx + 1,
            'needle_id': needle_id,
            'expected_q_sim': expected_q_sim,
            'actual_q_sim': actual_q_sim,
            'difference': difference,
            'status': status
        })
        
        # Store row for CSV export with same format as input + actual_q_sim column
        output_rows.append({
            'needle_id': needle_id,
            'haystack_similarity': haystack_similarity,
            'question_similarity': expected_q_sim,
            'needle_text': needle_text,
            'question_text': question_text,
            'actual_question_similarity': round(actual_q_sim, 4)
        })
        
        print(f"\nRow {idx + 1} | Needle ID: {needle_id}")
        print(f"  Expected Q-Sim: {expected_q_sim:.2f}")
        print(f"  Actual Q-Sim:   {actual_q_sim:.4f}")
        print(f"  Difference:     {difference:+.4f}")
        print(f"  Status:         {status}")
        print(f"  Question: {question_text[:80]}...")
    
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    
    # Group by needle_id and show progression
    for needle_id in df['needle_id'].unique():
        print(f"\n📌 Needle {needle_id}:")
        needle_results = [r for r in results if r['needle_id'] == needle_id]
        
        for r in needle_results:
            print(f"  Row {r['row']:2d} | Expected: {r['expected_q_sim']:.1f} | "
                  f"Actual: {r['actual_q_sim']:.4f} | Diff: {r['difference']:+.4f} | {r['status']}")
    
    # Overall statistics
    print("\n" + "=" * 100)
    print("OVERALL STATISTICS")
    print("=" * 100)
    
    differences = [r['difference'] for r in results]
    good_matches = sum(1 for r in results if '✓' in r['status'])
    
    print(f"Total Questions:     {len(results)}")
    print(f"Good Matches:        {good_matches}/{len(results)} ({good_matches/len(results)*100:.1f}%)")
    print(f"Mean Difference:     {np.mean(differences):+.4f}")
    print(f"Std Dev:             {np.std(differences):.4f}")
    print(f"Max Over-estimate:   {max(differences):+.4f}")
    print(f"Max Under-estimate:  {min(differences):+.4f}")
    
    print("\n" + "=" * 100)
    print("RECOMMENDATIONS")
    print("=" * 100)
    
    for r in results:
        if '✗' in r['status']:
            if r['difference'] > 0:
                print(f"⚠️  Row {r['row']}: Question is TOO SIMILAR to needle "
                      f"(expected {r['expected_q_sim']:.1f}, got {r['actual_q_sim']:.4f})")
                print(f"    → Make question more indirect, use different vocabulary")
            else:
                print(f"⚠️  Row {r['row']}: Question is TOO DIFFERENT from needle "
                      f"(expected {r['expected_q_sim']:.1f}, got {r['actual_q_sim']:.4f})")
                print(f"    → Make question more similar, use more words from needle")
    
    if good_matches == len(results):
        print("✓ All questions match their intended similarity levels!")
    
    # Export to CSV with same format as input + actual_question_similarity column
    output_df = pd.DataFrame(output_rows)
    output_csv = "needles_and_questions_with_actual_similarity.csv"
    output_df.to_csv(output_csv, index=False)
    print(f"\n✓ CSV with actual similarity saved to: {output_csv}")
    
    return output_df

if __name__ == "__main__":
    CSV_FILE = "needles_and_questions.csv"
    
    print(f"File: {CSV_FILE}\n")
    output_df = check_question_similarities(CSV_FILE)
