"""
Automatically generate questions for empty fields in needles_and_questions.csv
Uses Gemini to generate questions and MiniLM embeddings for similarity checking
"""

import pandas as pd
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer, util
import os
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize models
model = SentenceTransformer("all-MiniLM-L6-v2")

# Configure Gemini API - get API key from environment
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in .env file or system environment.")

client = genai.Client(api_key=api_key)

# Rate limiting configuration
RATE_LIMIT_INTERVAL = 63  # seconds between API calls

def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate cosine similarity between two texts using Sentence Transformer embeddings."""
    try:
        embedding1 = model.encode(text1, convert_to_tensor=True)
        embedding2 = model.encode(text2, convert_to_tensor=True)
        similarity = util.cos_sim(embedding1, embedding2).item()
        return similarity
    except:
        return 0.0

def generate_question_with_gemini(needle_text: str, target_similarity: float) -> str:
    """
    Generate a question using Gemini that targets a specific similarity score.
    Keeps retrying until similarity is within 0.02 of target.
    """
    
    # Create a prompt based on target similarity
    if target_similarity == 0.4:
        similarity_instruction = """
        Generate a question with MODERATE similarity (target: 0.4).
        - Use some key terms from the needle but rephrase significantly
        - Ask about related concepts mentioned in the needle
        - Don't use the exact phrasing from the needle
        """
    elif target_similarity == 0.6:
        similarity_instruction = """
        Generate a question with HIGH similarity (target: 0.6).
        - Use many key terms from the needle
        - Ask directly about the main concept in the needle
        - Keep similar structure but still ask a question
        """
    elif target_similarity == 0.8:
        similarity_instruction = """
        Generate a question with VERY HIGH similarity (target: 0.8).
        - Use almost all key terms and phrases from the needle
        - Very closely mirror the needle's structure and vocabulary
        - Only change it enough to make it a question
        """
    else:
        similarity_instruction = "Generate an appropriate question."
    
    prompt = f"""You are generating questions based on a given statement (needle).

{similarity_instruction}

IMPORTANT RULES:
- Generate ONLY the question text, nothing else
- Do NOT include any explanation, labels, or extra text
- Do NOT use yes/no questions
- The question should be a single sentence
- Keep it natural and conversational
- Keep the question MEDIUM or SHORT length (10-20 words maximum)
- Do NOT make questions too long or overly complex

Needle text: {needle_text}

Generate the question:"""

    attempt = 0
    while True:
        attempt += 1
        try:
            # Generate question with Gemini
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            question = response.text.strip()
            
            # Remove any quotes if present
            question = question.strip('"').strip("'")
            
            # Calculate similarity
            actual_similarity = calculate_similarity(needle_text, question)
            difference = abs(actual_similarity - target_similarity)
            raw_difference = actual_similarity - target_similarity
            
            print(f"  Attempt {attempt}: Generated question with similarity {actual_similarity:.4f} (target: {target_similarity}, diff: {difference:.4f})")
            
            # Check if within tolerance
            if difference <= 0.02:
                print(f"  ✓ Success! Question meets criteria.")
                return question
            
            # If not within tolerance, adjust the prompt for next attempt with detailed feedback
            if raw_difference < 0:  # Too different
                if difference > 0.2:  # WAY TOO DIFFERENT
                    feedback = f"\n\n⚠️  Previous attempt was WAY TOO DIFFERENT (got {actual_similarity:.4f}, expected {target_similarity})."
                    feedback += f"\nYou need to use MUCH MORE words and phrases directly from the needle text."
                    feedback += f"\nInclude more specific terms, concepts, and vocabulary from: '{needle_text}'"
                elif difference > 0.1:  # Very different
                    feedback = f"\n\nPrevious attempt was TOO DIFFERENT (got {actual_similarity:.4f}, expected {target_similarity})."
                    feedback += f"\nUse MORE key words and phrases from the needle."
                else:  # Slightly different
                    feedback = f"\n\nPrevious attempt was slightly too different (got {actual_similarity:.4f}, expected {target_similarity})."
                    feedback += f"\nAdd a few more specific terms from the needle."
            else:  # Too similar
                if difference > 0.2:  # WAY TOO SIMILAR
                    feedback = f"\n\n⚠️  Previous attempt was WAY TOO SIMILAR (got {actual_similarity:.4f}, expected {target_similarity})."
                    feedback += f"\nYou need to use MUCH MORE indirect language and completely different vocabulary."
                    feedback += f"\nRephrase using synonyms and ask about the concept in a more abstract way."
                elif difference > 0.1:  # Very similar
                    feedback = f"\n\nPrevious attempt was TOO SIMILAR (got {actual_similarity:.4f}, expected {target_similarity})."
                    feedback += f"\nUse MORE indirect phrasing and different vocabulary."
                else:  # Slightly similar
                    feedback = f"\n\nPrevious attempt was slightly too similar (got {actual_similarity:.4f}, expected {target_similarity})."
                    feedback += f"\nMake the question a bit more indirect."
            
            prompt += feedback
            
            # Wait for rate limiting before retry
            print(f"  ⏳ Waiting {RATE_LIMIT_INTERVAL} seconds for rate limiting...")
            time.sleep(RATE_LIMIT_INTERVAL)
            
        except Exception as e:
            print(f"  Error on attempt {attempt}: {e}")
            print(f"  ⏳ Waiting {RATE_LIMIT_INTERVAL} seconds before retry...")
            time.sleep(RATE_LIMIT_INTERVAL)

def process_csv():
    """Process the CSV file and generate questions for empty fields."""
    
    csv_file = "needles_and_questions.csv"
    
    # Read CSV
    df = pd.read_csv(csv_file)
    
    print("=" * 100)
    print("AUTO-GENERATING QUESTIONS FOR EMPTY FIELDS")
    print("=" * 100)
    print(f"Target similarity tolerance: ±0.02")
    print(f"Generating for similarity levels: 0.4, 0.6, 0.8 only (skipping 0.2)")
    print()
    
    # Track changes
    rows_updated = 0
    
    # Process each row
    for idx, row in df.iterrows():
        needle_text = row['needle_text']
        question_text = row['question_text']
        target_similarity = row['question_similarity']
        
        # Skip if question already exists
        if pd.notna(question_text) and str(question_text).strip() != "":
            continue
        
        # Skip 0.2 similarity level as requested
        if target_similarity == 0.2:
            print(f"Row {idx + 1}: Skipping (similarity = 0.2)")
            continue
        
        # Only generate for 0.4, 0.6, 0.8
        if target_similarity not in [0.4, 0.6, 0.8]:
            continue
        
        print(f"\nRow {idx + 1}: Generating question for similarity {target_similarity}")
        print(f"Needle: {needle_text[:80]}...")
        
        # Generate question
        generated_question = generate_question_with_gemini(needle_text, target_similarity)
        
        # Update dataframe
        df.at[idx, 'question_text'] = generated_question
        rows_updated += 1
        
        # Save after each successful generation (in case of interruption)
        df.to_csv(csv_file, index=False)
        print(f"  💾 Saved to CSV")
    
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"Total rows updated: {rows_updated}")
    print(f"CSV file saved: {csv_file}")
    
    # Run validation
    print("\n" + "=" * 100)
    print("VALIDATING ALL QUESTIONS")
    print("=" * 100)
    
    validation_issues = 0
    for idx, row in df.iterrows():
        needle_text = row['needle_text']
        question_text = row['question_text']
        target_similarity = row['question_similarity']
        
        if pd.isna(question_text) or str(question_text).strip() == "":
            continue
        
        actual_similarity = calculate_similarity(needle_text, question_text)
        difference = abs(actual_similarity - target_similarity)
        
        status = "✓ GOOD" if difference <= 0.02 else "✗ OFF"
        
        if difference > 0.02:
            validation_issues += 1
            print(f"Row {idx + 1}: {status} | Target: {target_similarity:.2f} | Actual: {actual_similarity:.4f} | Diff: {difference:+.4f}")
    
    if validation_issues == 0:
        print("✓ All questions are within the 0.02 tolerance!")
    else:
        print(f"\n⚠ {validation_issues} questions still need adjustment.")

if __name__ == "__main__":
    # Check if API key is set
    if not os.environ.get('GEMINI_API_KEY'):
        print("ERROR: GEMINI_API_KEY environment variable not set.")
        print("Please set it with: set GEMINI_API_KEY=your_api_key_here")
        exit(1)
    
    process_csv()
