# Needle-Question.py - Calculate Needle-Question Similarity
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from abc import ABC, abstractmethod
import warnings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sentence_transformers")

# --- 1. Define Abstract Model Class ---

class EmbeddingModel(ABC):
    """Abstract base class for all embedding models."""
    def __init__(self, model_name):
        self.model_name = model_name
        print(f"Initializing model: {self.model_name}")

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embeds a list of texts and returns a numpy array."""
        pass

# --- 2. Create Concrete Model Classes ---

class STModel(EmbeddingModel):
    """Wrapper for SentenceTransformers."""
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        super().__init__(model_name)
        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(self.model_name)
        except ImportError:
            print("Please install sentence-transformers: pip install sentence-transformers")
            raise

    def embed(self, texts: list[str]) -> np.ndarray:
        # Sentence Transformers typically normalize by default
        return self.encoder.encode(texts, show_progress_bar=False)


# --- 3. Define Data ---

# List of (Needle, Question, Target_Needle_Question_Similarity) tuplesneedle = "# List of (Needle, Question, Target_Needle_Question_Similarity) tuples
needle = "A study asked people to mark the single feature they first noticed and then measured how that mark changed option choice in later tasks"

pairs_to_test = [
    # Target 0.8 - High overlap (Using needle words + safe words)
    (needle, "How does a first‑seen detail become the reference in later evaluation?", 0.4),
]



# --- 4. Main Execution Function ---

def main():
    """
    Main function to initialize models, run similarity tests,
    and print results.
    """
    
    # List of models to test
    try:
        models_to_test = [
            STModel(model_name="all-MiniLM-L6-v2")
        ]
        print(f"\nInitialized {len(models_to_test)} models for cross-validation.")
    except Exception as e:
        print(f"Error initializing models: {e}")
        print("Please ensure all API keys are set (e.g., GEMINI_API_KEY in .env) and packages are installed.")
        return # Exit main function if models fail

    all_results = []
    print("\n" + "="*60)
    print(f"Calculating Needle-Question Similarities for {len(pairs_to_test)} pairs...")
    print("="*60)

    for model in models_to_test:
        print(f"\n--- Testing with Model: {model.model_name} ---")

        # 1. Separate needles, questions, and targets
        needles = [pair[0] for pair in pairs_to_test]
        questions = [pair[1] for pair in pairs_to_test]
        targets = [pair[2] for pair in pairs_to_test]

        # 2. Embed all at once (batching)
        print(f"Embedding {len(needles)} needles and {len(questions)} questions...")
        needle_embeddings = model.embed(needles)
        question_embeddings = model.embed(questions)

        if needle_embeddings.size == 0 or question_embeddings.size == 0:
            print(f"Error: Embedding failed for model {model.model_name}. Skipping...")
            continue

        # Check for embedding count mismatch
        if len(needle_embeddings) != len(needles) or len(question_embeddings) != len(questions):
            print(f"Warning: Mismatch in embedding count for {model.model_name}.")
            continue

        # 3. Calculate similarity for each pair
        for i, pair in enumerate(pairs_to_test):
            n_emb = needle_embeddings[i:i+1]  # Keep 2D shape (1, D)
            q_emb = question_embeddings[i:i+1] # Keep 2D shape (1, D)

            needle_question_similarity = cosine_similarity(n_emb, q_emb)[0][0]

            all_results.append({
                "pair_id": i,
                "model": model.model_name,
                "target": targets[i],
                "needle_question_sim": needle_question_similarity,
                "needle": f"'{pair[0][:50]}...'",
                "question": f"'{pair[1][:50]}...'"
            })

    # --- 5. Display Individual Results ---
    print("\n" + "="*60)
    print("Individual Results (All Models)")
    print("="*60)

    if not all_results:
        print("No results generated. Check for embedding errors.")
        return

    results_df = pd.DataFrame(all_results)
    print(results_df.to_string(index=False))

    # --- 6. Filter and Display Needle-Question Pairs that Match Target (±0.02) ---
    print("\n" + "="*80)
    print("NEEDLE-QUESTION PAIRS MATCHING TARGET SIMILARITY (±0.02 difference)")
    print("="*80)

    matching_pairs = []
    for idx, row in results_df.iterrows():
        difference = abs(row['needle_question_sim'] - row['target'])
        if difference <= 0.02:
            pair = pairs_to_test[row['pair_id']]
            matching_pairs.append({
                "Pair #": row['pair_id'],
                "Target": row['target'],
                "Actual": round(row['needle_question_sim'], 4),
                "Diff": round(difference, 4),
                "Needle": pair[0],
                "Question": pair[1]
            })
    
    if matching_pairs:
        match_df = pd.DataFrame(matching_pairs)
        print(f"\nFound {len(matching_pairs)} matching needle-question pairs:\n")
        print(match_df.to_string(index=False))
        
        # Summary by target
        print("\n" + "="*80)
        print("SUMMARY BY TARGET SIMILARITY")
        print("="*80)
        for target in [0.2, 0.4, 0.6, 0.8]:
            count = sum(1 for m in matching_pairs if m['Target'] == target)
            print(f"Target {target}: {count} pairs matched")
    else:
        print("❌ No needle-question pairs found that match their target within ±0.02 tolerance.")
        print("\nClosest pairs:")
        results_df['difference'] = abs(results_df['needle_question_sim'] - results_df['target'])
        closest = results_df.nsmallest(5, 'difference')[['pair_id', 'target', 'needle_question_sim', 'difference']]
        print(closest.to_string(index=False))


# --- 5. Standard Python Entry Point ---
if __name__ == "__main__":
    main()