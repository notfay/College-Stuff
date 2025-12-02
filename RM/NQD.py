# Needle-Question.py (Refactored)
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from abc import ABC, abstractmethod
import os
import warnings
from dotenv import load_dotenv  # Import dotenv

# Load environment variables from .env file
# This is the correct way to do it.
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

# List of (Needle, Question, Distractor, Target_Needle_Distractor_Similarity) tuples
pairs_to_test = [
    # Needle 1: Memory and sleep
    (
        "Memory consolidation during sleep helps learners retain verbal information better than visual patterns in most cases.",
        "What finding resulted from analyzing operator reliance on algorithmic recommendations concerning error rates in time-critical situations?",
        "Studies of automated decision systems found that adding manager oversight to algorithmic recommendations cut errors by 51% during time-sensitive tasks.",
        0.8  # Target similarity between needle and distractor
    ),
    # (
    #     "Memory consolidation during sleep helps learners retain verbal information better than visual patterns in most cases.",
    #     "What does sleep's memory consolidation process favor between verbal and visual information?",
    #     0.4
    # ),
    # (
    #     "Memory consolidation during sleep helps learners retain verbal information better than visual patterns in most cases.",
    #     "How does memory consolidation during sleep affect learners' retention of verbal versus visual information?",
    #     0.6
    # ),
    # (
    #     "Memory consolidation during sleep helps learners retain verbal information better than visual patterns in most cases.",
    #     "How does memory consolidation during sleep help learners retain verbal information better than visual patterns in most cases?",
    #     0.8
    # ),

    # # Needle 2: User adoption and trust
    # (
    #     "User adoption of complex technological systems often depends on establishing initial trust through consistent performance and transparent operational feedback mechanisms.",
    #     "What influences whether people will accept new technical tools?",
    #     0.2
    # ),
    # (
    #     "User adoption of complex technological systems often depends on establishing initial trust through consistent performance and transparent operational feedback mechanisms.",
    #     "What factors establish the foundation for users accepting complex technological systems?",
    #     0.4
    # ),
    # (
    #     "User adoption of complex technological systems often depends on establishing initial trust through consistent performance and transparent operational feedback mechanisms.",
    #     "What does user adoption of complex technological systems depend on regarding initial trust and performance?",
    #     0.6
    # ),
    # (
    #     "User adoption of complex technological systems often depends on establishing initial trust through consistent performance and transparent operational feedback mechanisms.",
    #     "What does user adoption of complex technological systems often depend on in terms of establishing initial trust through consistent performance and transparent operational feedback mechanisms?",
    #     0.8
    # ),

    # # Needle 3: Operator reliance and error
    # (
    #     "Analysis of automated decision systems revealed that operator reliance on algorithmic recommendations increased error rates by 19% in time-critical scenarios.",
    #     "What impact can automated systems have on human performance under pressure?",
    #     0.2
    # ),
    # (
    #     "Analysis of automated decision systems revealed that operator reliance on algorithmic recommendations increased error rates by 19% in time-critical scenarios.",
    #     "What did the analysis find about operator reliance on algorithms during urgent situations?",
    #     0.4
    # ),
    # (
    #     "Analysis of automated decision systems revealed that operator reliance on algorithmic recommendations increased error rates by 19% in time-critical scenarios.",
    #     "What did analysis of automated decision systems reveal about how operator reliance on algorithmic recommendations affected error rates in time-critical scenarios?",
    #     0.6
    # ),
    # (
    #     "Analysis of automated decision systems revealed that operator reliance on algorithmic recommendations increased error rates by 19% in time-critical scenarios.",
    #     "What percentage increase in error rates did analysis of automated decision systems reveal when operators relied on algorithmic recommendations in time-critical scenarios?",
    #     0.8
    # ),

    # # Needle 4: Automation bias and cognitive miser
    # (
    #     "Automation bias is partly explained by the cognitive miser principle, where humans prefer less effortful automated solutions over more demanding manual analysis, even when accuracy suffers.",
    #     "What human tendency affects how people approach problem-solving methods?",
    #     0.2
    # ),
    # (
    #     "Automation bias is partly explained by the cognitive miser principle, where humans prefer less effortful automated solutions over more demanding manual analysis, even when accuracy suffers.",
    #     "What principle helps explain why people favor automated solutions over manual work?",
    #     0.4
    # ),
    # (
    #     "Automation bias is partly explained by the cognitive miser principle, where humans prefer less effortful automated solutions over more demanding manual analysis, even when accuracy suffers.",
    #     "How does the cognitive miser principle explain automation bias regarding human preferences for automated versus manual solutions?",
    #     0.6
    # ),
    # (
    #     "Automation bias is partly explained by the cognitive miser principle, where humans prefer less effortful automated solutions over more demanding manual analysis, even when accuracy suffers.",
    #     "How is automation bias partly explained by the cognitive miser principle where humans prefer less effortful automated solutions over more demanding manual analysis even when accuracy suffers?",
    #     0.8
    # )
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

        # 1. Separate needles, questions, distractors, and targets
        needles = [pair[0] for pair in pairs_to_test]
        questions = [pair[1] for pair in pairs_to_test]
        distractors = [pair[2] for pair in pairs_to_test]
        targets = [pair[3] for pair in pairs_to_test]

        # 2. Embed all at once (batching)
        print(f"Embedding {len(needles)} needles, {len(questions)} questions, and {len(distractors)} distractors...")
        needle_embeddings = model.embed(needles)
        question_embeddings = model.embed(questions)
        distractor_embeddings = model.embed(distractors)

        if needle_embeddings.size == 0 or question_embeddings.size == 0 or distractor_embeddings.size == 0:
            print(f"Error: Embedding failed for model {model.model_name}. Skipping...")
            continue

        # Check for embedding count mismatch
        if len(needle_embeddings) != len(needles) or len(question_embeddings) != len(questions) or len(distractor_embeddings) != len(distractors):
            print(f"Warning: Mismatch in embedding count for {model.model_name}.")
            continue

        # 3. Calculate similarity for each pair
        for i, pair in enumerate(pairs_to_test):
            n_emb = needle_embeddings[i:i+1]  # Keep 2D shape (1, D)
            q_emb = question_embeddings[i:i+1] # Keep 2D shape (1, D)
            d_emb = distractor_embeddings[i:i+1] # Keep 2D shape (1, D)

            needle_distractor_similarity = cosine_similarity(n_emb, d_emb)[0][0]

            all_results.append({
                "pair_id": i,
                "model": model.model_name,
                "target": targets[i],
                "needle_distractor_sim": needle_distractor_similarity,
                "needle": f"'{pair[0][:50]}...'",
                "distractor": f"'{pair[2][:50]}...'"
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

    # --- 6. Filter and Display Needle-Distractor Pairs that Match Target (±0.02) ---
    print("\n" + "="*80)
    print("NEEDLE-DISTRACTOR PAIRS MATCHING TARGET SIMILARITY (±0.02 difference)")
    print("="*80)

    matching_pairs = []
    for idx, row in results_df.iterrows():
        difference = abs(row['needle_distractor_sim'] - row['target'])
        if difference <= 0.02:
            pair = pairs_to_test[row['pair_id']]
            matching_pairs.append({
                "Pair #": row['pair_id'],
                "Target": row['target'],
                "Actual": round(row['needle_distractor_sim'], 4),
                "Diff": round(difference, 4),
                "Needle": pair[0],
                "Distractor": pair[2]
            })
    
    if matching_pairs:
        match_df = pd.DataFrame(matching_pairs)
        print(f"\nFound {len(matching_pairs)} matching needle-distractor pairs:\n")
        print(match_df.to_string(index=False))
        
        # Summary by target
        print("\n" + "="*80)
        print("SUMMARY BY TARGET SIMILARITY")
        print("="*80)
        for target in [0.6, 0.7, 0.8, 0.9]:
            count = sum(1 for m in matching_pairs if m['Target'] == target)
            print(f"Target {target}: {count} pairs matched")
    else:
        print("❌ No needle-distractor pairs found that match their target within ±0.02 tolerance.")
        print("\nClosest pairs:")
        results_df['difference'] = abs(results_df['needle_distractor_sim'] - results_df['target'])
        closest = results_df.nsmallest(5, 'difference')[['pair_id', 'target', 'needle_distractor_sim', 'difference']]
        print(closest.to_string(index=False))


# --- 5. Standard Python Entry Point ---
if __name__ == "__main__":
    main()