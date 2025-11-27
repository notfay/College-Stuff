# check_multiple_needles.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import pandas as pd

# Import your main class from your script
# (Make sure Topic.py is in the same folder)
from Topic import NIAHTopicSelector 

# --- Your List of Needles ---
# You can add as many as you want here
needles_to_check = [
    "The determination of the will to action is driven not by a greater apparent good, but by the immediate pain of desire, for the removing of this present uneasiness is the first and most necessary step towards happiness.",
    "Man, seldom free from the solicitation of desires, finds his will constantly turned by a succession of uneasinesses; the determination to action focuses on removing present pain for immediate happiness, leaving less force to the attraction of a greater absent good or pleasure.",
    "Man, seldom free from the solicitation of desires, finds his will, a power of the mind, constantly turned by a succession of uneasinesses; the determination to action focuses on removing present pain for immediate happiness, often leaving less force to the attraction of a greater absent good or pleasure.",
    "The will is determined to action not by a greater good, but by a constant succession of present uneasinesses, from both natural wants and acquired fantastical desires, which in their turn set the mind to work."
]

target_cluster_id = 0
blend_in_threshold = 0.9 # Your target from the prompt

print(f"Checking {len(needles_to_check)} needles against Cluster {target_cluster_id}...")
print("="*60)

try:
    # --- 1. Run the Main Pipeline Once ---
    print("Loading haystack and running pipeline (this may take a moment)...")
    with open('CognitiveBiass4.txt', 'r', encoding='utf-8') as f:
        haystack_text = f.read()

    # We use the default 'sentence-transformers' from your example
    selector = NIAHTopicSelector(
        embedding_model="all-MiniLM-L6-v2",
        embedding_provider="sentence-transformers",
        chunk_size=256,
        chunk_overlap=50,
        min_cluster_size=10,
        use_umap=True
    )
    
    # Run the full pipeline to get all data
    results_df = selector.run_pipeline(
        haystack_text,
        min_prevalence=0.02,
        max_prevalence=0.20
    )
    
    if target_cluster_id not in selector.cluster_labels:
        print(f"Error: Cluster {target_cluster_id} not found in results.")
    else:
        # --- 2. Calculate Cluster Centroid Once ---
        print(f"\nCalculating centroid for Cluster {target_cluster_id}...")
        
        # Get all embeddings for our target cluster
        cluster_mask = (selector.cluster_labels == target_cluster_id)
        cluster_embeddings = selector.embeddings[cluster_mask]
        
        # Calculate the average vector (centroid or "fingerprint")
        centroid = np.mean(cluster_embeddings, axis=0)
        normalized_centroid = normalize(centroid.reshape(1, -1))

        # --- 3. Embed ALL Needles at Once (Very Fast) ---
        print(f"Embedding all {len(needles_to_check)} needles in one batch...")
        needle_embeddings = selector._embed_fn(needles_to_check)
        normalized_needles = normalize(needle_embeddings)

        # --- 4. Compare All Needles to the Centroid ---
        # This one line calculates all scores at once
        blend_in_scores = cosine_similarity(normalized_needles, normalized_centroid).flatten()
        
        print("\n--- Blend-in Results ---")
        
        # Use a list to build a DataFrame for a nice table
        results_data = []
        
        for i, needle_text in enumerate(needles_to_check):
            score = blend_in_scores[i]
            verdict = "PASSED" if score > blend_in_threshold else "FAILED"
            
            results_data.append({
                "Needle #": i + 1,
                "Needle Text (Snippet)": f"'{needle_text[:50]}...'",
                "Score": f"{score:.4f}",
                "Verdict": verdict
            })
        
        # Print as a clean DataFrame
        results_table = pd.DataFrame(results_data)
        print(results_table.to_string(index=False))

        print(f"\nThreshold for 'PASSED' was: {blend_in_threshold}")

except FileNotFoundError:
    print("Error: 'John_Lock_V2.txt' not found. Please add it to the directory.")
except Exception as e:
    print(f"An error occurred: {e}")