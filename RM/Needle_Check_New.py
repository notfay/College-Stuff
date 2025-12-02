# check_multiple_needles.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import pandas as pd
import os

# Import your main class from your script
from Topic_New import NIAHLexicalTopicSelector 

# --- Your List of Needles ---
needles_to_check = [
    # === Low Scoring Control Needles (Target < 0.2) ===
    "The 'Velara' system rate was found to be at a high level.",
    "The large 'Chronos' group example shows a different case.",
    "The 'Alpha' process uses a small time to show the difference.",
    "People present at the 'Type-Z' event saw the second option.",
    "The available 'Gamma' information is in the general system."
]

blend_in_threshold = 0.9 
print(f"Checking {len(needles_to_check)} needles against Top Cluster...")
print("="*60)

try:
    # --- 1. Run the Main Pipeline Once ---
    # Attempt to locate a default haystack file if simpler
    haystack_file = 'CognitiveBiasHaystack.txt'
    if not os.path.exists(haystack_file):
        # Fallback to the one mentioned in original file if needed
        haystack_file = 'CognitiveBias4.txt'
        
    print(f"Loading haystack from {haystack_file} and running pipeline...")
    with open(haystack_file, 'r', encoding='utf-8') as f:
        haystack_text = f.read()

    # UPDATED: Use new Class Name
    selector = NIAHLexicalTopicSelector(
        embedding_model="all-MiniLM-L6-v2",
        embedding_provider="sentence-transformers",
        chunk_size=256,
        chunk_overlap=50,
        min_cluster_size=10,
        use_umap=True
    )
    
    # UPDATED: No prevalence args needed anymore
    results_df = selector.run_pipeline(haystack_text)
    
    # 1. Get the Top 1 Cluster ID automatically
    if selector.ranked_stats:
        top_cluster = selector.ranked_stats[0]  # Get the highest scoring cluster
        target_cluster_id = top_cluster.cluster_id
        print(f"\n✅ Automatically targeting Top Cluster: {target_cluster_id}")
        print(f"   Score: {top_cluster.score:.3f} | Terms: {', '.join(top_cluster.top_terms[:5])}")
    else:
        print("Error: No clusters were found in the haystack.")
        exit()

    # Access the stored labels (enabled by the update to Topic.py)
    if selector.cluster_labels is None:
        print("Error: Pipeline did not generate cluster labels.")
        exit()
        
    unique_clusters = set(selector.cluster_labels) - {-1}
    
    if target_cluster_id not in unique_clusters:
        print(f"Error: Cluster {target_cluster_id} not found in results.")
        print(f"Available clusters: {unique_clusters}")
    else:
        # --- 2. Prepare Cluster Data ---
        print(f"\nRetrieving embeddings for Cluster {target_cluster_id}...")
        
        # Get all embeddings for the target cluster (Raw, High-Dimensional)
        cluster_mask = (selector.cluster_labels == target_cluster_id)
        cluster_embeddings = selector.embeddings[cluster_mask]
        
        # Note: They are already normalized in Topic.py, but safe to re-normalize
        # normalized_cluster = normalize(cluster_embeddings) 
        
        # --- 3. Embed ALL Needles ---
        print(f"Embedding {len(needles_to_check)} needles...")
        needle_embeddings = selector._embed_fn(needles_to_check)
        normalized_needles = normalize(needle_embeddings)

        # --- 4. Compare Needles to Nearest Neighbors (k-NN) ---
        print("Calculating k-NN blending scores...")
        
        blend_in_scores = []
        k_neighbors = 5
        
        for needle_vec in normalized_needles:
            # Calculate similarity of this needle against ALL cluster chunks
            sims = cosine_similarity(needle_vec.reshape(1, -1), cluster_embeddings).flatten()
            
            # Sort and take the top K highest scores
            top_k_sims = np.sort(sims)[-k_neighbors:]
            
            # Average the top K scores to get the final blend score
            avg_score = np.mean(top_k_sims)
            blend_in_scores.append(avg_score)
        
        print("\n--- Blend-in Results ---")
        
        results_data = []
        for i, needle_text in enumerate(needles_to_check):
            score = blend_in_scores[i]
            pass_fail = "PASS" if score >= blend_in_threshold else "FAIL"
            
            results_data.append({
                "ID": i + 1,
                "Needle Snippet": f"{needle_text[:40]}...",
                "Score": f"{score:.4f}",
                "Status": pass_fail
            })
        
        results_table = pd.DataFrame(results_data)
        print(results_table.to_string(index=False))

        print(f"\nTarget Threshold: {blend_in_threshold}")

except FileNotFoundError:
    print("Error: Haystack file not found. Please ensure 'CognitiveBiasHaystack.txt' exists.")
except Exception as e:
    print(f"An error occurred: {e}")