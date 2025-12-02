"""
Haystack Duplicate Checker - Check if needles already exist in haystack
Uses semantic similarity to detect duplicate or near-duplicate facts
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
import pandas as pd
from typing import List, Dict, Tuple
import re


class HaystackDuplicateChecker:
    """
    Check if needle sentences already exist in a haystack document
    using semantic similarity (cosine similarity of embeddings).
    """
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 chunk_size: int = 256,
                 chunk_overlap: int = 128,
                 similarity_threshold: float = 0.85):
        """
        Initialize the checker.
        
        Args:
            model_name: Sentence transformer model to use
            chunk_size: Size of text chunks (in characters)
            chunk_overlap: Overlap between chunks
            similarity_threshold: Default threshold for "duplicate" detection
        """
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_threshold = similarity_threshold
        
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        
        # Storage for haystack data
        self.haystack_chunks = None
        self.haystack_embeddings = None
        
    
    def _clean_text(self, text: str) -> str:
        """Basic text cleaning."""
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.strip()
        return text
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks (Word-aware).
        """
        # 1. Split into words first to avoid cutting words in half
        words = self._clean_text(text).split()
        
        # Convert char params to approx word count (1 word ≈ 5 chars)
        # 256 chars ≈ 50 words
        word_chunk_size = int(self.chunk_size / 5) 
        word_overlap = int(self.chunk_overlap / 5)
        
        chunks = []
        position = 0
        
        while position < len(words):
            # 2. Slice by words
            chunk_words = words[position : position + word_chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            if chunk_text:
                chunks.append(chunk_text)
            
            # 3. Move window
            position += (word_chunk_size - word_overlap)
        
        return chunks
    
    
    def load_haystack(self, haystack_text: str = None, haystack_file: str = None):
        """
        Load and process haystack document.
        
        Args:
            haystack_text: Raw text string (if provided)
            haystack_file: Path to text file (if provided)
        """
        if haystack_text is None and haystack_file is None:
            raise ValueError("Must provide either haystack_text or haystack_file")
        
        # Load text
        if haystack_file:
            print(f"Loading haystack from file: {haystack_file}")
            with open(haystack_file, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            text = haystack_text
        
        # Chunk the text
        print("Chunking haystack...")
        self.haystack_chunks = self._chunk_text(text)
        print(f"Created {len(self.haystack_chunks)} chunks")
        
        # Embed all chunks
        print("Embedding haystack chunks (this may take a moment)...")
        self.haystack_embeddings = self.model.encode(
            self.haystack_chunks,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        self.haystack_embeddings = normalize(self.haystack_embeddings)
        
        print(f"Haystack ready: {len(self.haystack_chunks)} chunks embedded")
    
    
    def check_needle(self, 
                     needle: str, 
                     threshold: float = None,
                     top_k: int = 5) -> Dict:
        """
        Check if a single needle exists in the haystack.
        
        Args:
            needle: The needle sentence to check
            threshold: Similarity threshold (uses default if None)
            top_k: Number of top matches to return
            
        Returns:
            Dictionary with results
        """
        if self.haystack_embeddings is None:
            raise ValueError("Must load haystack first using load_haystack()")
        
        threshold = threshold or self.similarity_threshold
        
        # Embed the needle
        needle_embedding = self.model.encode([needle], convert_to_numpy=True)
        needle_embedding = normalize(needle_embedding)
        
        # Calculate similarities
        similarities = cosine_similarity(
            needle_embedding, 
            self.haystack_embeddings
        ).flatten()
        
        # Get top matches
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_scores = similarities[top_indices]
        top_chunks = [self.haystack_chunks[i] for i in top_indices]
        
        # Determine if duplicate
        max_similarity = float(np.max(similarities))
        is_duplicate = max_similarity >= threshold
        
        return {
            'needle': needle,
            'is_duplicate': is_duplicate,
            'max_similarity': max_similarity,
            'threshold': threshold,
            'top_matches': [
                {
                    'similarity': float(score),
                    'chunk': chunk,
                    'chunk_index': int(idx)
                }
                for score, chunk, idx in zip(top_scores, top_chunks, top_indices)
            ]
        }
    
    
    def check_multiple_needles(self,
                              needles: List[str],
                              threshold: float = None,
                              top_k: int = 3) -> pd.DataFrame:
        """
        Check multiple needles at once (batch processing).
        
        Args:
            needles: List of needle sentences
            threshold: Similarity threshold
            top_k: Number of top matches per needle
            
        Returns:
            DataFrame with results for all needles
        """
        if self.haystack_embeddings is None:
            raise ValueError("Must load haystack first using load_haystack()")
        
        threshold = threshold or self.similarity_threshold
        
        print(f"Embedding {len(needles)} needles...")
        needle_embeddings = self.model.encode(
            needles,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        needle_embeddings = normalize(needle_embeddings)
        
        print("Calculating similarities...")
        similarities = cosine_similarity(
            needle_embeddings,
            self.haystack_embeddings
        )
        
        # Process results
        results = []
        for i, needle in enumerate(needles):
            needle_sims = similarities[i]
            max_sim = float(np.max(needle_sims))
            is_dup = max_sim >= threshold
            
            # Get top match
            top_idx = int(np.argmax(needle_sims))
            top_chunk = self.haystack_chunks[top_idx]
            
            results.append({
                'needle': needle[:60] + '...' if len(needle) > 60 else needle,
                'max_similarity': f"{max_sim:.4f}",
                'is_duplicate': '⚠️ YES' if is_dup else '✓ SAFE',
                'top_match_preview': top_chunk[:80] + '...' if len(top_chunk) > 80 else top_chunk
            })
        
        return pd.DataFrame(results)
    
    
    def find_safe_needles(self,
                         candidate_needles: List[str],
                         threshold: float = None) -> List[Tuple[str, float]]:
        """
        Filter candidate needles to find only those NOT in haystack.
        
        Args:
            candidate_needles: List of potential needles
            threshold: Maximum similarity allowed
            
        Returns:
            List of (needle, max_similarity) tuples for safe needles
        """
        threshold = threshold or self.similarity_threshold
        
        results = []
        needle_embeddings = self.model.encode(
            candidate_needles,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        needle_embeddings = normalize(needle_embeddings)
        
        similarities = cosine_similarity(
            needle_embeddings,
            self.haystack_embeddings
        )
        
        for i, needle in enumerate(candidate_needles):
            max_sim = float(np.max(similarities[i]))
            if max_sim < threshold:
                results.append((needle, max_sim))
        
        return results


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    
    # Initialize checker
    checker = HaystackDuplicateChecker(
        model_name="all-MiniLM-L6-v2",
        chunk_size=256,
        chunk_overlap=50,
        similarity_threshold=0.85  # Adjust based on your needs
    )
    
    # Load your haystack
    checker.load_haystack(haystack_file='CognitiveBias4.txt')
    
    # Test needles
    test_needles = [
       "The 'Recent Memory' news uses a current story to tell the truth. The report provided a clear account of the situation."
    ]

    
    print("\n" + "="*60)
    print("CHECKING MULTIPLE NEEDLES")
    print("="*60)
    
    # Check all at once
    results_df = checker.check_multiple_needles(
        test_needles,
        threshold=0.85,
        top_k=3
    )
    
    print(results_df.to_string(index=False))
    
    for i, needle in enumerate(test_needles, 1):
        print(f"\n--- Checking Needle {i}/{len(test_needles)} ---")
        
        # Call the detailed check for this specific needle
        detailed = checker.check_needle(
            needle,
            threshold=0.85,
            top_k=5  # Show top 5 matches as requested
        )
        
        print(f"\nNeedle: {detailed['needle']}")
        print(f"Is Duplicate: {detailed['is_duplicate']} (Max Similarity: {detailed['max_similarity']:.4f})")
        print(f"Threshold: {detailed['threshold']}")
        print("\nTop 5 Matches:")
        
        # Loop through the top matches and print their FULL text
        for j, match in enumerate(detailed['top_matches'], 1):
            print(f"\n  [{j}] Similarity: {match['similarity']:.4f}")
            # This line prints the full, uncropped chunk text
            print(f"      Full Chunk: {match['chunk']}")
        
        print("\n" + ("-" * 60))
    
    print("\n" + "="*60)
    print("FINDING SAFE NEEDLES")
    print("="*60)
    
    # Filter to find safe needles
    safe_needles = checker.find_safe_needles(
        test_needles,
        threshold=0.85
    )
    
    print(f"\nFound {len(safe_needles)} safe needles:")
    for needle, sim in safe_needles:
        print(f"  - [{sim:.4f}] {needle[:70]}...")