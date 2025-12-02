"""
Multi-Model Needle Checker - Check if needles blend with haystack AND are novel
Uses 4 embedding models to validate:
1. BLEND-IN: Does the needle semantically match the haystack? (similarity > threshold)
2. NOVELTY: Is the needle unique and not duplicate? (similarity < threshold)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from dataclasses import dataclass
import warnings
import re

warnings.filterwarnings('ignore')

# --- MODEL CONFIGURATIONS ---
MODEL_CONFIGS = [
    {
        "name": "all-MiniLM-L6-v2",
        "provider": "sentence-transformers",
        "prefix": "",
        "trust_remote": False,
        "dims": 384
    },
    {
        "name": "nomic-ai/nomic-embed-text-v1.5",
        "provider": "sentence-transformers",
        "prefix": "search_document: ",
        "trust_remote": True,
        "dims": 768
    },
    {
        "name": "intfloat/e5-base-v2",
        "provider": "sentence-transformers",
        "prefix": "passage: ",
        "trust_remote": False,
        "dims": 768
    },
    {
        "name": "BAAI/bge-m3",
        "provider": "sentence-transformers",
        "prefix": "",
        "trust_remote": False,
        "dims": 1024
    }
]

@dataclass
class NeedleCheckResult:
    """Result for a single needle across all models"""
    needle_text: str
    blend_in_scores: Dict[str, float]  # Model -> similarity to haystack
    novelty_scores: Dict[str, float]   # Model -> max similarity to haystack chunks
    blend_in_status: str  # PASS/FAIL
    novelty_status: str   # PASS/FAIL
    overall_status: str   # PASS/FAIL
    mean_blend_score: float
    mean_novelty_score: float


class MultiModelNeedleChecker:
    """
    Check needles using multiple embedding models for:
    1. Blend-in: Overall semantic similarity to haystack
    2. Novelty: Uniqueness (not duplicating existing content)
    """
    
    def __init__(
        self,
        model_configs: List[Dict] = None,
        chunking_method: str = "semantic",  # "semantic", "paragraph", or "sliding"
        chunk_size: int = 256,
        chunk_overlap: int = 128,
        blend_in_threshold: float = 0.75,  # Min similarity to haystack (must be >= this)
        novelty_threshold: float = 0.85    # Max similarity to any chunk (must be < this)
    ):
        """
        Initialize with multiple models
        
        Args:
            model_configs: List of model configuration dicts
            chunking_method: "semantic" (sentence clustering), "paragraph" (natural breaks), or "sliding" (fixed-size)
            chunk_size: Size of text chunks for sliding window (ignored for semantic/paragraph)
            chunk_overlap: Overlap for sliding window (ignored for semantic/paragraph)
            blend_in_threshold: Minimum score to pass blend-in test (higher = more similar required)
            novelty_threshold: Maximum score to pass novelty test (lower = more unique required)
        """
        self.model_configs = model_configs or MODEL_CONFIGS
        self.chunking_method = chunking_method
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.blend_in_threshold = blend_in_threshold
        self.novelty_threshold = novelty_threshold
        
        # Load all models
        self.models = {}
        print("\n" + "="*80)
        print("INITIALIZING MULTI-MODEL NEEDLE CHECKER")
        print("="*80)
        
        for config in self.model_configs:
            model_name = config["name"]
            print(f"\nLoading: {model_name}...")
            try:
                self.models[model_name] = {
                    "model": SentenceTransformer(
                        model_name, 
                        trust_remote_code=config["trust_remote"]
                    ),
                    "prefix": config["prefix"],
                    "dims": config["dims"]
                }
                print(f"✓ Loaded successfully (dims: {config['dims']})")
            except Exception as e:
                print(f"✗ Failed to load {model_name}: {e}")
        
        self.haystack_text = None
        self.haystack_chunks = []
        self.haystack_embeddings = {}  # model_name -> embeddings
        
        print(f"\n✓ Initialized {len(self.models)} models")
        print(f"  Chunking method: {self.chunking_method}")
        print(f"  Blend-in threshold: >= {self.blend_in_threshold:.2f}")
        print(f"  Novelty threshold: < {self.novelty_threshold:.2f}")
        print("="*80 + "\n")
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex"""
        # Split on sentence endings (.!?) followed by space and capital letter
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _chunk_semantic(self, text: str, min_sentences: int = 3, max_sentences: int = 8) -> List[str]:
        """
        Semantic chunking: Group sentences into coherent chunks based on natural flow
        Better for academic/structured text with clear topic boundaries
        """
        sentences = self._split_into_sentences(text)
        chunks = []
        current_chunk = []
        
        for sentence in sentences:
            current_chunk.append(sentence)
            
            # Create chunk if we hit max size or if sentence ends a clear topic
            # (heuristic: sentence > 100 chars or next sentence starts with transition words)
            if len(current_chunk) >= max_sentences:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
            elif len(current_chunk) >= min_sentences and len(sentence) > 100:
                # Long sentence often marks end of concept
                chunks.append(' '.join(current_chunk))
                current_chunk = []
        
        # Add remaining sentences
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _chunk_paragraph(self, text: str) -> List[str]:
        """
        Paragraph-based chunking: Split on natural paragraph breaks
        Best for well-formatted documents with clear section divisions
        """
        # Split on double newlines (paragraph breaks)
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        
        for para in paragraphs:
            para = para.strip()
            if len(para.split()) >= 10:  # Minimum 10 words
                chunks.append(para)
        
        return chunks
    
    def _chunk_sliding_window(self, text: str) -> List[str]:
        """
        Traditional sliding window chunking: Fixed-size chunks with overlap
        Fallback for unstructured text
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            if len(chunk.split()) >= 20:  # Minimum chunk size
                chunks.append(chunk)
        
        return chunks
    
    def _chunk_text(self, text: str) -> List[str]:
        """Route to appropriate chunking method"""
        if self.chunking_method == "semantic":
            return self._chunk_semantic(text)
        elif self.chunking_method == "paragraph":
            return self._chunk_paragraph(text)
        elif self.chunking_method == "sliding":
            return self._chunk_sliding_window(text)
        else:
            raise ValueError(f"Unknown chunking method: {self.chunking_method}")
    
    def _embed_with_prefix(self, texts: List[str], model_name: str) -> np.ndarray:
        """Embed texts with model-specific prefix"""
        model_info = self.models[model_name]
        prefix = model_info["prefix"]
        
        if prefix:
            texts = [prefix + text for text in texts]
        
        embeddings = model_info["model"].encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        
        return embeddings
    
    def load_haystack(self, haystack_text: str = None, haystack_file: str = None):
        """
        Load haystack and create embeddings with all models
        """
        if haystack_file:
            with open(haystack_file, 'r', encoding='utf-8') as f:
                self.haystack_text = f.read()
        elif haystack_text:
            self.haystack_text = haystack_text
        else:
            raise ValueError("Must provide either haystack_text or haystack_file")
        
        # Create chunks
        self.haystack_chunks = self._chunk_text(self.haystack_text)
        
        print("\n" + "="*80)
        print("LOADING HAYSTACK")
        print("="*80)
        print(f"Haystack size: {len(self.haystack_text)} chars")
        print(f"Chunking method: {self.chunking_method}")
        print(f"Number of chunks: {len(self.haystack_chunks)}")
        
        if self.haystack_chunks:
            chunk_lens = [len(c.split()) for c in self.haystack_chunks]
            print(f"Chunk size range: {min(chunk_lens)}-{max(chunk_lens)} words")
            print(f"Average chunk size: {np.mean(chunk_lens):.1f} words\n")
        else:
            print(f"WARNING: No chunks created!\n")
        
        # Embed haystack with each model
        for model_name in self.models.keys():
            print(f"Embedding haystack with {model_name}...")
            
            # Embed full haystack
            full_embedding = self._embed_with_prefix([self.haystack_text], model_name)
            
            # Embed chunks
            chunk_embeddings = self._embed_with_prefix(self.haystack_chunks, model_name)
            
            self.haystack_embeddings[model_name] = {
                "full": full_embedding,
                "chunks": chunk_embeddings
            }
            
            print(f"  ✓ Full: {full_embedding.shape}")
            print(f"  ✓ Chunks: {chunk_embeddings.shape}\n")
        
        print("="*80 + "\n")
    
    def check_needle(self, needle: str, verbose: bool = True) -> NeedleCheckResult:
        """
        Check a single needle across all models
        
        Returns:
            NeedleCheckResult with blend-in and novelty scores
        """
        if not self.haystack_text:
            raise ValueError("Must load haystack first using load_haystack()")
        
        blend_in_scores = {}
        novelty_scores = {}
        
        if verbose:
            print(f"\n{'─'*80}")
            print(f"CHECKING NEEDLE")
            print(f"{'─'*80}")
            print(f"Text: {needle[:100]}{'...' if len(needle) > 100 else ''}\n")
        
        for model_name in self.models.keys():
            # Embed needle
            needle_embedding = self._embed_with_prefix([needle], model_name)
            
            # 1. BLEND-IN: Similarity to full haystack
            full_sim = cosine_similarity(
                needle_embedding,
                self.haystack_embeddings[model_name]["full"]
            )[0][0]
            
            blend_in_scores[model_name] = float(full_sim)
            
            # 2. NOVELTY: Max similarity to any chunk
            chunk_sims = cosine_similarity(
                needle_embedding,
                self.haystack_embeddings[model_name]["chunks"]
            )[0]
            
            max_chunk_sim = float(np.max(chunk_sims))
            novelty_scores[model_name] = max_chunk_sim
            
            if verbose:
                print(f"{model_name}:")
                print(f"  Blend-in:  {full_sim:.4f} {'✓ PASS' if full_sim >= self.blend_in_threshold else '✗ FAIL'}")
                print(f"  Novelty:   {max_chunk_sim:.4f} {'✓ PASS' if max_chunk_sim < self.novelty_threshold else '✗ FAIL'}")
        
        # Calculate means
        mean_blend = np.mean(list(blend_in_scores.values()))
        mean_novelty = np.mean(list(novelty_scores.values()))
        
        # Determine pass/fail
        blend_pass = mean_blend >= self.blend_in_threshold
        novelty_pass = mean_novelty < self.novelty_threshold
        
        blend_status = "✓ PASS" if blend_pass else "✗ FAIL"
        novelty_status = "✓ PASS" if novelty_pass else "✗ FAIL"
        overall_status = "✓ PASS" if (blend_pass and novelty_pass) else "✗ FAIL"
        
        if verbose:
            print(f"\n{'─'*40}")
            print(f"SUMMARY:")
            print(f"  Blend-in (mean):  {mean_blend:.4f} {blend_status}")
            print(f"  Novelty (mean):   {mean_novelty:.4f} {novelty_status}")
            print(f"  Overall:          {overall_status}")
            print(f"{'─'*80}\n")
        
        return NeedleCheckResult(
            needle_text=needle,
            blend_in_scores=blend_in_scores,
            novelty_scores=novelty_scores,
            blend_in_status=blend_status,
            novelty_status=novelty_status,
            overall_status=overall_status,
            mean_blend_score=mean_blend,
            mean_novelty_score=mean_novelty
        )
    
    def check_multiple_needles(
        self, 
        needles: List[str],
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Check multiple needles and return results as DataFrame
        """
        results = []
        
        print("\n" + "="*80)
        print(f"CHECKING {len(needles)} NEEDLES")
        print("="*80)
        
        for i, needle in enumerate(needles, 1):
            print(f"\n[{i}/{len(needles)}]")
            result = self.check_needle(needle, verbose=verbose)
            
            # Create row
            row = {
                "needle_id": i,
                "needle": needle[:80] + ('...' if len(needle) > 80 else ''),
                "mean_blend_in": result.mean_blend_score,
                "mean_novelty": result.mean_novelty_score,
                "blend_in_status": result.blend_in_status,
                "novelty_status": result.novelty_status,
                "overall_status": result.overall_status
            }
            
            # Add individual model scores
            for model_name in self.models.keys():
                short_name = model_name.split('/')[-1][:15]
                row[f"blend_{short_name}"] = result.blend_in_scores[model_name]
                row[f"novel_{short_name}"] = result.novelty_scores[model_name]
            
            results.append(row)
        
        df = pd.DataFrame(results)
        
        # Print summary
        print("\n" + "="*80)
        print("FINAL SUMMARY")
        print("="*80)
        
        pass_count = sum(1 for r in results if '✓' in r['overall_status'])
        blend_pass = sum(1 for r in results if '✓' in r['blend_in_status'])
        novel_pass = sum(1 for r in results if '✓' in r['novelty_status'])
        
        print(f"\nOverall: {pass_count}/{len(needles)} passed both tests")
        print(f"Blend-in: {blend_pass}/{len(needles)} passed (>= {self.blend_in_threshold:.2f})")
        print(f"Novelty: {novel_pass}/{len(needles)} passed (< {self.novelty_threshold:.2f})")
        
        print("\nPer-Model Averages:")
        for model_name in self.models.keys():
            short_name = model_name.split('/')[-1][:15]
            avg_blend = df[f"blend_{short_name}"].mean()
            avg_novel = df[f"novel_{short_name}"].mean()
            print(f"  {model_name}:")
            print(f"    Blend-in: {avg_blend:.4f}")
            print(f"    Novelty:  {avg_novel:.4f}")
        
        print("\n" + "="*80 + "\n")
        
        return df


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    
    # Test needles
    test_needles = [
        "A study asked people to mark the single feature they first noticed and then measured how that mark changed option choice in later tasks",
        "Report and measure 'Future Bias' item rate in a system test to show a new way to compare options.",
        "Record and analyze 'Final Effect' group data in a field study to find a better method for decisions.",
        "Cognitive biases are systematic patterns of deviation from norm or rationality in judgment.",
        "The availability heuristic leads people to judge likelihood based on how easily examples come to mind."
    ]
    
    # Initialize checker with all 4 models
    # Try different chunking methods:
    # "semantic" - Groups sentences into coherent units (BEST for cognitive bias haystack)
    # "paragraph" - Splits on paragraph breaks (good for well-formatted text)
    # "sliding" - Traditional fixed-size sliding window (fallback)
    
    checker = MultiModelNeedleChecker(
        model_configs=MODEL_CONFIGS,
        chunking_method="semantic",  # ← RECOMMENDED for this haystack
        chunk_size=256,              # Only used if chunking_method="sliding"
        chunk_overlap=128,           # Only used if chunking_method="sliding"
        blend_in_threshold=0.75,     # Must be >= 0.75 to pass
        novelty_threshold=0.85       # Must be < 0.85 to pass
    )
    
    # Load haystack
    print("Loading haystack...")
    checker.load_haystack(haystack_file='CognitiveBias4.txt')
    
    # Check all needles
    results_df = checker.check_multiple_needles(test_needles, verbose=True)
    
    # Save results
    output_file = "needle_check_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")
    
    # Display compact results
    print("\n" + "="*80)
    print("COMPACT RESULTS TABLE")
    print("="*80 + "\n")
    
    display_cols = [
        "needle_id",
        "needle",
        "mean_blend_in",
        "mean_novelty",
        "overall_status"
    ]
    
    print(results_df[display_cols].to_string(index=False))
