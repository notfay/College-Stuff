import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import re
from collections import Counter, defaultdict
import spacy
import os
import sys
import argparse

# For embeddings and clustering
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# --- DATACLASSES ---

@dataclass
class Chunk:
    chunk_id: int
    text: str
    position: int
    token_count: int

@dataclass
class WordLexicalStats:
    word: str
    lemma: str
    pos: str
    
    # Global Stats (Context: Entire Haystack)
    global_count: int
    global_spread: float
    
    # Cluster Stats (Context: Specific Cluster)
    cluster_count: int
    cluster_spread: float
    
    # Classification
    is_global_safe: bool
    is_cluster_safe: bool

@dataclass
class ClusterStats:
    cluster_id: int
    prevalence: float
    cohesion: float
    jargon: float
    spread: float
    ner_density: float
    top_terms: List[str]
    lexical_profile: List[WordLexicalStats]
    score: float
    representatives: List[str]

class NIAHLexicalTopicSelector:
    """
    Integrated Pipeline: Topic Selection + Lexical Camouflage Analysis.
    Identifies topics and finds 'Safe Words' specific to each cluster.
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_provider: str = "sentence-transformers",
        chunk_size: int = 256,
        chunk_overlap: int = 50,
        min_cluster_size: int = 10,
        # CONTEXT ROT PARAMETER: Increased to 15 for robust/dense clusters
        min_samples: int = 15,
        use_umap: bool = True,
        # CONTEXT ROT PARAMETER: Increased to 50 for better information retention
        umap_n_components: int = 50 
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.use_umap = use_umap
        self.umap_n_components = umap_n_components
        self.ranked_stats = []
        
        # Store these for external access (e.g., Needle_Check.py)
        self.embeddings = None
        self.cluster_labels = None
        self.chunks = []
        
        print("Loading spaCy model...")
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
            
        # Optimization: Disable pipes we don't strictly need for counting
        self.nlp_disable_pipes = [pipe for pipe in self.nlp.pipe_names 
                                 if pipe not in ['tok2vec', 'tagger', 'attribute_ruler', 'lemmatizer']]

        print(f"Loading embedding model: {embedding_provider}/{embedding_model}...")
        self.embedding_provider = embedding_provider
        
        if embedding_provider == "sentence-transformers":
            self.encoder = SentenceTransformer(embedding_model)
            self._embed_fn = self._embed_sentence_transformers
        
        # Standard stopwords + common verbs
        self.common_words = self._load_common_words()
        
        # Storage for global stats
        self.global_word_stats = {} 
        self.total_doc_segments = 20

    def _load_common_words(self) -> set:
        return set([
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
            'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
            'is', 'are', 'was', 'were', 'been', 'has', 'had'
        ])

    def calculate_juillands_d(self, counts: List[int]) -> float:
        """
        Calculates Juilland's D dispersion metric.
        Range: 0.0 (clumped in one place) to 1.0 (perfectly even distribution).
        """
        n = len(counts)
        total = sum(counts)
        if total == 0 or n == 0: return 0.0
        
        mean = total / n
        if mean == 0: return 0.0
        
        variance = sum((c - mean) ** 2 for c in counts) / n
        cv = np.sqrt(variance) / mean
        d = 1 - (cv / np.sqrt(n - 1))
        
        return max(0.0, min(1.0, d))

    def preprocess_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def chunk_text(self, text: str) -> List[Chunk]:
        text = self.preprocess_text(text)
        words = text.split()
        chunks = []
        position = 0
        chunk_id = 0
        
        while position < len(words):
            chunk_words = words[position:position + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            chunks.append(Chunk(
                chunk_id=chunk_id, text=chunk_text,
                position=position, token_count=len(chunk_words)
            ))
            position += self.chunk_size - self.chunk_overlap
            chunk_id += 1
        return chunks

    def analyze_global_lexicon(self, full_text: str):
        """
        1. Global Vocabulary Mapping
        Builds a map of word frequency and dispersion across the entire document
        BEFORE clustering begins.
        """
        print("Running Global Lexical Analysis...")
        
        # Divide text into segments for global dispersion calculation
        segment_size = max(100, len(full_text) // self.total_doc_segments)
        segments = [full_text[i:i+segment_size] for i in range(0, len(full_text), segment_size)]
        real_n_segments = len(segments)
        
        word_segment_counts = defaultdict(lambda: [0] * real_n_segments)
        word_meta = {}

        with self.nlp.select_pipes(disable=self.nlp_disable_pipes):
            for i, seg in enumerate(segments):
                doc = self.nlp(seg)
                for token in doc:
                    if token.is_alpha and not token.is_stop and len(token.text) > 2:
                        lemma = token.lemma_.lower()
                        # Only track Nouns, Verbs, Adjectives for safety candidates
                        if token.pos_ in ['NOUN', 'VERB', 'ADJ']:
                            word_segment_counts[lemma][i] += 1
                            if lemma not in word_meta:
                                word_meta[lemma] = {'pos': token.pos_}

        for lemma, counts in word_segment_counts.items():
            total = sum(counts)
            d_score = self.calculate_juillands_d(counts)
            self.global_word_stats[lemma] = {
                'count': total,
                'spread': d_score,
                'pos': word_meta[lemma]['pos']
            }
        print(f"Global analysis complete. Vocabulary size: {len(self.global_word_stats)}")

    def _embed_sentence_transformers(self, texts: List[str]) -> np.ndarray:
        return self.encoder.encode(texts, show_progress_bar=True)

    def get_cluster_lexical_profile(self, cluster_chunks: List[Chunk], top_n=150) -> List[WordLexicalStats]:
        """
        2. Per-Cluster Vocabulary Analysis
        Calculates lexical statistics strictly within the context of the cluster.
        """
        cluster_text = " ".join([c.text for c in cluster_chunks])
        
        # 1. Count frequencies strictly within this cluster
        cluster_word_counts = Counter()
        with self.nlp.select_pipes(disable=self.nlp_disable_pipes):
            doc = self.nlp(cluster_text)
            for token in doc:
                if token.is_alpha and not token.is_stop and len(token.text) > 2:
                    lemma = token.lemma_.lower()
                    if token.pos_ in ['NOUN', 'VERB', 'ADJ']:
                        cluster_word_counts[lemma] += 1
        
        top_words_lemma = [w for w, c in cluster_word_counts.most_common(top_n)]
        lexical_stats = []
        
        for lemma in top_words_lemma:
            # 2. Calculate Cluster Dispersion
            # How evenly is this word spread across the specific chunks that make up this cluster?
            counts_per_chunk = []
            for chunk in cluster_chunks:
                # Simple regex count is faster for this specific check
                count = len(re.findall(r'\b' + re.escape(lemma) + r'\b', chunk.text.lower()))
                counts_per_chunk.append(count)
            
            cluster_spread = self.calculate_juillands_d(counts_per_chunk)
            
            # Retrieve global stats (pre-calculated)
            g_stats = self.global_word_stats.get(lemma, {'count': 0, 'spread': 0.0, 'pos': 'UNK'})
            
            # 3. Determine Safety/Flagging
            # Global Safe: Common everywhere, spread everywhere.
            is_global_safe = (g_stats['spread'] > 0.6) and (g_stats['count'] > 50)
            
            # Cluster Safe (⭐): Very spread out within this cluster (integral to topic),
            # but maybe not globally ubiquitous.
            is_cluster_safe = (cluster_spread > 0.7)
            
            lexical_stats.append(WordLexicalStats(
                word=lemma, 
                lemma=lemma, 
                pos=g_stats['pos'],
                global_count=g_stats['count'], 
                global_spread=g_stats['spread'],
                cluster_count=cluster_word_counts[lemma], 
                cluster_spread=cluster_spread,
                is_global_safe=is_global_safe,
                is_cluster_safe=is_cluster_safe
            ))
            
        return lexical_stats

    def compute_cluster_stats(
        self, cluster_id: int, chunks: List[Chunk], 
        embeddings: np.ndarray, cluster_labels: np.ndarray
    ) -> Optional[ClusterStats]:
        
        if cluster_id == -1: return None
        mask = cluster_labels == cluster_id
        cluster_chunks = [c for c, m in zip(chunks, mask) if m]
        if not cluster_chunks: return None
        
        cluster_embeddings = embeddings[mask]
        cluster_texts = [c.text for c in cluster_chunks]
        
        # --- Metrics ---
        prevalence = len(cluster_chunks) / len(chunks)
        
        if len(cluster_embeddings) < 2: cohesion = 1.0
        else:
            sim = cosine_similarity(cluster_embeddings)
            cohesion = float(np.mean(sim[np.triu_indices_from(sim, k=1)]))
            
        # Geographic Spread in Doc (Position based)
        positions = [c.position for c in cluster_chunks]
        norm_pos = np.array(positions) / (chunks[-1].position + 1)
        spread = min(np.std(norm_pos) / 0.289, 1.0) if len(positions) > 1 else 0.0
        
        # --- Lexical Analysis ---
        # Get the detailed breakdown (top 50)
        lexical_profile = self.get_cluster_lexical_profile(cluster_chunks, top_n=150)
        
        # TF-IDF for "Distinctive Terms" (Jargon)
        vectorizer = TfidfVectorizer(max_features=10, stop_words='english')
        try:
            vectorizer.fit_transform(cluster_texts)
            top_terms = vectorizer.get_feature_names_out().tolist()
        except: top_terms = []
        
        # Jargon score
        uncommon = sum(1 for t in top_terms if t not in self.common_words)
        jargon = uncommon / len(top_terms) if top_terms else 0
        
        # NER Density
        ner_sample = cluster_texts[:20]
        ne_count, token_count = 0, 0
        for txt in ner_sample:
            doc = self.nlp(txt)
            ne_count += len(doc.ents)
            token_count += len(doc)
        ner_density = (ne_count / token_count * 100) if token_count else 0
        
        # Representatives (Finding the text closest to the cluster center)
        centroid = cluster_embeddings.mean(axis=0, keepdims=True)
        relevance = cosine_similarity(cluster_embeddings, centroid).flatten()
        # Get top 3 most relevant chunks
        top_idx = np.argsort(relevance)[::-1][:3]
        representatives = [cluster_texts[i] for i in top_idx]
        
        # Scoring
        score = (0.3 * prevalence + 0.3 * cohesion + 0.4 * spread - 0.1 * jargon)
        
        return ClusterStats(
            cluster_id=cluster_id, prevalence=prevalence, cohesion=cohesion,
            jargon=jargon, spread=spread,
            ner_density=ner_density, top_terms=top_terms,
            lexical_profile=lexical_profile,
            score=score, representatives=representatives
        )

    def run_pipeline(self, text: str):
        print("="*60 + "\nNIAH LEXICAL TOPIC PIPELINE\n" + "="*60)
        
        # 1. Run Global Analysis First
        self.analyze_global_lexicon(text)
        
        self.chunks = self.chunk_text(text)
        print(f"Total Chunks: {len(self.chunks)}")
        
        # Save embeddings to self so Needles_Check.py can access them
        self.embeddings = normalize(self._embed_fn([c.text for c in self.chunks]))
        
        if self.use_umap:
            print(f"Reducing dimensions (UMAP {self.umap_n_components})...")
            # CONTEXT ROT PARAMETER: Neighbors = 30
            target_neighbors = 30
            n_neighbors = min(target_neighbors, len(self.chunks) - 1)
            
            # CONTEXT ROT PARAMETER: min_dist = 0.05
            embeddings_reduced = umap.UMAP(
                n_components=self.umap_n_components, 
                n_neighbors=n_neighbors,
                min_dist=0.05,
                metric='cosine',
                random_state=42 
            ).fit_transform(self.embeddings)
        else: embeddings_reduced = self.embeddings
            
        print("Clustering...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size, 
            min_samples=self.min_samples
        )
        self.cluster_labels = clusterer.fit_predict(embeddings_reduced)
        
        unique_labels = set(self.cluster_labels) - {-1}
        stats_list = []
        print(f"Analyzing {len(unique_labels)} clusters...")
        
        for cid in unique_labels:
            s = self.compute_cluster_stats(cid, self.chunks, self.embeddings, self.cluster_labels)
            if s: stats_list.append(s)
            
        stats_list.sort(key=lambda x: x.score, reverse=True)
        self.ranked_stats = stats_list
        
        # Filter for display return (though we will use ranked_stats for export)
        return self._generate_report_df(stats_list[:3])

    def _generate_report_df(self, stats_list):
        data = []
        for s in stats_list:
            # Find a "Cluster Safe" word to show in summary
            safe_word = next((w.lemma for w in s.lexical_profile if w.is_cluster_safe), "None")
            data.append({
                'ID': s.cluster_id,
                'Score': round(s.score, 3),
                'Prev%': f"{s.prevalence*100:.1f}%",
                'Cohesion': round(s.cohesion, 2),
                'SafeWord': safe_word,
                'TopTerms': ", ".join(s.top_terms[:3])
            })
        return pd.DataFrame(data)

    def export_results_to_txt(self, filename: str):
        """
        Exports the report highlighting Cluster-Safe vs Global-Safe words.
        Restricted to Top 3 Clusters.
        """
        if not self.ranked_stats:
            print("No results to export.")
            return

        # FILTER: Only keep top 3
        top_stats = self.ranked_stats[:1]

        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"NIAH TOPIC & LEXICAL CAMOUFLAGE REPORT (Top 3 Only)\n")
            f.write("="*80 + "\n\n")
            
            f.write("LEGEND:\n")
            f.write("  ⭐ (Cluster Safe): High spread within this cluster. Good for local camouflage.\n")
            f.write("  🛡️ (Global Safe): High spread everywhere. Good for generic camouflage.\n")
            f.write("  📍 (Localized):   High freq in cluster, but clumped/rare globally.\n\n")

            f.write("TOPIC SUMMARY:\n")
            f.write("-" * 80 + "\n")
            df = self._generate_report_df(top_stats)
            f.write(df.to_string(index=False))
            f.write("\n\n" + "="*80 + "\n\n")

            f.write("DETAILED CLUSTER BREAKDOWNS\n")
            f.write("(Ordered by Score - Best to Worst)\n\n")

            for i, stats in enumerate(top_stats, 1):
                f.write(f"CLUSTER {stats.cluster_id} (Rank #{i})\n")
                f.write("-" * 40 + "\n")
                f.write(f"Score:      {stats.score:.3f}\n")
                f.write(f"Prevalence: {stats.prevalence * 100:.2f}%\n")
                f.write(f"Top Terms:  {', '.join(stats.top_terms)}\n\n")
                
                # --- NEW SECTION: REPRESENTATIVE EXAMPLES ---
                f.write("TYPICAL EXAMPLES (Representative Sentences/Chunks):\n")
                for idx, rep in enumerate(stats.representatives, 1):
                    # Truncate slightly if massive, but usually keep readable
                    clean_rep = rep.replace('\n', ' ').strip()
                    f.write(f"  {idx}. \"{clean_rep}\"\n")
                f.write("\n")
                # --------------------------------------------
                
                f.write("LEXICAL PROFILE (Top 50 common words in this cluster):\n")
                f.write(f"{'Flag':<4} {'Word':<15} {'ClustFreq':<10} {'ClustSpread':<12} {'GlobSpread':<12}\n")
                f.write("-" * 75 + "\n")

                for w in stats.lexical_profile:
                    flag = ""
                    if w.is_cluster_safe and w.is_global_safe:
                        flag = "⭐🛡️"
                    elif w.is_cluster_safe:
                        flag = "⭐"
                    elif w.is_global_safe:
                        flag = "🛡️"
                    else:
                        flag = "📍" # Topic marker/Localized
                    
                    f.write(f"{flag:<4} {w.lemma:<15} {w.cluster_count:<10} {w.cluster_spread:<12.3f} {w.global_spread:<12.3f}\n")
                
                f.write("\n" + "="*80 + "\n\n")
        
        print(f"Full report exported to: {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NIAH Topic Selector with Lexical Analysis")
    parser.add_argument("filename", nargs='?', help="Path to the haystack .txt file")
    parser.add_argument("--output", default="topic_analysis_results.txt", help="Output .txt file name")
    
    args = parser.parse_args()

    file_path = args.filename
    if not file_path:
        # Fallback for testing/default
        default_file = "CognitiveBiasHaystack.txt"
        if os.path.exists(default_file):
            print(f"No file argument. Using default: {default_file}")
            file_path = default_file
        else:
            print("Usage: python Topic.py <path_to_text_file.txt>")
            sys.exit(1)

    print(f"Reading haystack from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            haystack_text = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    # Initialize
    selector = NIAHLexicalTopicSelector(
        embedding_model="all-MiniLM-L6-v2",
        embedding_provider="sentence-transformers",
        chunk_size=256,
        chunk_overlap=50,
        min_cluster_size=10,
        use_umap=True
    )
    
    # Run
    results_df = selector.run_pipeline(haystack_text)
    
    # Export Results
    selector.export_results_to_txt(args.output)
    
    # Print Summary
    print("\nSummary Results:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(results_df)