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
import pickle
import gc

from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# --- CONFIGURATION ---

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
        "trust_remote": False, # Changed to False as BGE-M3 usually loads fine natively now
        "dims": 1024
    }
]

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
    global_count: int
    global_spread: float
    cluster_count: int
    cluster_spread: float
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
    CPU Mode Forced.
    """
    
    def __init__(
        self,
        model_config: Dict,
        chunk_size: int = 256,
        chunk_overlap: int = 50,
        min_cluster_size: int = 10,
        min_samples: int = 15,
        use_umap: bool = True,
        umap_n_components: int = 50 
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.use_umap = use_umap
        self.umap_n_components = umap_n_components
        
        # Load Config
        self.model_name = model_config["name"]
        self.model_prefix = model_config["prefix"]
        self.trust_remote = model_config["trust_remote"]
        
        # --- FORCED CPU ---
        self.device = "cpu"
        # ------------------

        self.ranked_stats = []
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
            
        self.nlp_disable_pipes = [pipe for pipe in self.nlp.pipe_names 
                                 if pipe not in ['tok2vec', 'tagger', 'attribute_ruler', 'lemmatizer']]

        print(f"Loading embedding model: {self.model_name} on {self.device.upper()}...")
        
        # --- MODEL LOADING ---
        # We removed the 'model_kwargs={"use_safetensors": True}' to avoid the OS Error.
        # The library will now automatically choose the best available file.
        self.encoder = SentenceTransformer(
            self.model_name, 
            trust_remote_code=self.trust_remote,
            device=self.device
        )
        
        self.common_words = self._load_common_words()
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
        print("Running Global Lexical Analysis...")
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

    def embed_chunks(self, texts: List[str]) -> np.ndarray:
        if self.model_prefix:
            print(f"Applying prefix: '{self.model_prefix}'")
            texts_to_embed = [self.model_prefix + t for t in texts]
        else:
            texts_to_embed = texts
            
        print(f"Embedding {len(texts)} chunks with {self.model_name}...")
        
        # Normalize is TRUE to match standard practice and Topic_New.py logic
        embeddings = self.encoder.encode(
            texts_to_embed, 
            show_progress_bar=True, 
            batch_size=32,
            normalize_embeddings=True 
        )
        return embeddings

    def get_cluster_lexical_profile(self, cluster_chunks: List[Chunk], top_n=150) -> List[WordLexicalStats]:
        cluster_text = " ".join([c.text for c in cluster_chunks])
        
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
            counts_per_chunk = []
            for chunk in cluster_chunks:
                count = len(re.findall(r'\b' + re.escape(lemma) + r'\b', chunk.text.lower()))
                counts_per_chunk.append(count)
            
            cluster_spread = self.calculate_juillands_d(counts_per_chunk)
            
            g_stats = self.global_word_stats.get(lemma, {'count': 0, 'spread': 0.0, 'pos': 'UNK'})
            
            is_global_safe = (g_stats['spread'] > 0.6) and (g_stats['count'] > 50)
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
        
        prevalence = len(cluster_chunks) / len(chunks)
        
        if len(cluster_embeddings) < 2: cohesion = 1.0
        else:
            sim = cosine_similarity(cluster_embeddings)
            cohesion = float(np.mean(sim[np.triu_indices_from(sim, k=1)]))
            
        positions = [c.position for c in cluster_chunks]
        norm_pos = np.array(positions) / (chunks[-1].position + 1)
        spread = min(np.std(norm_pos) / 0.289, 1.0) if len(positions) > 1 else 0.0
        
        lexical_profile = self.get_cluster_lexical_profile(cluster_chunks, top_n=150)
        
        vectorizer = TfidfVectorizer(max_features=10, stop_words='english')
        try:
            vectorizer.fit_transform(cluster_texts)
            top_terms = vectorizer.get_feature_names_out().tolist()
        except: top_terms = []
        
        uncommon = sum(1 for t in top_terms if t not in self.common_words)
        jargon = uncommon / len(top_terms) if top_terms else 0
        
        ner_sample = cluster_texts[:20]
        ne_count, token_count = 0, 0
        for txt in ner_sample:
            doc = self.nlp(txt)
            ne_count += len(doc.ents)
            token_count += len(doc)
        ner_density = (ne_count / token_count * 100) if token_count else 0
        
        centroid = cluster_embeddings.mean(axis=0, keepdims=True)
        relevance = cosine_similarity(cluster_embeddings, centroid).flatten()
        top_idx = np.argsort(relevance)[::-1][:10]
        representatives = [cluster_texts[i] for i in top_idx]
        
        score = (0.3 * prevalence + 0.3 * cohesion + 0.4 * spread - 0.1 * jargon)
        
        return ClusterStats(
            cluster_id=cluster_id, prevalence=prevalence, cohesion=cohesion,
            jargon=jargon, spread=spread,
            ner_density=ner_density, top_terms=top_terms,
            lexical_profile=lexical_profile,
            score=score, representatives=representatives
        )

    def run_pipeline(self, text: str):
        print(f"\n--- Running Pipeline for {self.model_name} ---")
        
        if not self.global_word_stats:
            self.analyze_global_lexicon(text)
        
        if not self.chunks:
            self.chunks = self.chunk_text(text)
            print(f"Total Chunks: {len(self.chunks)}")
        
        self.embeddings = self.embed_chunks([c.text for c in self.chunks])
        
        if self.use_umap:
            print(f"Reducing dimensions (UMAP {self.umap_n_components})...")
            target_neighbors = 30
            n_neighbors = min(target_neighbors, len(self.chunks) - 1)
            
            embeddings_reduced = umap.UMAP(
                n_components=self.umap_n_components, 
                n_neighbors=n_neighbors,
                min_dist=0.05,
                metric='cosine',
                random_state=42 
            ).fit_transform(self.embeddings)
        else: embeddings_reduced = self.embeddings
            
        print("Clustering (HDBSCAN)...")
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
        
        return self._generate_report_df(stats_list[:3])

    def _generate_report_df(self, stats_list):
        data = []
        for s in stats_list:
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
        if not self.ranked_stats:
            print("No results to export.")
            return

        top_stats = self.ranked_stats[:3] # Export Top 3

        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"NIAH TOPIC REPORT - MODEL: {self.model_name}\n")
            f.write("="*80 + "\n\n")
            
            f.write("TOPIC SUMMARY:\n")
            f.write("-" * 80 + "\n")
            df = self._generate_report_df(top_stats)
            f.write(df.to_string(index=False))
            f.write("\n\n" + "="*80 + "\n\n")

            for i, stats in enumerate(top_stats, 1):
                f.write(f"CLUSTER {stats.cluster_id} (Rank #{i})\n")
                f.write("-" * 40 + "\n")
                f.write(f"Score:      {stats.score:.3f}\n")
                f.write(f"Prevalence: {stats.prevalence * 100:.2f}%\n")
                f.write(f"Top Terms:  {', '.join(stats.top_terms)}\n\n")
                
                f.write("TYPICAL EXAMPLES:\n")
                for idx, rep in enumerate(stats.representatives[:3], 1):
                    clean_rep = rep.replace('\n', ' ').strip()[:200] + "..."
                    f.write(f"  {idx}. \"{clean_rep}\"\n")
                f.write("\n")
                
                f.write("LEXICAL PROFILE (Top Cluster-Safe Words):\n")
                f.write(f"{'Flag':<4} {'Word':<15} {'ClustFreq':<10} {'ClustSpread':<12} {'GlobSpread':<12}\n")
                f.write("-" * 75 + "\n")

                count = 0
                for w in stats.lexical_profile:
                    if count >= 15: break
                    flag = ""
                    if w.is_cluster_safe and w.is_global_safe: flag = "⭐🛡️"
                    elif w.is_cluster_safe: flag = "⭐"
                    elif w.is_global_safe: flag = "🛡️"
                    else: flag = "📍" 
                    
                    f.write(f"{flag:<4} {w.lemma:<15} {w.cluster_count:<10} {w.cluster_spread:<12.3f} {w.global_spread:<12.3f}\n")
                    count += 1
                
                f.write("\n" + "="*80 + "\n\n")
        
        print(f"Report exported to: {filename}")

    def save_analysis(self, filename: str):
        print(f"Saving binary state to {filename}...")
        state = {
            'chunks': self.chunks,
            'embeddings': self.embeddings,
            'cluster_labels': self.cluster_labels,
            'ranked_stats': self.ranked_stats,
            'global_word_stats': self.global_word_stats,
            'model_name': self.model_name
        }
        with open(filename, 'wb') as f:
            pickle.dump(state, f)

# --- MAIN EXECUTION LOOP ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Model NIAH Topic Selector")
    parser.add_argument("filename", nargs='?', help="Path to the haystack .txt file")
    
    args = parser.parse_args()

    file_path = args.filename
    if not file_path:
        default_file = "CognitiveBiasHaystack.txt"
        if os.path.exists(default_file):
            print(f"No file argument. Using default: {default_file}")
            file_path = default_file
        else:
            print("Usage: python Topic_MultiModel.py <path_to_text_file.txt>")
            sys.exit(1)
            
    print("\n" + "="*60)
    print("🚀 RUNNING ON CPU MODE")
    print("="*60)

    print(f"Reading haystack from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            haystack_text = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    print(f"\nSTARTING MULTI-MODEL ANALYSIS")
    print(f"Models to run: {len(MODEL_CONFIGS)}\n")

    for config in MODEL_CONFIGS:
        
        safe_name = config['name'].split('/')[-1]
        txt_output = f"topic_report_{safe_name}.txt"
        pkl_output = f"analysis_state_{safe_name}.pkl"
        
        # --- SKIP LOGIC ---
        if os.path.exists(pkl_output):
            print(f"✅ Found existing analysis for {safe_name} ({pkl_output}). Skipping...")
            continue
        # ------------------

        gc.collect()
        
        selector = NIAHLexicalTopicSelector(
            model_config=config,
            chunk_size=256,
            chunk_overlap=50,
            min_cluster_size=10,
            use_umap=True
        )
        
        results_df = selector.run_pipeline(haystack_text)
        
        print(f"Top Cluster for {safe_name}:")
        print(results_df.head(1))
        
        selector.export_results_to_txt(txt_output)
        selector.save_analysis(pkl_output)
        
        del selector
        print(f"Finished {safe_name}. Moving to next...\n")

    print("\n" + "="*60)
    print("ALL MODELS COMPLETED SUCCESSFULLY")
    print("="*60)