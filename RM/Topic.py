import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import re
from collections import Counter
import spacy

# For embeddings and clustering
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

@dataclass
class Chunk:
    chunk_id: int
    text: str
    position: int  # Position in document (for spread calculation)
    token_count: int

@dataclass
class ClusterStats:
    cluster_id: int
    prevalence: float
    cohesion: float
    jargon: float
    redundancy: float
    spread: float  # New: how distributed across document
    ner_density: float  # New: named entity density
    top_terms: List[str]
    score: float
    representatives: List[str]

class NIAHTopicSelector:
    """
    Pipeline for selecting optimal topics from a haystack document for NIAH testing.
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedding_provider: str = "sentence-transformers",  # or "openai", "cohere", "gemini"
        api_key: Optional[str] = None,
        chunk_size: int = 256,
        chunk_overlap: int = 50,
        min_cluster_size: int = 10,
        min_samples: int = 10,
        use_umap: bool = True,
        umap_n_components: int = 32
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.use_umap = use_umap
        self.umap_n_components = umap_n_components
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        
        # Load embedding model based on provider
        print(f"Loading embedding model: {embedding_provider}/{embedding_model}...")
        if embedding_provider == "sentence-transformers":
            self.encoder = SentenceTransformer(embedding_model)
            self._embed_fn = self._embed_sentence_transformers
        elif embedding_provider == "openai":
            if not api_key:
                raise ValueError("OpenAI API key required! Set api_key parameter or OPENAI_API_KEY env var")
            import openai
            openai.api_key = api_key
            self.openai_client = openai
            self._embed_fn = self._embed_openai
        elif embedding_provider == "cohere":
            if not api_key:
                raise ValueError("Cohere API key required! Set api_key parameter or COHERE_API_KEY env var")
            import cohere
            self.cohere_client = cohere.Client(api_key)
            self._embed_fn = self._embed_cohere
        elif embedding_provider == "gemini":
            if not api_key:
                raise ValueError("Gemini API key required! Set api_key parameter or GOOGLE_API_KEY env var")
            from google import genai as google_genai
            self.gemini_client = google_genai.Client(api_key=api_key)
            self._embed_fn = self._embed_gemini
        else:
            raise ValueError(f"Unknown provider: {embedding_provider}")
        
        print("Loading spaCy model for NER...")
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spaCy model...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        # Common words for jargon detection (top 5000 English words)
        self.common_words = self._load_common_words()
        
    def _load_common_words(self) -> set:
        """Load common English words for jargon detection."""
        # Basic set - you can replace with a file load if you have a word list
        common = set([
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
            'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
            # Add more common words as needed
        ])
        return common
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\:\;]', '', text)
        return text.strip()
    
    def chunk_text(self, text: str) -> List[Chunk]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Input document text
            
        Returns:
            List of Chunk objects
        """
        text = self.preprocess_text(text)
        words = text.split()
        chunks = []
        
        position = 0
        chunk_id = 0
        
        while position < len(words):
            chunk_words = words[position:position + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append(Chunk(
                chunk_id=chunk_id,
                text=chunk_text,
                position=position,
                token_count=len(chunk_words)
            ))
            
            position += self.chunk_size - self.chunk_overlap
            chunk_id += 1
        
        print(f"Created {len(chunks)} chunks")
        return chunks
    
    def _embed_sentence_transformers(self, texts: List[str]) -> np.ndarray:
        """Embed using sentence-transformers (local)."""
        embeddings = self.encoder.encode(texts, show_progress_bar=True)
        return embeddings
    
    def _embed_openai(self, texts: List[str]) -> np.ndarray:
        """Embed using OpenAI API."""
        print(f"Calling OpenAI API for {len(texts)} chunks...")
        embeddings = []
        
        # Batch requests (OpenAI allows up to 2048 inputs per request)
        batch_size = 2048
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,  # e.g., "text-embedding-3-small"
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
            print(f"  Processed {min(i + batch_size, len(texts))}/{len(texts)} chunks")
        
        return np.array(embeddings)
    
    def _embed_cohere(self, texts: List[str]) -> np.ndarray:
        """Embed using Cohere API."""
        print(f"Calling Cohere API for {len(texts)} chunks...")
        
        # Cohere allows up to 96 texts per request
        batch_size = 96
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.cohere_client.embed(
                texts=batch,
                model=self.embedding_model,  # e.g., "embed-english-v3.0"
                input_type="search_document"
            )
            embeddings.extend(response.embeddings)
            print(f"  Processed {min(i + batch_size, len(texts))}/{len(texts)} chunks")
        
        return np.array(embeddings)
    
    def _embed_gemini(self, texts: List[str]) -> np.ndarray:
        """Embed using Google Gemini API."""
        from google import genai
        
        print(f"Calling Gemini API for {len(texts)} chunks...")
        embeddings = []
        
        # Gemini allows batch embedding
        # Process in batches for rate limiting
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            result = genai.embed_content(
                model=self.embedding_model,  # e.g., "models/embedding-001"
                content=batch,
                task_type="retrieval_document"
            )
            embeddings.extend(result['embedding'])
            print(f"  Processed {min(i + batch_size, len(texts))}/{len(texts)} chunks")
        
        return np.array(embeddings)
    
    def embed_chunks(self, chunks: List[Chunk]) -> np.ndarray:
        """
        Generate embeddings for all chunks.
        
        Args:
            chunks: List of Chunk objects
            
        Returns:
            Normalized embedding matrix (n_chunks x embedding_dim)
        """
        print("Generating embeddings...")
        texts = [chunk.text for chunk in chunks]
        
        # Call appropriate embedding function
        embeddings = self._embed_fn(texts)
        
        # L2 normalize
        embeddings = normalize(embeddings, norm='l2')
        
        return embeddings
    
    def reduce_dimensions(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Apply UMAP for dimensionality reduction.
        
        Args:
            embeddings: High-dimensional embeddings
            
        Returns:
            Reduced embeddings
        """
        if not self.use_umap:
            return embeddings
        
        print("Reducing dimensions with UMAP...")
        reducer = umap.UMAP(
            n_components=self.umap_n_components,
            n_neighbors=30,
            min_dist=0.05,
            metric='cosine',
            random_state=42
        )
        
        reduced = reducer.fit_transform(embeddings)
        return reduced
    
    def cluster_chunks(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Cluster embeddings using HDBSCAN.
        
        Args:
            embeddings: Embedding matrix
            
        Returns:
            Cluster assignments (n_chunks,)
        """
        print("Clustering with HDBSCAN...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        
        cluster_labels = clusterer.fit_predict(embeddings)
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        print(f"Found {n_clusters} clusters, {n_noise} noise points")
        return cluster_labels
    
    def compute_cohesion(self, cluster_embeddings: np.ndarray) -> float:
        """
        Compute average pairwise cosine similarity within cluster.
        
        Args:
            cluster_embeddings: Embeddings for chunks in the cluster
            
        Returns:
            Cohesion score (0-1)
        """
        if len(cluster_embeddings) < 2:
            return 1.0
        
        sim_matrix = cosine_similarity(cluster_embeddings)
        # Get upper triangle (excluding diagonal)
        triu_indices = np.triu_indices_from(sim_matrix, k=1)
        similarities = sim_matrix[triu_indices]
        
        return float(np.mean(similarities))
    
    def compute_jargon_score(self, texts: List[str], top_terms: List[str]) -> float:
        """
        Compute jargon score based on term rarity and domain specificity.
        
        Args:
            texts: Chunk texts in cluster
            top_terms: Top TF-IDF terms
            
        Returns:
            Jargon score (higher = more jargon)
        """
        # Check how many top terms are NOT in common words
        uncommon_count = sum(1 for term in top_terms if term.lower() not in self.common_words)
        uncommon_ratio = uncommon_count / len(top_terms) if top_terms else 0
        
        # Average word length (technical terms tend to be longer)
        all_words = ' '.join(texts).split()
        avg_word_len = np.mean([len(w) for w in all_words]) if all_words else 0
        word_len_score = min(avg_word_len / 10.0, 1.0)  # Normalize to 0-1
        
        # Combined jargon score
        jargon = 0.7 * uncommon_ratio + 0.3 * word_len_score
        
        return float(jargon)
    
    def compute_redundancy(self, texts: List[str]) -> float:
        """
        Compute redundancy via n-gram overlap among texts.
        
        Args:
            texts: Sample of chunk texts
            
        Returns:
            Redundancy score (0-1, higher = more redundant)
        """
        if len(texts) < 2:
            return 0.0
        
        # Use trigrams
        def get_ngrams(text, n=3):
            words = text.lower().split()
            return set(zip(*[words[i:] for i in range(n)]))
        
        ngrams_list = [get_ngrams(text) for text in texts]
        
        # Average Jaccard similarity between all pairs
        similarities = []
        for i in range(len(ngrams_list)):
            for j in range(i + 1, len(ngrams_list)):
                if len(ngrams_list[i]) == 0 or len(ngrams_list[j]) == 0:
                    continue
                intersection = len(ngrams_list[i] & ngrams_list[j])
                union = len(ngrams_list[i] | ngrams_list[j])
                similarities.append(intersection / union if union > 0 else 0)
        
        return float(np.mean(similarities)) if similarities else 0.0
    
    def compute_spread(self, chunk_positions: List[int], total_chunks: int) -> float:
        """
        Compute how spread out the topic is across the document.
        
        Args:
            chunk_positions: Positions of chunks in this cluster
            total_chunks: Total number of chunks in document
            
        Returns:
            Spread score (0-1, higher = more spread out)
        """
        if len(chunk_positions) < 2:
            return 0.0
        
        # Normalize positions to 0-1
        normalized_positions = np.array(chunk_positions) / total_chunks
        
        # Standard deviation of positions (higher = more spread)
        std = np.std(normalized_positions)
        
        # Normalize to 0-1 (max std of uniform distribution is ~0.289)
        spread = min(std / 0.289, 1.0)
        
        return float(spread)
    
    def compute_ner_density(self, texts: List[str]) -> float:
        """
        Compute named entity density (to avoid flashy topics).
        
        Args:
            texts: Chunk texts
            
        Returns:
            NER density (entities per 100 tokens)
        """
        total_entities = 0
        total_tokens = 0
        
        # Sample up to 20 texts to avoid slow processing
        sample_texts = texts[:20]
        
        for text in sample_texts:
            doc = self.nlp(text)
            total_entities += len(doc.ents)
            total_tokens += len(doc)
        
        if total_tokens == 0:
            return 0.0
        
        # Entities per 100 tokens
        ner_density = (total_entities / total_tokens) * 100
        
        return float(ner_density)
    
    def extract_top_terms(self, texts: List[str], n_terms: int = 10) -> List[str]:
        """
        Extract top TF-IDF terms for cluster.
        
        Args:
            texts: Chunk texts in cluster
            
        Returns:
            List of top terms
        """
        if not texts:
            return []
        
        vectorizer = TfidfVectorizer(
            max_features=n_terms,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get average TF-IDF score for each term
            avg_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
            top_indices = avg_scores.argsort()[-n_terms:][::-1]
            
            return [feature_names[i] for i in top_indices]
        except:
            return []
    
    def select_representatives_mmr(
        self,
        cluster_embeddings: np.ndarray,
        cluster_texts: List[str],
        n_reps: int = 15,
        lambda_param: float = 0.7
    ) -> List[str]:
        """
        Select representative chunks using Maximal Marginal Relevance.
        
        Args:
            cluster_embeddings: Embeddings for chunks in cluster
            cluster_texts: Texts for chunks in cluster
            n_reps: Number of representatives to select
            lambda_param: Trade-off between relevance and diversity (0-1)
            
        Returns:
            List of representative texts
        """
        if len(cluster_texts) <= n_reps:
            return cluster_texts
        
        # Centroid
        centroid = cluster_embeddings.mean(axis=0, keepdims=True)
        
        # Relevance: similarity to centroid
        relevance = cosine_similarity(cluster_embeddings, centroid).flatten()
        
        selected_indices = []
        remaining_indices = list(range(len(cluster_texts)))
        
        # Select first: most relevant
        first_idx = np.argmax(relevance)
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        # Select rest using MMR
        for _ in range(n_reps - 1):
            if not remaining_indices:
                break
            
            mmr_scores = []
            selected_embeddings = cluster_embeddings[selected_indices]
            
            for idx in remaining_indices:
                # Relevance to centroid
                rel = relevance[idx]
                
                # Max similarity to already selected
                sim_to_selected = cosine_similarity(
                    cluster_embeddings[idx:idx+1],
                    selected_embeddings
                ).max()
                
                # MMR score
                mmr = lambda_param * rel - (1 - lambda_param) * sim_to_selected
                mmr_scores.append(mmr)
            
            # Select best MMR score
            best_idx = remaining_indices[np.argmax(mmr_scores)]
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        return [cluster_texts[i] for i in selected_indices]
    
    def compute_cluster_stats(
        self,
        cluster_id: int,
        chunks: List[Chunk],
        embeddings: np.ndarray,
        cluster_labels: np.ndarray
    ) -> Optional[ClusterStats]:
        """
        Compute comprehensive statistics for a cluster.
        
        Args:
            cluster_id: ID of cluster
            chunks: All chunks
            embeddings: All embeddings (original, not reduced)
            cluster_labels: Cluster assignment for each chunk
            
        Returns:
            ClusterStats object or None if invalid cluster
        """
        # Skip noise cluster
        if cluster_id == -1:
            return None
        
        # Get chunks in this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_chunks = [c for c, m in zip(chunks, cluster_mask) if m]
        cluster_embeddings = embeddings[cluster_mask]
        cluster_texts = [c.text for c in cluster_chunks]
        cluster_positions = [c.position for c in cluster_chunks]
        
        if len(cluster_chunks) == 0:
            return None
        
        # Compute metrics
        prevalence = len(cluster_chunks) / len(chunks)
        cohesion = self.compute_cohesion(cluster_embeddings)
        spread = self.compute_spread(cluster_positions, len(chunks))
        top_terms = self.extract_top_terms(cluster_texts)
        jargon = self.compute_jargon_score(cluster_texts, top_terms)
        ner_density = self.compute_ner_density(cluster_texts)
        
        # Select representatives
        representatives = self.select_representatives_mmr(
            cluster_embeddings,
            cluster_texts,
            n_reps=min(15, len(cluster_texts))
        )
        
        redundancy = self.compute_redundancy(representatives)
        
        # Compute composite score
        # Normalize NER density (lower is better, typical range 0-20)
        ner_penalty = min(ner_density / 20.0, 1.0)
        
        score = (
            0.30 * prevalence +
            0.30 * cohesion +
            0.20 * spread +
            -0.10 * jargon +
            -0.05 * redundancy +
            -0.05 * ner_penalty
        )
        
        return ClusterStats(
            cluster_id=cluster_id,
            prevalence=prevalence,
            cohesion=cohesion,
            jargon=jargon,
            redundancy=redundancy,
            spread=spread,
            ner_density=ner_density,
            top_terms=top_terms,
            score=score,
            representatives=representatives
        )
    
    def filter_and_rank(
        self,
        cluster_stats: List[ClusterStats],
        min_prevalence: float = 0.02,
        max_prevalence: float = 0.20
    ) -> List[ClusterStats]:
        """
        Filter clusters by prevalence and rank by score.
        
        Args:
            cluster_stats: List of cluster statistics
            min_prevalence: Minimum prevalence threshold
            max_prevalence: Maximum prevalence threshold
            
        Returns:
            Filtered and sorted list of cluster stats
        """
        # Filter by prevalence
        filtered = [
            stats for stats in cluster_stats
            if min_prevalence <= stats.prevalence <= max_prevalence
        ]
        
        # Sort by score (descending)
        ranked = sorted(filtered, key=lambda x: x.score, reverse=True)
        
        return ranked
    
    def run_pipeline(
        self,
        text: str,
        min_prevalence: float = 0.02,
        max_prevalence: float = 0.20
    ) -> pd.DataFrame:
        """
        Run the complete topic selection pipeline.
        
        Args:
            text: Input document text
            min_prevalence: Minimum topic prevalence
            max_prevalence: Maximum topic prevalence
            
        Returns:
            DataFrame with ranked topics and their statistics
        """
        print("=" * 60)
        print("NIAH TOPIC SELECTION PIPELINE")
        print("=" * 60)
        
        # Step 1: Chunk
        chunks = self.chunk_text(text)
        
        # Step 2: Embed
        embeddings = self.embed_chunks(chunks)
        original_embeddings = embeddings.copy()  # Keep for final stats
        
        # Step 3: Reduce dimensions (optional)
        if self.use_umap:
            embeddings = self.reduce_dimensions(embeddings)
        
        # Step 4: Cluster
        cluster_labels = self.cluster_chunks(embeddings)
        
        # Step 5: Compute stats for each cluster
        print("\nComputing cluster statistics...")
        unique_clusters = set(cluster_labels)
        unique_clusters.discard(-1)  # Remove noise
        
        all_stats = []
        for cluster_id in sorted(unique_clusters):
            stats = self.compute_cluster_stats(
                cluster_id,
                chunks,
                original_embeddings,  # Use original embeddings
                cluster_labels
            )
            if stats:
                all_stats.append(stats)
        
        # Step 6: Filter and rank
        print("\nFiltering and ranking topics...")
        ranked_stats = self.filter_and_rank(all_stats, min_prevalence, max_prevalence)
        
        # Convert to DataFrame for easy viewing
        df_data = []
        for stats in ranked_stats:
            df_data.append({
                'cluster_id': stats.cluster_id,
                'score': round(stats.score, 3),
                'prevalence_%': round(stats.prevalence * 100, 2),
                'cohesion': round(stats.cohesion, 3),
                'spread': round(stats.spread, 3),
                'jargon': round(stats.jargon, 3),
                'redundancy': round(stats.redundancy, 3),
                'ner_density': round(stats.ner_density, 2),
                'top_terms': ', '.join(stats.top_terms[:5]),
                'n_representatives': len(stats.representatives)
            })
        
        df = pd.DataFrame(df_data)
        
        print("\n" + "=" * 60)
        print(f"RESULTS: Found {len(ranked_stats)} candidate topics")
        print("=" * 60)
        
        # Store for later access
        self.embeddings = original_embeddings  # <-- ADD THIS LINE
        self.cluster_labels = cluster_labels  # <-- ADD THIS LINE
        self.ranked_stats = ranked_stats
        
        return df
    
    def get_representatives(self, cluster_id: int) -> List[str]:
        """
        Get representative texts for a specific cluster.
        
        Args:
            cluster_id: Cluster ID
            
        Returns:
            List of representative texts
        """
        if not hasattr(self, 'ranked_stats'):
            raise ValueError("Run pipeline first!")
        
        for stats in self.ranked_stats:
            if stats.cluster_id == cluster_id:
                return stats.representatives
        
        return []
    
    def print_cluster_details(self, cluster_id: int, n_examples: int = 5):
        """
        Print detailed information about a specific cluster.
        
        Args:
            cluster_id: Cluster ID
            n_examples: Number of example texts to show
        """
        if not hasattr(self, 'ranked_stats'):
            raise ValueError("Run pipeline first!")
        
        for stats in self.ranked_stats:
            if stats.cluster_id == cluster_id:
                print(f"\n{'=' * 60}")
                print(f"CLUSTER {cluster_id} DETAILS")
                print(f"{'=' * 60}")
                print(f"Score: {stats.score:.3f}")
                print(f"Prevalence: {stats.prevalence * 100:.2f}%")
                print(f"Cohesion: {stats.cohesion:.3f}")
                print(f"Spread: {stats.spread:.3f}")
                print(f"Jargon: {stats.jargon:.3f}")
                print(f"Redundancy: {stats.redundancy:.3f}")
                print(f"NER Density: {stats.ner_density:.2f}")
                print(f"\nTop Terms: {', '.join(stats.top_terms)}")
                print(f"\nExample Representatives (showing {n_examples}):")
                print("-" * 60)
                for i, text in enumerate(stats.representatives[:n_examples], 1):
                    print(f"\n[{i}] {text[:300]}...")
                print("\n" + "=" * 60)
                return
        
        print(f"Cluster {cluster_id} not found!")


# Example usage
if __name__ == "__main__":
    # Sample text (replace with your actual haystack)
    with open('CognitiveBias4.txt', 'r', encoding='utf-8') as f:
        haystack_text = f.read()

    # OPTION 1: Local embedding (FREE, no API key)
    selector = NIAHTopicSelector(
        embedding_model="all-MiniLM-L6-v2",
        embedding_provider="sentence-transformers",
        chunk_size=256,
        chunk_overlap=50,
        min_cluster_size=10,
        use_umap=True
    )
    
    # OPTION 2: OpenAI embeddings (PAID, requires API key)
    # selector = NIAHTopicSelector(
    #     embedding_model="text-embedding-3-small",  # or "text-embedding-3-large"
    #     embedding_provider="openai",
    #     api_key="your-openai-api-key-here",  # or set OPENAI_API_KEY env var
    #     chunk_size=256,
    #     chunk_overlap=50,
    #     min_cluster_size=10,
    #     use_umap=True
    # )
    
    # OPTION 3: Cohere embeddings (PAID, requires API key)
    # selector = NIAHTopicSelector(
    #     embedding_model="embed-english-v3.0",
    #     embedding_provider="cohere",
    #     api_key="your-cohere-api-key-here",  # or set COHERE_API_KEY env var
    #     chunk_size=256,
    #     chunk_overlap=50,
    #     min_cluster_size=10,
    #     use_umap=True
    # )
    
    # OPTION 4: Gemini embeddings (FREE tier available, requires API key)
    # selector = NIAHTopicSelector(
    #     embedding_model="models/embedding-001",
    #     embedding_provider="gemini",
    #     api_key="your-gemini-api-key-here",  # or set GOOGLE_API_KEY env var
    #     chunk_size=256,
    #     chunk_overlap=50,
    #     min_cluster_size=10,
    #     use_umap=True
    # )
    
    # Run pipeline
    results_df = selector.run_pipeline(
        haystack_text,
        min_prevalence=0.02,
        max_prevalence=0.20
    )
    if hasattr(selector, 'ranked_stats') and selector.ranked_stats:
    
        # Get the top 3 clusters (which are just the first 3 items in the list)
        top_3_clusters = selector.ranked_stats[:3]
        
        print("\n" + "=" * 60)
        print("DISPLAYING TOP 10 CHUNKS FROM TOP 3 CLUSTERS")
        print("=" * 60)

        # Loop through each of the top 3 clusters
        for i, stats in enumerate(top_3_clusters, 1):
            
            cluster_id = stats.cluster_id
            score = stats.score
            top_terms = ", ".join(stats.top_terms[:5])

            print(f"\n\n--- Cluster #{i} (ID: {cluster_id}) ---")
            print(f"--- Score: {score:.3f} | Top Terms: {top_terms} ---")
            
            # Get the top 10 representative chunks for this cluster
            # The 'representatives' list is already curated by MMR
            top_10_chunks = stats.representatives[:10]
            
            # Print each chunk
            for j, chunk_text in enumerate(top_10_chunks, 1):
                print(f"\n[{j}] {chunk_text[:300]}...") # Print first 300 chars
                
    else:
        print("You must run the pipeline first! Call selector.run_pipeline(text)")
    # Display results
    print("\nTop Candidate Topics:")
    print(results_df.to_string(index=False))
    
    # Inspect top cluster in detail
    if len(results_df) > 0:
        top_cluster_id = results_df.iloc[0]['cluster_id']
        selector.print_cluster_details(top_cluster_id, n_examples=5)