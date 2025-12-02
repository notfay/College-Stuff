"""
Lexical Camouflage: Adversarial Needle Constructor for NIAH Evaluations

This script helps construct needles that are lexically indistinguishable from 
the haystack, forcing LLMs to use semantic reasoning rather than pattern matching.

Author: NIAH Evaluation Framework
Purpose: Eliminate lexical overlap bias in long-context evaluations
"""

import spacy
from collections import Counter, defaultdict
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass, asdict
import re

@dataclass
class WordStatistics:
    """Statistics for a single word in the haystack"""
    word: str
    lemma: str
    pos: str
    total_count: int
    segments_present: int
    total_segments: int
    dispersion_score: float  # 0-1, higher = more evenly distributed
    frequency_rank: int
    is_safe: bool

@dataclass
class NeedleAnalysis:
    """Analysis results for a candidate needle"""
    needle_text: str
    safety_score: float  # 0-100
    total_words: int
    safe_words: int
    danger_words: List[Dict]
    recommendations: List[str]


class LexicalCamouflage:
    """
    Main class for analyzing haystack vocabulary and validating needles
    """
    
    def __init__(self, haystack_path: str, n_segments: int = 20):
        """
        Initialize the analyzer
        
        Args:
            haystack_path: Path to the haystack text file
            n_segments: Number of segments to divide haystack into (default: 20)
        """
        self.haystack_path = Path(haystack_path)
        self.n_segments = n_segments
        self.nlp = None
        self.word_stats: Dict[str, WordStatistics] = {}
        self.safe_words: Dict[str, List[str]] = {'NOUN': [], 'VERB': [], 'ADJ': []}
        self.total_tokens = 0
        
        print("Loading spaCy model (this may take a moment)...")
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("spaCy model not found. Downloading 'en_core_web_sm'...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        # Optimize spaCy for speed - we only need POS tagging and lemmatization
        self.nlp.disable_pipes([pipe for pipe in self.nlp.pipe_names 
                                if pipe not in ['tok2vec', 'tagger', 'attribute_ruler', 'lemmatizer']])
    
    def calculate_dispersion(self, segment_counts: List[int]) -> float:
        """
        Calculate Juilland's D dispersion coefficient
        
        This measures how evenly a word is distributed across segments.
        D = 1 means perfectly even distribution
        D = 0 means all occurrences in one segment
        
        Args:
            segment_counts: List of counts per segment
            
        Returns:
            Dispersion score between 0 and 1
        """
        n = len(segment_counts)
        total = sum(segment_counts)
        
        if total == 0 or n == 0:
            return 0.0
        
        # Calculate variance
        mean = total / n
        variance = sum((count - mean) ** 2 for count in segment_counts) / n
        
        # Juilland's D formula
        if mean == 0:
            return 0.0
        
        cv = np.sqrt(variance) / mean  # Coefficient of variation
        d = 1 - (cv / np.sqrt(n - 1))  # Juilland's D
        
        return max(0.0, min(1.0, d))  # Clamp to [0, 1]
    
    def analyze_haystack(self, min_frequency: int = 10, min_dispersion: float = 0.5):
        """
        Analyze the haystack to build vocabulary and dispersion maps
        
        Args:
            min_frequency: Minimum word count to be considered "common"
            min_dispersion: Minimum dispersion score to be considered "well-distributed"
        """
        print(f"\n{'='*70}")
        print(f"PHASE 1: HAYSTACK ANALYSIS")
        print(f"{'='*70}\n")
        
        # Read haystack
        print(f"Reading haystack from: {self.haystack_path}")
        with open(self.haystack_path, 'r', encoding='utf-8') as f:
            haystack_text = f.read()
        
        print(f"Haystack size: {len(haystack_text):,} characters")
        
        # Divide into segments
        segment_size = len(haystack_text) // self.n_segments
        segments = [haystack_text[i*segment_size:(i+1)*segment_size] 
                   for i in range(self.n_segments)]
        
        print(f"Divided into {self.n_segments} segments of ~{segment_size:,} chars each")
        
        # Track word occurrences per segment
        word_segment_counts = defaultdict(lambda: [0] * self.n_segments)
        word_metadata = {}  # Store lemma and POS
        
        print("\nProcessing segments with spaCy (this may take a few minutes)...")
        
        for seg_idx, segment in enumerate(segments):
            print(f"  Processing segment {seg_idx + 1}/{self.n_segments}...", end='\r')
            
            doc = self.nlp(segment)
            
            for token in doc:
                # Filter: only content words, no punctuation/spaces
                if token.is_alpha and not token.is_stop and len(token.text) > 2:
                    lemma = token.lemma_.lower()
                    pos = token.pos_
                    
                    # Only track nouns, verbs, adjectives
                    if pos in ['NOUN', 'VERB', 'ADJ']:
                        word_segment_counts[lemma][seg_idx] += 1
                        
                        if lemma not in word_metadata:
                            word_metadata[lemma] = {
                                'word': token.text.lower(),
                                'pos': pos
                            }
        
        print(f"\n\nFound {len(word_segment_counts):,} unique content words")
        
        # Calculate statistics for each word
        print("\nCalculating dispersion scores...")
        
        for lemma, segment_counts in word_segment_counts.items():
            total_count = sum(segment_counts)
            segments_present = sum(1 for count in segment_counts if count > 0)
            dispersion = self.calculate_dispersion(segment_counts)
            
            meta = word_metadata[lemma]
            
            self.word_stats[lemma] = WordStatistics(
                word=meta['word'],
                lemma=lemma,
                pos=meta['pos'],
                total_count=total_count,
                segments_present=segments_present,
                total_segments=self.n_segments,
                dispersion_score=dispersion,
                frequency_rank=0,  # Will be set later
                is_safe=False  # Will be set later
            )
        
        # Rank by frequency
        sorted_words = sorted(self.word_stats.values(), 
                            key=lambda x: x.total_count, 
                            reverse=True)
        
        for rank, word_stat in enumerate(sorted_words, 1):
            self.word_stats[word_stat.lemma].frequency_rank = rank
        
        # Identify safe words
        print(f"\nIdentifying safe words (freq >= {min_frequency}, dispersion >= {min_dispersion:.2f})...")
        
        for lemma, stats in self.word_stats.items():
            if stats.total_count >= min_frequency and stats.dispersion_score >= min_dispersion:
                stats.is_safe = True
                self.safe_words[stats.pos].append(lemma)
        
        # Sort safe words by frequency
        for pos in self.safe_words:
            self.safe_words[pos].sort(
                key=lambda lemma: self.word_stats[lemma].total_count,
                reverse=True
            )
        
        print(f"\nSafe word counts:")
        print(f"  Nouns: {len(self.safe_words['NOUN'])}")
        print(f"  Verbs: {len(self.safe_words['VERB'])}")
        print(f"  Adjectives: {len(self.safe_words['ADJ'])}")
    
    def export_safe_words(self, output_path: str = "safe_words_report.txt", top_n: int = 50):
        """
        Export the top safe words to a text file
        
        Args:
            output_path: Path to output file
            top_n: Number of top words to export per POS category
        """
        print(f"\n{'='*70}")
        print(f"EXPORTING SAFE WORDS")
        print(f"{'='*70}\n")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("LEXICALLY SAFE VOCABULARY FOR NEEDLE CONSTRUCTION\n")
            f.write("="*70 + "\n\n")
            f.write(f"Haystack: {self.haystack_path.name}\n")
            f.write(f"Analysis segments: {self.n_segments}\n")
            f.write(f"Total unique words analyzed: {len(self.word_stats):,}\n\n")
            
            for pos, label in [('NOUN', 'NOUNS'), ('VERB', 'VERBS'), ('ADJ', 'ADJECTIVES')]:
                f.write("\n" + "="*70 + "\n")
                f.write(f"TOP {top_n} SAFE {label}\n")
                f.write("="*70 + "\n\n")
                f.write(f"{'Rank':<6} {'Word':<20} {'Count':<8} {'Dispersion':<12} {'Coverage'}\n")
                f.write("-"*70 + "\n")
                
                for i, lemma in enumerate(self.safe_words[pos][:top_n], 1):
                    stats = self.word_stats[lemma]
                    coverage = f"{stats.segments_present}/{stats.total_segments}"
                    f.write(f"{i:<6} {stats.word:<20} {stats.total_count:<8} "
                           f"{stats.dispersion_score:<12.3f} {coverage}\n")
        
        print(f"Safe words exported to: {output_path}")
    
    def validate_needle(self, needle_text: str, danger_threshold_freq: int = 5, 
                       danger_threshold_disp: float = 0.3) -> NeedleAnalysis:
        """
        Validate a candidate needle against the haystack vocabulary
        
        Args:
            needle_text: The candidate needle string
            danger_threshold_freq: Words below this frequency are flagged as rare
            danger_threshold_disp: Words below this dispersion are flagged as clumped
            
        Returns:
            NeedleAnalysis object with detailed results
        """
        print(f"\n{'='*70}")
        print(f"PHASE 2: NEEDLE VALIDATION")
        print(f"{'='*70}\n")
        print(f"Analyzing needle: \"{needle_text}\"\n")
        
        # Process needle
        doc = self.nlp(needle_text)
        
        danger_words = []
        safe_word_count = 0
        total_content_words = 0
        
        for token in doc:
            if token.is_alpha and not token.is_stop and len(token.text) > 2:
                lemma = token.lemma_.lower()
                total_content_words += 1
                
                if lemma in self.word_stats:
                    stats = self.word_stats[lemma]
                    
                    # Check if word is dangerous
                    issues = []
                    if stats.total_count < danger_threshold_freq:
                        issues.append(f"RARE (only {stats.total_count} occurrences)")
                    
                    if stats.dispersion_score < danger_threshold_disp:
                        issues.append(f"CLUMPED (dispersion: {stats.dispersion_score:.2f}, " +
                                    f"present in {stats.segments_present}/{self.n_segments} segments)")
                    
                    if issues:
                        danger_words.append({
                            'word': token.text,
                            'lemma': lemma,
                            'pos': stats.pos,
                            'issues': issues,
                            'frequency': stats.total_count,
                            'dispersion': stats.dispersion_score
                        })
                    else:
                        safe_word_count += 1
                else:
                    # Word not in haystack at all - extremely dangerous
                    danger_words.append({
                        'word': token.text,
                        'lemma': lemma,
                        'pos': 'UNKNOWN',
                        'issues': ['NOT IN HAYSTACK (model can trivially pattern-match this)'],
                        'frequency': 0,
                        'dispersion': 0.0
                    })
        
        # Calculate safety score
        if total_content_words == 0:
            safety_score = 0.0
        else:
            safety_score = (safe_word_count / total_content_words) * 100
        
        # Generate recommendations
        recommendations = []
        if safety_score < 50:
            recommendations.append("❌ CRITICAL: This needle is highly vulnerable to lexical pattern matching")
            recommendations.append("   Recommendation: Rewrite using words from the safe vocabulary list")
        elif safety_score < 75:
            recommendations.append("⚠️  WARNING: This needle has lexical overlap issues")
            recommendations.append("   Recommendation: Replace flagged words with safer alternatives")
        else:
            recommendations.append("✓ GOOD: This needle has good lexical camouflage")
        
        if danger_words:
            recommendations.append(f"\n🎯 Replace these {len(danger_words)} words:")
            for dw in danger_words:
                if dw['pos'] in self.safe_words and self.safe_words[dw['pos']]:
                    alts = self.safe_words[dw['pos']][:5]
                    recommendations.append(f"   • '{dw['word']}' → try: {', '.join(alts)}")
        
        return NeedleAnalysis(
            needle_text=needle_text,
            safety_score=safety_score,
            total_words=total_content_words,
            safe_words=safe_word_count,
            danger_words=danger_words,
            recommendations=recommendations
        )
    
    def print_needle_report(self, analysis: NeedleAnalysis):
        """Print a formatted needle analysis report"""
        print(f"\n{'='*70}")
        print(f"NEEDLE SAFETY REPORT")
        print(f"{'='*70}\n")
        print(f"Needle: \"{analysis.needle_text}\"")
        print(f"\nSafety Score: {analysis.safety_score:.1f}/100")
        print(f"Safe Words: {analysis.safe_words}/{analysis.total_words}")
        
        if analysis.danger_words:
            print(f"\n⚠️  DANGER WORDS DETECTED ({len(analysis.danger_words)}):")
            print("-"*70)
            for dw in analysis.danger_words:
                print(f"\n'{dw['word']}' ({dw['pos']})")
                for issue in dw['issues']:
                    print(f"  • {issue}")
        else:
            print("\n✓ No danger words detected!")
        
        print(f"\n{'='*70}")
        print("RECOMMENDATIONS")
        print(f"{'='*70}")
        for rec in analysis.recommendations:
            print(rec)
        print()


def main():
    """
    Main execution function with example usage
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Lexical Camouflage: Build adversarial needles for NIAH evaluations"
    )
    parser.add_argument(
        "haystack", 
        help="Path to haystack text file (e.g., CognitiveBiasHaystack.txt)"
    )
    parser.add_argument(
        "--segments", 
        type=int, 
        default=20,
        help="Number of segments to divide haystack into (default: 20)"
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=10,
        help="Minimum frequency for safe words (default: 10)"
    )
    parser.add_argument(
        "--min-disp",
        type=float,
        default=0.5,
        help="Minimum dispersion for safe words (default: 0.5)"
    )
    parser.add_argument(
        "--needle",
        type=str,
        help="Validate a specific needle string"
    )
    parser.add_argument(
        "--export",
        type=str,
        default="safe_words_report.txt",
        help="Path for safe words export (default: safe_words_report.txt)"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Number of top safe words to export per POS (default: 50)"
    )
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = LexicalCamouflage(args.haystack, n_segments=args.segments)
    
    # Analyze haystack
    analyzer.analyze_haystack(
        min_frequency=args.min_freq,
        min_dispersion=args.min_disp
    )
    
    # Export safe words
    analyzer.export_safe_words(args.export, top_n=args.top_n)
    
    # Validate needle if provided
    if args.needle:
        analysis = analyzer.validate_needle(args.needle)
        analyzer.print_needle_report(analysis)
    else:
        # Interactive mode
        print("\n" + "="*70)
        print("INTERACTIVE NEEDLE VALIDATOR")
        print("="*70)
        print("\nEnter candidate needles to validate (or 'quit' to exit)\n")
        
        while True:
            needle = input("Enter needle: ").strip()
            
            if needle.lower() in ['quit', 'exit', 'q']:
                break
            
            if needle:
                analysis = analyzer.validate_needle(needle)
                analyzer.print_needle_report(analysis)
                print("\n")


if __name__ == "__main__":
    main()