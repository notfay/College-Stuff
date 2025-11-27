"""
Script to generate needles using Gemini 2.5 Flash that blend with the haystack.
Checks similarity and saves needles with haystack_similarity > 0.8 to CSV.
"""

import os
import time
import csv
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class NeedleGenerator:
    def __init__(self, haystack_file: str, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name
        self.haystack_file = haystack_file
        self.haystack_text = self._load_haystack()
        
        if not os.getenv("GEMINI_API_KEY"):
            raise ValueError("GEMINI_API_KEY not found in .env file.")
        
        self.client = genai.Client()
        self.generated_needles = []
    
    def _load_haystack(self) -> str:
        """Load the haystack text from file."""
        with open(self.haystack_file, 'r', encoding='utf-8') as f:
            return f.read()
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts using TF-IDF."""
        try:
            vectorizer = TfidfVectorizer()
            vectors = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
            return similarity
        except:
            return 0.0
    
    def generate_needle(self, attempt_number: int) -> str:
        """
        Use Gemini to generate a needle that blends with the haystack.
        """
        # Add context about previously generated needles to encourage diversity
        previous_themes = []
        if self.generated_needles:
            previous_themes = [f"- {n['needle_text'][:60]}..." for n in self.generated_needles[-3:]]
            previous_context = "\n\nPreviously generated needles (DO NOT repeat or closely paraphrase these):\n" + "\n".join(previous_themes)
        else:
            previous_context = ""
        
        prompt = f"""You are analyzing John Locke's "Essay Concerning Human Understanding".

Generate ONE single sentence (30-50 words) that:
1. Uses Locke's philosophical vocabulary (e.g., "ideas", "reflection", "sensation", "understanding", "mind", "knowledge", "perception")
2. Matches his writing style and sentence structure
3. Discusses epistemology, the nature of knowledge, or the workings of the mind
4. Could plausibly appear in the text
5. Is grammatically correct and philosophically coherent
6. Is COMPLETELY DIFFERENT from any previously generated sentences - explore different philosophical concepts

DO NOT copy existing sentences. Create a UNIQUE and ORIGINAL statement that sounds like Locke but covers a DIFFERENT philosophical point.{previous_context}

Generate attempt #{attempt_number}:"""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.9,  # Higher creativity
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                ),
            )
            needle = response.text.strip()
            # Remove quotes if present
            needle = needle.strip('"').strip("'").strip()
            return needle
        except Exception as e:
            print(f"Error generating needle: {e}")
            return None
    
    def is_needle_unique(self, new_needle: str, min_difference: float = 0.3) -> bool:
        """
        Check if the new needle is sufficiently different from all previously generated needles.
        Returns True if unique enough, False if too similar to existing needles.
        """
        if not self.generated_needles:
            return True
        
        for existing in self.generated_needles:
            existing_needle = existing['needle_text']
            similarity = self.calculate_similarity(new_needle, existing_needle)
            
            if similarity > (1 - min_difference):  # If similarity > 0.7, they're too similar
                print(f"   ⚠️  Too similar to existing needle (similarity: {similarity:.4f})")
                print(f"       Existing: {existing_needle[:80]}...")
                return False
        
        return True
    
    def generate_and_check_needles(self, target_count: int = None, max_attempts: int = None):
        """
        Generate needles with similarity >= 0.2, 0.4, 0.6, or 0.8
        Ensures each needle is unique and not too similar to previously generated ones.
        """
        print(f"🎯 Goal: Generate needles with similarity >= 0.2, 0.4, 0.6, or 0.8")
        print(f"📄 Haystack: {self.haystack_file}")
        print(f"⏱️  Rate limit: 63 seconds between requests (60 + 3)")
        print(f"🛑 Press Ctrl+C to stop generation\n")
        print("=" * 80)
        
        attempt = 0
        successful = 0
        threshold_counts = {'0.2': 0, '0.4': 0, '0.6': 0, '0.8': 0}
        
        while True:  # Run indefinitely until Ctrl+C
            attempt += 1
            
            print(f"\n🔄 Attempt {attempt}")
            print(f"✅ Successful so far: {successful} (0.2+: {threshold_counts['0.2']}, 0.4+: {threshold_counts['0.4']}, 0.6+: {threshold_counts['0.6']}, 0.8+: {threshold_counts['0.8']})")
            
            # Generate needle
            print("   Generating needle with Gemini...")
            needle = self.generate_needle(attempt)
            
            if not needle:
                print("   ❌ Failed to generate needle")
                time.sleep(63)  # Still wait to avoid rate limit
                continue
            
            print(f"   📝 Generated: {needle[:100]}...")
            
            # Check if needle is unique enough
            if not self.is_needle_unique(needle):
                print("   ❌ Duplicate or too similar to existing needle - rejecting")
                time.sleep(63)  # Still wait to avoid rate limit
                continue
            
            # Calculate similarity with haystack
            similarity = self.calculate_similarity(needle, self.haystack_text)
            print(f"   📊 Haystack similarity: {similarity:.4f}")
            
            # Categorize by similarity threshold
            similarity_threshold = None
            if similarity >= 0.8:
                similarity_threshold = '0.8'
                print(f"   ✅ SUCCESS! Similarity {similarity:.4f} >= 0.8")
            elif similarity >= 0.6:
                similarity_threshold = '0.6'
                print(f"   ✅ SUCCESS! Similarity {similarity:.4f} >= 0.6")
            elif similarity >= 0.4:
                similarity_threshold = '0.4'
                print(f"   ✅ SUCCESS! Similarity {similarity:.4f} >= 0.4")
            elif similarity >= 0.2:
                similarity_threshold = '0.2'
                print(f"   ✅ SUCCESS! Similarity {similarity:.4f} >= 0.2")
            else:
                print(f"   ❌ Too low. Similarity {similarity:.4f} < 0.2")
            
            if similarity_threshold:
                self.generated_needles.append({
                    'needle_id': successful + 1,
                    'similarity_threshold': similarity_threshold,
                    'haystack_similarity': round(similarity, 4),
                    'needle_text': needle
                })
                successful += 1
                threshold_counts[similarity_threshold] += 1
                
                # Save immediately after each success
                self._save_to_csv()
            
            # Rate limit delay
            print(f"\n   ⏳ Waiting 63 seconds for rate limit...")
            time.sleep(63)
        
        print("\n" + "=" * 80)
        print(f"\n🎉 Generation complete!")
        print(f"   Successful needles: {successful}")
        print(f"   Total attempts: {attempt}")
        print(f"   Success rate: {successful/attempt*100:.1f}%")
        print(f"   By threshold: {threshold_counts}")
        
        return self.generated_needles
    
    def _save_to_csv(self):
        """Save generated needles to CSV file."""
        output_file = "generated_needles.csv"
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['needle_id', 'similarity_threshold', 'haystack_similarity', 'needle_text']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.generated_needles)
        
        print(f"   💾 Saved to {output_file}")


if __name__ == "__main__":
    # Configuration
    HAYSTACK_FILE = "CognitiveBiaS2.txt"
    
    print("🚀 Needle Generator")
    print("=" * 80)
    
    # Initialize generator
    generator = NeedleGenerator(HAYSTACK_FILE)
    
    # Generate needles (runs until Ctrl+C)
    try:
        needles = generator.generate_and_check_needles()
    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("🛑 Generation stopped by user (Ctrl+C)")
        print("=" * 80)
        needles = generator.generated_needles
    
    # Final summary
    print("\n" + "=" * 80)
    print("📋 GENERATED NEEDLES:")
    print("=" * 80)
    for needle in needles:
        print(f"\nID: {needle['needle_id']}")
        print(f"Threshold: {needle['similarity_threshold']}")
        print(f"Similarity: {needle['haystack_similarity']}")
        print(f"Text: {needle['needle_text']}")
    
    print(f"\n✅ Results saved to: generated_needles.csv")
