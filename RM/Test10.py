import os
import re
import json
import time
import csv
import random
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from google.genai import types
from google import genai
from dotenv import load_dotenv

# --- 1. SETUP & CONFIGURATION ---
load_dotenv()

@dataclass
class NeedleConfig:
    text: str
    custom_position_percent: float
    needle_id: int
    repeat_count: int = 1  # Needle frequency: 2, 4, 6, 8

@dataclass
class QuestionConfig:
    text: str

class AugmentedNeedleHaystack:
    def __init__(self, haystack_text: str):
        self.haystack_text = haystack_text
        self.haystack_sentences = self._split_into_sentences(haystack_text)

    def _split_into_sentences(self, text: str) -> List[str]:
        # Split by sentence endings (.!?) followed by whitespace
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _calculate_index(self, percent: float, current_length: int) -> int:
        idx = int(current_length * percent)
        return max(0, min(idx, current_length))

    def create_context(self, needle_config: NeedleConfig) -> str:
        sentences = self.haystack_sentences.copy()
        
        # --- INSERTION LOGIC ---
        insertions = [] # List of (index, text)

        # Insert needle multiple times with jitter
        needle_base_pos = needle_config.custom_position_percent
        
        for _ in range(needle_config.repeat_count):
            # Add random jitter within ±5% range of target position
            jitter = (random.random() - 0.5) * 0.10  # ±5%
            final_pos = max(0.0, min(1.0, needle_base_pos + jitter))
            
            needle_idx = self._calculate_index(final_pos, len(sentences))
            insertions.append((needle_idx, needle_config.text))

        # Execute Insertions (Sort by Index Descending to preserve order)
        # Inserting from back to front prevents index shifting problems
        insertions.sort(key=lambda x: x[0], reverse=True)

        for idx, text in insertions:
            sentences.insert(idx, text)

        return ' '.join(sentences)

class GeminiProvider:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name
        if not os.getenv("GEMINI_API_KEY"): raise ValueError("No API Key found in .env")
        self.client = genai.Client()
        self.config = types.GenerateContentConfig(
            temperature=0.0,
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )

    def evaluate(self, prompt: str) -> str:
        try:
            response = self.client.models.generate_content(
                model=self.model_name, contents=prompt, config=self.config
            )
            return response.text
        except Exception as e:
            print(f"API Error: {e}")
            return "Error"

class ExperimentRunner:
    def __init__(self, provider: GeminiProvider, haystack_text: str, delay: int = 0):
        self.provider = provider
        self.haystack_text = haystack_text
        self.delay = delay
        self.results = []

    def run_experiment(self, config: Dict):
        nc = config['needle']
        qc = config['question']
        
        print(f"\n--- Running: {config['name']} ---")
        print(f"Pos: {nc.custom_position_percent:.0%} | Needle Frequency: {nc.repeat_count}")

        niah = AugmentedNeedleHaystack(self.haystack_text)
        context = niah.create_context(nc)

        # Prompt for frequency and distractor testing
        prompt = f"You are a helpful AI bot that answers questions for a user. Keep your response short and direct.\n\nDOCUMENT:\n{context}\n\nQUESTION:\n{qc.text}\n\nDon't give information from outside the document / context given or repeat your findings."

        start = time.time()
        response = self.provider.evaluate(prompt)
        duration = time.time() - start
        
        print(f"Response: {response[:100]}...")
        
        # Save detailed result
        self.results.append({
            "experiment_name": config['name'],
            "needle_id": nc.needle_id,
            "needle_text": nc.text,
            "position": nc.custom_position_percent,
            "needle_frequency": nc.repeat_count,
            "response": response,
            "latency": duration,
            "timestamp": datetime.now().isoformat()
        })

    def run_all(self, experiments: List[Dict]):
        total = len(experiments)
        for i, exp in enumerate(experiments):
            print(f"[{i+1}/{total}]")
            self.run_experiment(exp)
            if i < total - 1:
                time.sleep(self.delay)

    def save_results(self, filename="Test10_Frequency_Results.json"):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=4)
        print(f"Results saved to {filename}")

def load_csv_data(filepath: str) -> List[Dict]:
    data = []
    try:
        with open(filepath, mode='r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        return data
    except Exception as e:
        print(f"Error loading CSV {filepath}: {e}")
        return []

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # File Paths
    HAYSTACK_FILE = "CognitiveBias4.txt"
    # This CSV must contain: needle_id, needle_text, question_text
    INPUT_CSV = "freq.csv" 
    DELAY_SECONDS = 60

    # 1. Load Data
    print("Loading data...")
    all_data = load_csv_data(INPUT_CSV)
    if not all_data:
        print("No data found in CSV. Exiting.")
        exit()

    try:
        with open(HAYSTACK_FILE, "r", encoding="utf-8") as f:
            haystack = f.read()
    except FileNotFoundError:
        print("Haystack file not found.")
        exit()

    # 2. Generate Experiment Plan
    # Map needle_id to position: ID 1 = 0%, ID 2 = 10%, ID 3 = 20%, etc.
    needle_frequencies = [2, 4, 6, 8]  # How many copies of needle to insert
    experiments = []

    for item in all_data:
        n_id = int(item['needle_id'])
        
        # Calculate position based on needle_id: (ID - 1) * 0.1
        # ID 1 -> 0%, ID 2 -> 10%, ID 3 -> 20%, ..., ID 11 -> 100%
        pos = (n_id - 1) * 0.1
        pos = min(pos, 1.0)  # Cap at 100%

        for needle_freq in needle_frequencies:
            nc = NeedleConfig(
                text=item['needle_text'],
                custom_position_percent=pos,
                needle_id=n_id,
                repeat_count=needle_freq
            )
            qc = QuestionConfig(text=item['question_text'])

            experiments.append({
                "name": f"FrequencyTest_ID{n_id}_Pos{pos*100:.0f}%_Freq{needle_freq}",
                "needle": nc,
                "question": qc
            })

    print(f"Generated {len(experiments)} experiments.")

    # 3. Run
    # Uncomment to execute
 
    provider = GeminiProvider(model_name="gemini-2.5-flash")
    runner = ExperimentRunner(provider, haystack, DELAY_SECONDS)
    runner.run_all(experiments)
    runner.save_results()
    print("Experiment completed.")