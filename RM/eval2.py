# run_niah_gemini_dynamic_v2_fixed.py
import os
import re
import json
import time
import random
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from google.genai import types
# The import is now from the top-level 'google' package
from google import genai
from dotenv import load_dotenv

# --- CONFIGURATION & SETUP ---
load_dotenv()

class Position(Enum):
    START = "start"
    MIDDLE = "middle"
    END = "end"
    RANDOM = "random"

@dataclass
class NeedleConfig:
    text: str
    position: Optional[Position] = None
    min_percent: Optional[float] = None
    max_percent: Optional[float] = None
    frequency: int = 1

@dataclass
class DistractorConfig:
    text: str
    count: int
    position: Optional[Position] = None
    min_percent: Optional[float] = None  # Custom range support
    max_percent: Optional[float] = None

# --- COMPONENT 1: The Assembly Line (Context Generator) ---
class AugmentedNeedleHaystack:
    def __init__(self, haystack_text: str):
        self.haystack_text = haystack_text
        self.haystack_sentences = self._split_into_sentences(haystack_text)

    def _split_into_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _calculate_position_index(self, position: Position = None, 
                                      min_percent: float = None, 
                                      max_percent: float = None) -> int:
        total_sentences = len(self.haystack_sentences)
        
        final_percent = None

        if min_percent is not None:
            use_max = max_percent if max_percent is not None else min_percent
            final_percent = random.uniform(min_percent, use_max)
        
        if final_percent is not None:
            percent = max(0.0, min(1.0, final_percent))
            index = int(total_sentences * percent)
            return min(index, total_sentences - 1) if total_sentences > 0 else 0

        if position == Position.START:
            return random.randint(0, max(1, int(total_sentences * 0.20)))
        elif position == Position.MIDDLE:
            return random.randint(int(total_sentences * 0.40), int(total_sentences * 0.60))
        elif position == Position.END:
            return random.randint(int(total_sentences * 0.80), max(0, total_sentences - 1))
        elif position == Position.RANDOM:
            return random.randint(0, max(0, total_sentences - 1))
        
        return total_sentences // 2

    def create_context(self, needle_config: NeedleConfig, distractor_configs: List[DistractorConfig] = None) -> str:
        sentences = self.haystack_sentences.copy()
        
        # ==================================================================
        # =================== LOGIKA YANG DIPERBAIKI ADA DI SINI =====================
        # ==================================================================
        
        # Loop sebanyak 'frequency' kali (misal: 3, 5, 7x)
        for _ in range(needle_config.frequency):
            
            # 1. Hitung posisi acak BARU SETIAP KALI loop berjalan
            #    di dalam rentang yang ditentukan (misal: 15%-25%)
            needle_pos = self._calculate_position_index(
                position=needle_config.position,
                min_percent=needle_config.min_percent,
                max_percent=needle_config.max_percent
            )
            
            # 2. Masukkan needle di posisi acak yang baru ditemukan
            sentences.insert(needle_pos, needle_config.text)
        
        # =================== AKHIR DARI PERBAIKAN =========================
        
        if distractor_configs:
            for distractor in distractor_configs:
                for _ in range(distractor.count):
                    dist_pos = self._calculate_position_index(
                        position=distractor.position,
                        min_percent=distractor.min_percent,
                        max_percent=distractor.max_percent
                    )
                    sentences.insert(dist_pos, distractor.text)
        return ' '.join(sentences)

# --- COMPONENT 2: The Quality Control Robot (LLM Caller) ---
# ... (Tidak ada perubahan di GeminiProvider) ...
class GeminiProvider:
    """
    A dedicated class to handle all interactions with the Google Gemini API,
    updated to use the latest genai.Client() interface.
    """
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        self.model_name = model_name
        if not os.getenv("GEMINI_API_KEY"):
            raise ValueError("GEMINI_API_KEY not found in .env file.")
        
        # Initialize the client, which automatically picks up the API key
        # from the environment variable 'GEMINI_API_KEY'.
        self.client = genai.Client()

    def evaluate(self, prompt: str) -> str:
        """
        Sends the prompt to the Gemini model and returns the text response.
        """
        try:
            # The API call now uses client.models.generate_content and passes
            # the prompt to the 'contents' parameter.
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0) # Disables thinking
                ),
            )
            # The response text is accessed via the .text attribute.
            return response.text
        except Exception as e:
            print(f"An error occurred while calling the Gemini API: {e}")
            return "Error: Could not retrieve response from model."


# --- COMPONENT 3: The Factory Manager (Experiment Runner) ---
# ... (Tidak ada perubahan di ExperimentRunner) ...
class ExperimentRunner:
    def __init__(self, provider: GeminiProvider, haystack_text: str, question: str, delay_seconds: int = 0):
        self.provider = provider
        self.haystack_text = haystack_text
        self.question = question
        self.delay_seconds = delay_seconds
        self.results = []

    def run_experiment(self, experiment_config: Dict):
        print(f"\n--- Running Experiment: {experiment_config['name']} ---")
        niah = AugmentedNeedleHaystack(self.haystack_text)
        augmented_context = niah.create_context(
            needle_config=experiment_config['needle'],
            distractor_configs=experiment_config.get('distractors', [])
        )
        prompt = f"You are a helpful AI bot that answers questions for a user. Keep your response short and direct.\n\nDOCUMENT:\n{augmented_context}\n\nQUESTION:\n{self.question}\n\nDon't give information outside the document or repeat your findings."
        
        start_time = time.time()
        response_text = self.provider.evaluate(prompt)
        end_time = time.time()
        
        latency = end_time - start_time
        print(f"Response: {response_text}...")
        print(f"Latency: {latency:.2f} seconds")

        result_data = {
            "experiment_name": experiment_config['name'],
            "config": {
                "needle": asdict(experiment_config['needle']),
                "distractors": [asdict(d) for d in experiment_config.get('distractors', [])]
            },
            "response": response_text,
            "latency_seconds": latency,
            "timestamp": datetime.now().isoformat()
        }
        self.results.append(result_data)

    def run_all(self, experiments: List[Dict]):
        for i, exp in enumerate(experiments):
            self.run_experiment(exp)
            if i < len(experiments) - 1 and self.delay_seconds > 0:
                print(f"\nWaiting for {self.delay_seconds} seconds before the next API call...")
                time.sleep(self.delay_seconds)

    def save_results(self, output_file: str = "niah_results.json"):
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=4)
        print(f"\n✓ All results saved to: {output_path}")

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # ==================================================================
    # =================== EXPERIMENT CONTROL PANEL =====================
    # ==================================================================
    HAYSTACK_FILE = "Frankenstein_shortened.txt"
    QUESTION = "According to Bishop Myriel, why did one of the silver candlesticks feel heavier than the other?"
    NEEDLE_TEXT = "Bishop Myreil confided in his sister, Bapstistine, that one of the sliver candlestikcs always felt slightly heavier in his hand, a minor imperfetion he attrbuted to a fllaw. in it baes."
    DISTRACTOR_TEXT = "Madame Myriel noticed that one of the silver candlesticks felt slightly heavier, which she attributed to accumulation of wax residue over the years of nightly use."
    DELAY_BETWEEN_CALLS_SECONDS = 70
    
    NEEDLE_POSITION_RANGES = [
        (0.00, 0.10),  # 0% - 10%
        (0.10, 0.20),  # 10% - 20%
        (0.20, 0.30),  # 20% - 30%
        (0.30, 0.40),  # 30% - 40%
        (0.40, 0.50),  # 40% - 50%
        (0.50, 0.60),  # 50% - 60%
        (0.60, 0.70),  # 60% - 70%
        (0.70, 0.80),  # 70% - 80%
        (0.80, 0.90),  # 80% - 90%
        (0.90, 1.00),  # 90% - 100%
    ]
    
    # Test combination of needle frequency and distractor count
    # e.g., 1 needle with 1/3/5/7 distractors, 3 needles with 1/3/5/7 distractors, etc.
    NEEDLE_FREQUENCIES_TO_TEST = [1, 3, 5, 7]
    
    # Distractor frequency matches needle frequency (same as needle count)
    MATCH_DISTRACTOR_TO_NEEDLE_FREQUENCY = False  # Use independent distractor counts below
    DISTRACTOR_AMOUNTS_TO_TEST = [1, 3, 5, 7]
    DISTRACTOR_IN_SAME_RANGE_AS_NEEDLE = True  # Distractors appear in same range as needle
    DISTRACTOR_LOCATION = Position.MIDDLE  # Only used if DISTRACTOR_IN_SAME_RANGE_AS_NEEDLE is False
    # ==================================================================
    
    print(">>> Generating experiment plan from control panel settings...")
    
    needle_ranges = NEEDLE_POSITION_RANGES
    print(f"Using {len(needle_ranges)} custom position ranges:")
    for min_p, max_p in needle_ranges:
        print(f"  - Range: {min_p*100:.0f}% to {max_p*100:.0f}%")


    experiments_to_run = []
    for min_p, max_p in needle_ranges:
        for needle_freq in NEEDLE_FREQUENCIES_TO_TEST:
            # Determine distractor count based on configuration
            if MATCH_DISTRACTOR_TO_NEEDLE_FREQUENCY:
                distractor_counts = [needle_freq]  # Match needle frequency
            else:
                distractor_counts = DISTRACTOR_AMOUNTS_TO_TEST
            
            for dist_count in distractor_counts:
                exp_name = f"Needle in {min_p*100:.0f}-{max_p*100:.0f}%, {dist_count} distractors, {needle_freq}x frequency"
                
                needle_config = NeedleConfig(
                    text=NEEDLE_TEXT, 
                    min_percent=min_p,
                    max_percent=max_p,
                    frequency=needle_freq
                )
                
                distractor_config = []
                if dist_count > 0:
                    if DISTRACTOR_IN_SAME_RANGE_AS_NEEDLE:
                        # Distractors use same range as needle
                        distractor_config.append(DistractorConfig(
                            text=DISTRACTOR_TEXT, 
                            count=dist_count, 
                            position=None,
                            min_percent=min_p,  # Same as needle
                            max_percent=max_p   # Same as needle
                        ))
                    else:
                        # Distractors use fixed position (e.g., MIDDLE)
                        distractor_config.append(DistractorConfig(
                            text=DISTRACTOR_TEXT, 
                            count=dist_count, 
                            position=DISTRACTOR_LOCATION
                        ))
                
                experiments_to_run.append({
                    "name": exp_name, "needle": needle_config, "distractors": distractor_config
                })

    print(f">>> Plan generated. Total experiments to run: {len(experiments_to_run)}")
    for i, exp in enumerate(experiments_to_run):
        print(f"  {i+1}. {exp['name']}")

    try:
        with open(HAYSTACK_FILE, "r", encoding="utf-8") as f:
            base_haystack = f.read()
    except FileNotFoundError:
        print(f"Error: Haystack file '{HAYSTACK_FILE}' not found. Please create it.")
        exit()
        
    gemini_provider = GeminiProvider(model_name="gemini-2.5-flash")
    runner = ExperimentRunner(
        provider=gemini_provider,
        haystack_text=base_haystack,
        question=QUESTION,
        delay_seconds=DELAY_BETWEEN_CALLS_SECONDS
    )
    runner.run_all(experiments_to_run)
    runner.save_results()