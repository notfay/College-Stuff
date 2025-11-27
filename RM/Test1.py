# run_combined_test.py
import os
import re
import json
import time
import csv  # <<< Tambahkan import ini
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from enum import Enum  # <<< Pastikan ini ada
from pathlib import Path
from google.genai import types
from google import genai
from dotenv import load_dotenv

# --- KONFIGURASI & SETUP ---
load_dotenv()

# ==================================================================
# =================== BAGIAN YANG HILANG & DIPERBAIKI =================
# ==================================================================
# Class ini hilang di file Anda, menyebabkan NameError
class Position(Enum):
    START = "start"
    MIDDLE = "middle"
    END = "end"
    RANDOM = "random"
# ==================================================================

@dataclass
class NeedleConfig:
    text: str
    custom_position_percent: Optional[float] = None
    # Tambahkan ID dan similarity untuk pencatatan
    needle_id: int = 0
    haystack_similarity: float = 0.0

@dataclass
class QuestionConfig:
    text: str
    question_similarity: float = 0.0

# ... (Class AugmentedNeedleHaystack dan GeminiProvider tetap sama) ...
class AugmentedNeedleHaystack:
    def __init__(self, haystack_text: str):
        self.haystack_text = haystack_text
        self.haystack_sentences = self._split_into_sentences(haystack_text)

    def _split_into_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    # Sekarang 'Position' dikenali karena class-nya sudah ditambahkan di atas
    def _calculate_position_index(self, position: Position = None, custom_position_percent: float = None) -> int:
        total_sentences = len(self.haystack_sentences)
        if custom_position_percent is not None:
            percent = max(0.0, min(1.0, custom_position_percent))
            index = int(total_sentences * percent)
            return min(index, total_sentences - 1) if total_sentences > 0 else 0
        
        # Bagian ini tidak akan terpakai di 'main' Anda saat ini,
        # tapi kami biarkan agar tidak error jika Anda butuh lagi
        if position == Position.START:
            return random.randint(0, max(1, int(total_sentences * 0.20)))
        elif position == Position.MIDDLE:
            return random.randint(int(total_sentences * 0.40), int(total_sentences * 0.60))
        elif position == Position.END:
            return random.randint(int(total_sentences * 0.80), max(0, total_sentences - 1))
        elif position == Position.RANDOM:
            return random.randint(0, max(0, total_sentences - 1))
            
        return total_sentences // 2

    def create_context(self, needle_config: NeedleConfig) -> str:
        sentences = self.haystack_sentences.copy()
        needle_pos = self._calculate_position_index(
            custom_position_percent=needle_config.custom_position_percent
        )
        sentences.insert(needle_pos, needle_config.text)
        return ' '.join(sentences)

class GeminiProvider:
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        self.model_name = model_name
        if not os.getenv("GEMINI_API_KEY"):
            raise ValueError("GEMINI_API_KEY not found in .env file.")
        self.client = genai.Client()
        self.generation_config = types.GenerateContentConfig(
            temperature=0.0,
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )
        
    def evaluate(self, prompt: str) -> str:
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=self.generation_config
            )
            return response.text
        except Exception as e:
            print(f"An error occurred while calling the Gemini API: {e}")
            return "Error: Could not retrieve response from model."
        
    def count_tokens(self, text: str) -> int:
        try:
            token_count_response = self.client.models.count_tokens(
                model=self.model_name,
                contents=text
            )
            return token_count_response.total_tokens
        except Exception as e:
            print(f"An error occurred while counting tokens: {e}")
            return -1

# --- RUNNER DIPERBARUI UNTUK MENANGANI QUESTION YANG BERUBAH-UBAH ---
class ExperimentRunner:
    def __init__(self, provider: GeminiProvider, haystack_text: str, delay_seconds: int = 0):
        self.provider = provider
        self.haystack_text = haystack_text
        self.delay_seconds = delay_seconds
        self.results = []

    def run_experiment(self, experiment_config: Dict):
        # Ambil Question dari config, bukan dari __init__
        needle_config = experiment_config['needle']
        question_config = experiment_config['question']
        
        print(f"\n--- Running Experiment: {experiment_config['name']} ---")
        
        niah = AugmentedNeedleHaystack(self.haystack_text)
        augmented_context = niah.create_context(needle_config=needle_config)
        
        prompt = f"You are a helpful AI bot that answers questions for a user. Keep your response short and direct.\n\nDOCUMENT:\n{augmented_context}\n\nQUESTION:\n{question_config.text}\n\nDon't give information from outside the document / context given or repeat your findings."
        
        # DISABLED: Token counting doubles API usage by sending the full haystack twice
        # token_count = self.provider.count_tokens(prompt)
        # print(f"Verified prompt token count: {token_count}") 
        
        start_time = time.time()
        response_text = self.provider.evaluate(prompt)
        end_time = time.time()
        
        latency = end_time - start_time
        print(f"Response: {response_text[:100]}...")
        print(f"Latency: {latency:.2f} seconds")

        result_data = {
            "experiment_name": experiment_config['name'],
            "config": {
                "needle": asdict(needle_config),
                "question": asdict(question_config) 
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

    def save_results(self, output_file: str = "Test1_results.json"):
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=4)
        print(f"\n✓ All results saved to: {output_path}")

# --- FUNGSI BARU UNTUK MEMBACA FILE CSV ---
def load_needles_and_questions(filepath: str) -> List[Dict]:
    """Membaca file CSV dan mengembalikannya sebagai daftar dictionary."""
    data = []
    try:
        with open(filepath, mode='r', encoding='utf-8-sig') as csvfile: # 'utf-8-sig' untuk menangani BOM
            reader = csv.DictReader(csvfile)
            for row in reader:
                data.append(row)
        print(f">>> Successfully loaded {len(data)} needle/question pairs from {filepath}")
        return data
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found. Please create it.")
        return []
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        return []

# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # ==================================================================
    # =================== EXPERIMENT CONTROL PANEL =====================
    # ==================================================================
    HAYSTACK_FILE = "CognitiveBias4.txt"
    # Nama file yang berisi needle dan question
    NEEDLE_QUESTION_FILE = "needles_and_questions.csv" 
    DELAY_BETWEEN_CALLS_SECONDS = 70
    # ==================================================================
    
    print(">>> Generating experiment plan from control panel settings...")
    
    # 1. Muat data Needle dan Question dari file
    nq_data = load_needles_and_questions(NEEDLE_QUESTION_FILE)
    if not nq_data:
        exit() # Keluar jika file tidak ada atau kosong

    # 2. Tentukan posisi yang akan diuji
    needle_positions = np.linspace(0, 1, 11) # 0, 0.1, 0.2 ... 1.0

    # 3. Buat rencana eksperimen dengan loop bersarang
    experiments_to_run = []
    for percent in needle_positions:
        for item in nq_data:
            try:
                # Ambil data dari baris CSV
                h_sim = float(item['haystack_similarity'])
                q_sim = float(item['question_similarity'])
                n_id = int(item['needle_id'])

                # Buat nama eksperimen yang deskriptif
                exp_name = f"Pos: {percent*100:.0f}%, N-ID: {n_id}, H-Sim: {h_sim}, Q-Sim: {q_sim}"
                
                # Buat objek config
                needle_config = NeedleConfig(
                    text=item['needle_text'],
                    custom_position_percent=float(percent),
                    needle_id=n_id,
                    haystack_similarity=h_sim
                )
                question_config = QuestionConfig(
                    text=item['question_text'],
                    question_similarity=q_sim
                )
                
                experiments_to_run.append({
                    "name": exp_name,
                    "needle": needle_config,
                    "question": question_config
                })
            except KeyError as e:
                print(f"Error: Missing column {e} in your CSV file. Please check the header.")
                exit()
            except ValueError as e:
                print(f"Error: Could not convert value in CSV to number (float/int). Check {e}")
                exit()

    print(f">>> Plan generated. Total experiments to run: {len(experiments_to_run)}")
    # for i, exp in enumerate(experiments_to_run):
    #     print(f"  {i+1}. {exp['name']}") # Matikan ini agar tidak terlalu panjang

    try:
        with open(HAYSTACK_FILE, "r", encoding="utf-8") as f:
            base_haystack = f.read()
    except FileNotFoundError:
        print(f"Error: Haystack file '{HAYSTACK_FILE}' not found. Please create it.")
        exit()
        
    gemini_provider = GeminiProvider(model_name="gemini-2.5-flash")
    # Perbarui Runner, sekarang tidak butuh question di awal
    runner = ExperimentRunner(
        provider=gemini_provider,
        haystack_text=base_haystack,
        delay_seconds=DELAY_BETWEEN_CALLS_SECONDS
    )
    runner.run_all(experiments_to_run)
    runner.save_results()