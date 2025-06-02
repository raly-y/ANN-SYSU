import json
import os
import random
import torch
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import yaml


def load_jsonl(file_path):
    """Load parallel corpus from a jsonl file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def calculate_bleu(references, hypothesis):
    # Ensure hypothesis is not empty
    if not hypothesis:
        print(f"BLEU error: Empty hypothesis, references={references}")
        return 0.0

    # Use smoothing to handle short sequences
    smoothing = SmoothingFunction().method1
    weights = (0.5, 0.5, 0, 0)
    try:
        score = sentence_bleu(references, hypothesis, weights=weights, smoothing_function=smoothing)
        return score
    except Exception as e:
        print(f"BLEU error: {e}, hypothesis={hypothesis}, references={references}")
        return 0.0


def log_results(log_dict, log_path):
    """Log experiment results to a file."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_dict, ensure_ascii=False) + '\n')


def save_translations(translations, output_path):
    """Save translation results to a jsonl file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for trans in translations:
            f.write(json.dumps(trans, ensure_ascii=False) + '\n')


def tokens_to_sentence(tokens, vocab, is_chinese=False):
    if not isinstance(tokens, (list, tuple)):
        tokens = [tokens]
    idx_to_word = {idx: word for word, idx in vocab.items()}
    words = []
    for idx in tokens:
        if isinstance(idx, int):
            word = idx_to_word.get(idx, '<unk>')
            if word not in ['<pad>', '<sos>', '<eos>']:
                words.append(word)
    return ''.join(words) if is_chinese else ' '.join(words)


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)