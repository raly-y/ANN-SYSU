import os
import json
import pickle
import torch
import jieba
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import load_config, calculate_bleu, tokens_to_sentence, load_jsonl
from src.data_preprocess import load_pretrained_embeddings, clean_text
from src.model import init_model


def greedy_decode(model, src, en_vocab, device, max_len=50):
    """
    Perform Greedy Decoding for a single source sentence.

    Args:
        model: Trained Seq2Seq model
        src: Source tensor (1, seq_len)
        en_vocab: English vocabulary
        device: torch.device
        max_len: Maximum target length

    Returns:
        pred_ids: List of predicted token IDs
        attn_weights: Attention weights (tgt_len, src_len)
    """
    model.eval()
    pred_ids = [en_vocab['<sos>']]
    attn_weights = []

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src)  # (1, seq_len, hidden*2), (num_layers, 1, hidden)

        for _ in range(max_len):
            input = torch.tensor([[pred_ids[-1]]], dtype=torch.long).to(device)  # (1, 1)
            output, hidden, attn = model.decoder(input, hidden, encoder_outputs)  # output: (1, 1, vocab_size)
            pred_id = output.argmax(-1).item()  # 选择概率最高的词

            pred_ids.append(pred_id)
            attn_weights.append(attn.squeeze(0).cpu().numpy())

            if pred_id == en_vocab['<eos>']:
                break

        attn_weights = np.array(attn_weights) if attn_weights else np.zeros((0, src.size(1)))
        return pred_ids, attn_weights


def beam_search(model, src, en_vocab, device, beam_size=5, max_len=50):
    """
    Perform Beam Search decoding for a single source sentence.

    Args:
        model: Trained Seq2Seq model
        src: Source tensor (1, seq_len)
        en_vocab: English vocabulary
        device: torch.device
        beam_size: Number of beams
        max_len: Maximum target length

    Returns:
        best_sequence: List of token IDs
        best_score: Log probability of the sequence
        attn_weights: Attention weights (tgt_len, src_len)
    """
    model.eval()
    idx_to_word = {idx: word for word, idx in en_vocab.items()}
    unk_idx = en_vocab['<unk>']

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src)
        beam = [([en_vocab['<sos>']], 0.0, hidden, [])]
        completed = []

        for t in range(max_len):
            candidates = []
            for seq, score, hidden, attn_list in beam:
                input = torch.tensor([[seq[-1]]], dtype=torch.long).to(device)
                output, new_hidden, attn_weights = model.decoder(input, hidden, encoder_outputs)
                probs = torch.softmax(output, dim=-1).squeeze(0)

                top_probs, top_indices = probs.topk(beam_size + 1)
                for i in range(beam_size + 1):
                    idx = top_indices[i].item()
                    prob = top_probs[i].item()
                    if idx == unk_idx and i < beam_size:
                        idx = top_indices[i + 1].item()
                        prob = top_probs[i + 1].item()

                    new_seq = seq + [idx]
                    new_score = score + np.log(prob + 1e-10)
                    new_attn = attn_list + [attn_weights.squeeze(0).cpu().numpy()]
                    candidates.append((new_seq, new_score, new_hidden, new_attn))

            candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]
            beam = []
            for seq, score, hidden, attn in candidates:
                if seq[-1] == en_vocab['<eos>']:
                    completed.append((seq, score, attn))
                else:
                    beam.append((seq, score, hidden, attn))

            if not beam or len(completed) >= beam_size:
                break

        if completed:
            best = max(completed, key=lambda x: x[1])
            best_sequence, best_score, attn_weights = best
        else:
            best = max(beam, key=lambda x: x[1])
            best_sequence, best_score, _, attn_weights = best

        attn_weights = np.array(attn_weights) if attn_weights else np.zeros((0, src.size(1)))
        return best_sequence, best_score, attn_weights


def visualize_attention(src_tokens, tgt_tokens, attn_weights, save_path):
    """
    Visualize attention weights as a heatmap.

    Args:
        src_tokens: List of source tokens
        tgt_tokens: List of target tokens
        attn_weights: Numpy array (tgt_len, src_len)
        save_path: Path to save heatmap
    """
    # 设置字体，防止中文乱码
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    num_src_tokens = len(src_tokens)
    num_tgt_tokens = len(tgt_tokens)

    # 调整图像大小以容纳所有标签
    figsize_width = max(12, num_src_tokens * 0.7)
    figsize_height = max(6, num_tgt_tokens * 0.5)

    plt.figure(figsize=(figsize_width, figsize_height))

    # 画热图
    ax = sns.heatmap(
        attn_weights,
        xticklabels=src_tokens,
        yticklabels=tgt_tokens,
        cmap='viridis',
        cbar=True,
        square=False
    )

    # 设置 X 轴标签属性
    ax.set_xticks([i + 0.5 for i in range(len(src_tokens))])  # 保证标签位于每个单元格中心
    ax.set_xticklabels(src_tokens, rotation=45, ha='right', fontsize=12)

    # 设置 Y 轴标签属性
    ax.set_yticks([i + 0.5 for i in range(len(tgt_tokens))])
    ax.set_yticklabels(tgt_tokens, rotation=0, va='center', fontsize=12)

    # 设置标题和坐标轴标签
    plt.xlabel('Source (Chinese)', fontsize=14)
    plt.ylabel('Target (English)', fontsize=14)
    plt.title('Attention Weights Heatmap', fontsize=16)

    # 保证标签完整可见
    plt.tight_layout()

    # 创建目录并保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def inference(config):
    """
    Perform inference on test set and visualize attention for first 3 samples.

    Args:
        config: Configuration dictionary

    Returns:
        results: List of dictionaries with translations and BLEU scores
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Validate decode_type
    decode_type = config['inference']['decode_type']
    beam_size = config['inference']['beam_size']

    print(f"Using decode_type: {decode_type}")

    # Load vocabularies
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    zh_vocab_path = os.path.join(PROJECT_ROOT, config['data']['vocab_path_zh'])
    en_vocab_path = os.path.join(PROJECT_ROOT, config['data']['vocab_path_en'])

    if not os.path.exists(zh_vocab_path) or not os.path.exists(en_vocab_path):
        raise FileNotFoundError(f"Vocabulary files not found: {zh_vocab_path}, {en_vocab_path}")

    with open(zh_vocab_path, 'rb') as f:
        zh_vocab = pickle.load(f)
    with open(en_vocab_path, 'rb') as f:
        en_vocab = pickle.load(f)

    print(f"Loaded vocabularies: zh_vocab size={len(zh_vocab)}, en_vocab size={len(en_vocab)}")

    # Create reverse vocabulary
    idx_to_word = {idx: word for word, idx in en_vocab.items()}

    # Load pretrained embeddings
    zh_embeddings = load_pretrained_embeddings(zh_vocab,
                                               os.path.join(PROJECT_ROOT, config['data']['embedding_path_zh']))
    en_embeddings = load_pretrained_embeddings(en_vocab,
                                               os.path.join(PROJECT_ROOT, config['data']['embedding_path_en']))

    # Initialize model
    model = init_model(config, len(zh_vocab), len(en_vocab), zh_embeddings, en_embeddings).to(device)

    # Load model
    model_path = os.path.join(PROJECT_ROOT, config['inference']['model_path'])
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from: {model_path}")

    # Load test data
    input_path = os.path.join(PROJECT_ROOT, config['data']['test_path'])
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    input_data = load_jsonl(input_path)

    results = []
    max_len = config['data']['max_len']
    num_visualize = config['inference']['num_visualize']

    for idx, sample in enumerate(input_data):
        zh_text = clean_text(sample['zh'])
        zh_tokens = jieba.lcut(zh_text)[:max_len]
        zh_indices = [zh_vocab.get(token, zh_vocab['<unk>']) for token in ['<sos>'] + zh_tokens + ['<eos>']]
        zh_indices = zh_indices + [zh_vocab['<pad>']] * (max_len + 2 - len(zh_indices))
        src = torch.tensor([zh_indices], dtype=torch.long).to(device)

        # Perform decoding based on decode_type
        if decode_type == 'greedy':
            pred_ids, attn_weights = greedy_decode(model, src, en_vocab, device, max_len)
        else:  # beam_search
            pred_ids, _, attn_weights = beam_search(model, src, en_vocab, device, beam_size, max_len)

        # Convert to text
        pred_text = tokens_to_sentence(pred_ids, en_vocab, is_chinese=False).split()
        src_tokens = ['<sos>'] + zh_tokens + ['<eos>']
        tgt_tokens = [idx_to_word.get(idx, '<unk>') for idx in pred_ids]

        # Calculate BLEU
        bleu_score = 0.0
        if 'en' in sample:
            ref_text = clean_text(sample['en']).lower().split()
            try:
                bleu_score = calculate_bleu([ref_text], pred_text)
            except Exception as e:
                print(f"BLEU error for sample {idx}: {e}")

        # Visualize attention for first num_visualize samples
        if idx < num_visualize:
            attn_save_path = os.path.join(PROJECT_ROOT, f'outputs/plots/sample_{idx}.png')
            visualize_attention(src_tokens, tgt_tokens, attn_weights, attn_save_path)
            print(f"Attention heatmap saved to: {attn_save_path}")

        # Store result
        result = {
            'bleu': bleu_score,
            'source': zh_text,
            'translation': ' '.join(pred_text)
        }
        if 'en' in sample:
            result['reference'] = sample['en']
        results.append(result)

    # Calculate and print average and highest BLEU scores
    bleu_scores = [result['bleu'] for result in results if result['bleu'] is not None]
    if bleu_scores:
        avg_bleu = np.mean(bleu_scores)
        max_bleu = max(bleu_scores)
        result = {
            'average_bleu': avg_bleu,
            'highest_bleu': max_bleu
        }
        results.append(result)
        print(f"Average BLEU: {avg_bleu:.4f}, Highest BLEU: {max_bleu:.4f}")
    else:
        print("No valid BLEU scores available.")

    attention_list = ["dot_product", "multiplicative", "additive"]  # 1 2 3
    train_list = ["teacher_forcing", "free_running", "mix_training"]  # 1 2 3
    data_list = ["data/train_10k.jsonl", "data/train_100k.jsonl"]  # 1 2 3
    decode_list = ["greedy", "beam_search"]  # 1 2
    attention_type = attention_list.index(config['model']['attention_type']) + 1
    train_type = train_list.index(config['training']['train_type']) + 1
    data_type = data_list.index(config['data']['train_path']) + 1
    decode_type = decode_list.index(config['inference']['decode_type']) + 1
    # Save results

    output_path = os.path.join(PROJECT_ROOT, f"outputs/translations/inference_results_"
                                             f"{attention_type}_{train_type}_{data_type}_{decode_type}.jsonl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    return results


if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_path = os.path.join(PROJECT_ROOT, 'config.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = load_config(config_path)

    results = inference(config)
