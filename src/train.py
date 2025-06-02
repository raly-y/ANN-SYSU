import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from src.utils import load_config, calculate_bleu, log_results, set_seed, tokens_to_sentence
from src.data_preprocess import preprocess_data
from src.model import init_model
import pickle

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def train_epoch(model, train_loader, optimizer, criterion, device, teacher_forcing_ratio, epoch, total_epochs, train_type):
    """Train one epoch with Teacher Forcing."""
    model.train()
    total_loss = 0
    start_time = time.time()
    if train_type == "mix_training":
        tf_ratio = teacher_forcing_ratio * (1 - epoch / total_epochs)   # 动态调整 teacher_forcing_ratio，随 epoch 减少
    elif train_type == "free_running":
        tf_ratio = 0.0
    else:
        tf_ratio = 1.0
    print(f"Epoch {epoch + 1}: Using teacher_forcing_ratio={tf_ratio:.3f}")

    for batch in train_loader:
        src = batch['zh_indices'].to(device, dtype=torch.long)  # (batch_size, seq_len)
        tgt = batch['en_indices'].to(device, dtype=torch.long)  # (batch_size, seq_len)

        # 调试输入形状
        assert src.dim() == 2, f"src must be 2D, got {src.shape}"
        assert tgt.dim() == 2, f"tgt must be 2D, got {tgt.shape}"

        optimizer.zero_grad()
        output, _ = model(src, tgt, tf_ratio)  # (batch_size, tgt_len, vocab_size)
        output = output[:, 1:].reshape(-1, output.size(-1))  # 忽略 <sos>
        tgt = tgt[:, 1:].reshape(-1)  # 忽略 <sos>

        loss = criterion(output, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
        optimizer.step()
        total_loss += loss.item()

    epoch_time = time.time() - start_time
    return total_loss / len(train_loader), epoch_time


def evaluate(model, valid_loader, criterion, device, en_vocab):
    model.eval()
    total_loss = 0
    bleu_scores = []
    start_time = time.time()

    with torch.no_grad():
        for batch in valid_loader:
            src = batch['zh_indices'].to(device, dtype=torch.long)  # (batch_size, seq_len)
            tgt = batch['en_indices'].to(device, dtype=torch.long)  # (batch_size, seq_len)
            en_text = batch['en_text']

            assert src.dim() == 2, f"src must be 2D, got {src.shape}"
            assert tgt.dim() == 2, f"tgt must be 2D, got {tgt.shape}"

            output, _ = model(src, tgt, teacher_forcing_ratio=0.0)  # (batch_size, tgt_len, vocab_size)
            output_for_loss = output[:, 1:].reshape(-1, output.size(-1))  # 忽略 <sos>
            tgt_for_loss = tgt[:, 1:].reshape(-1)  # 忽略 <sos>
            loss = criterion(output_for_loss, tgt_for_loss)
            total_loss += loss.item()

            # 计算 BLEU
            batch_size = src.size(0)
            output_pred = output[:, 1:]  # (batch_size, tgt_len-1, vocab_size)
            for i in range(batch_size):
                pred_ids = output_pred[i].argmax(-1).cpu().tolist()  # (tgt_len-1,)
                ref_text = en_text[i].lower().split()  # 参考文本
                pred_text = tokens_to_sentence(pred_ids, en_vocab, is_chinese=False).split()

                if pred_text and ref_text and len(pred_text) > 1:
                    try:
                        bleu = calculate_bleu([ref_text], pred_text)
                        bleu_scores.append(bleu)
                    except Exception as e:
                        print(f"BLEU error: {e}, pred_text={pred_text}, ref_text={ref_text}")

    eval_time = time.time() - start_time
    avg_loss = total_loss / len(valid_loader)
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    return avg_loss, avg_bleu, eval_time


def train(config):
    """Main training function."""
    # 设置随机种子
    set_seed(config.get('training', {}).get('seed', 42))

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载数据
    start_time = time.time()
    data = preprocess_data(config)
    preprocess_time = time.time() - start_time
    print(f"Data preprocessing time: {preprocess_time:.2f} seconds")
    train_loader = data['train_loader']
    valid_loader = data['valid_loader']
    zh_vocab = data['zh_vocab']
    en_vocab = data['en_vocab']

    os.makedirs(os.path.join(PROJECT_ROOT, 'outputs/vocab'), exist_ok=True)
    with open(os.path.join(PROJECT_ROOT, 'outputs/vocab/zh_vocab.pkl'), 'wb') as f:
        pickle.dump(zh_vocab, f)
    with open(os.path.join(PROJECT_ROOT, 'outputs/vocab/en_vocab.pkl'), 'wb') as f:
        pickle.dump(en_vocab, f)

    # 初始化模型
    model = init_model(config, len(zh_vocab), len(en_vocab),
                       data['zh_embeddings'], data['en_embeddings']).to(device)
    model = model.float()

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=en_vocab['<pad>'])
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=1e-5)

    # 学习率调度（可选）
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                                     patience=2)

    # 日志和模型保存路径
    log_path = os.path.join(PROJECT_ROOT, "outputs/logs/train.log")
    model_log_path = os.path.join(PROJECT_ROOT, "outputs/logs/model.log")
    models_dir = os.path.join(PROJECT_ROOT, "outputs/models")
    plots_dir = os.path.join(PROJECT_ROOT, "outputs/plots")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_log_path), exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # 生成唯一模型文件名
    attention_list = ["dot_product", "multiplicative", "additive"]      # 1 2 3
    train_list = ["teacher_forcing", "free_running", "mix_training"]    # 1 2 3
    data_list = ["data/train_10k.jsonl", "data/train_100k.jsonl"]       # 1 2 3
    attention_type = attention_list.index(config['model']['attention_type']) + 1
    train_type = train_list.index(config['training']['train_type']) + 1
    data_type = data_list.index(config['data']['train_path']) + 1
    best_model_path = os.path.join(models_dir, f"model_{attention_type}_{train_type}_{data_type}.pt")

    best_bleu = 0.0
    total_train_time = 0.0
    total_eval_time = 0.0
    patience = 5
    last_bleu = 0.0
    counter = 0
    total_epoch = 0
    train_losses = []
    valid_losses = []
    valid_bleus = []

    for epoch in range(config['training']['epochs']):
        # 训练
        train_loss, train_time = train_epoch(model, train_loader, optimizer, criterion, device,
                                             config['training']['teacher_forcing_ratio'], epoch,
                                             config['training']['epochs'], config['training']['train_type'])
        # 验证
        valid_loss, valid_bleu, eval_time = evaluate(model, valid_loader, criterion, device, en_vocab)

        total_train_time += train_time
        total_eval_time += eval_time

        # 记录指标
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        valid_bleus.append(valid_bleu)

        # 学习率调度
        scheduler.step(valid_bleu)

        # 日志记录
        log_results({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'valid_bleu': valid_bleu,
            'train_time': train_time,
            'eval_time': eval_time
        }, log_path)

        # 保存最佳模型
        if valid_bleu > best_bleu:
            best_bleu = valid_bleu
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with BLEU: {valid_bleu:.4f}")

        if valid_bleu <= last_bleu + 0.00014:
            counter += 1
        else:
            counter = 0

        last_bleu = valid_bleu

        print(f"Epoch {epoch + 1}/{config['training']['epochs']}, "
              f"Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Valid BLEU: {valid_bleu:.4f}, "
              f"Train Time: {train_time:.2f}s, Eval Time: {eval_time:.2f}s, "
              f"Counter: {counter}")
        total_epoch = epoch + 1

        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Plot Loss Curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, total_epoch + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, total_epoch + 1), valid_losses, label='Valid Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f"loss_curve_{attention_type}_{train_type}_{data_type}.png"))
    plt.close()
    print(f"Loss curve saved to: {os.path.join(plots_dir,f'loss_curve_{attention_type}_{train_type}_{data_type}.png')}")

    # Plot BLEU Curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, total_epoch + 1), valid_bleus, label='Valid BLEU', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('BLEU Score')
    plt.title('Validation BLEU Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f"bleu_curve_{attention_type}_{train_type}_{data_type}.png"))
    plt.close()
    print(f"BLEU curve saved to: {os.path.join(plots_dir,f'bleu_curve_{attention_type}_{train_type}_{data_type}.png')}")

    log_results({
        'best_bleu': best_bleu,
        'train_path': config['data']['train_path'],
        'batch_size': config['training']['batch_size'],
        'learning_rate': config['training']['learning_rate'],
        'epoch': config['training']['epochs'],
        'dropout': config['model']['dropout'],
        'attention_type': attention_type,
        'train_type': train_type
    }, model_log_path)

    print(f"Training completed. Best BLEU: {best_bleu:.4f}")
    print(f"Total time: {total_train_time + total_eval_time:.2f} seconds")

if __name__ == "__main__":
    config_path = os.path.join(PROJECT_ROOT, "config.yaml")
    config = load_config(config_path)
    train(config)