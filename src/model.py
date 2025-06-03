import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class EncoderGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout, pretrained_embeddings):
        super(EncoderGRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embeddings is not None:
            self.embedding.weight = nn.Parameter(pretrained_embeddings)
            self.embedding.weight.requires_grad = True  # 允许微调词嵌入
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: (batch_size, seq_len)
        embedded = self.dropout(self.embedding(src))  # (batch_size, seq_len, embedding_dim)
        outputs, hidden = self.gru(embedded)  # outputs: (batch_size, seq_len, hidden_size)
        # hidden: (num_layers, batch_size, hidden_size)
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, hidden_size, attention_type='dot_product'):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention_type = attention_type
        if attention_type == 'multiplicative':
            self.W = nn.Linear(hidden_size, hidden_size, bias=False)
        elif attention_type == 'additive':
            self.W1 = nn.Linear(hidden_size, hidden_size, bias=False)
            self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
            self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: (batch_size, hidden_size) [last layer]
        # encoder_outputs: (batch_size, seq_len, hidden_size)
        batch_size, seq_len, _ = encoder_outputs.size()

        # 调试形状
        assert decoder_hidden.dim() == 2, f"decoder_hidden must be 2D, got {decoder_hidden.shape}"
        assert encoder_outputs.dim() == 3, f"encoder_outputs must be 3D, got {encoder_outputs.shape}"

        if self.attention_type == 'dot_product':
            # score = h_t^T * h_s
            decoder_hidden_3d = decoder_hidden.unsqueeze(1)  # (batch_size, 1, hidden_size)
            encoder_outputs_t = encoder_outputs.transpose(1, 2)  # (batch_size, hidden_size, seq_len)
            scores = torch.bmm(decoder_hidden_3d, encoder_outputs_t)  # (batch_size, 1, seq_len)
            scores = scores.squeeze(1)
        elif self.attention_type == 'multiplicative':
            decoder_hidden_3d = decoder_hidden.unsqueeze(1)
            encoder_outputs = self.W(encoder_outputs)  # (batch_size, seq_len, hidden_size)
            encoder_outputs_t = encoder_outputs.transpose(1, 2)
            scores = torch.bmm(decoder_hidden_3d, encoder_outputs_t)
            scores = scores.squeeze(1)
        elif self.attention_type == 'additive':
            decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)  # (batch_size, seq_len, hidden_size)
            scores = self.v(torch.tanh(self.W1(decoder_hidden) + self.W2(encoder_outputs)))  # (batch_size, seq_len, 1)
            scores = scores.squeeze(-1)

        # 确保 scores 是 2D
        assert scores.dim() == 2, f"scores must be 2D, got {scores.shape}"

        attn_weights = F.softmax(scores, dim=-1)  # (batch_size, seq_len)
        assert attn_weights.dim() == 2, f"attn_weights must be 2D, got {attn_weights.shape}"
        # 计算上下文向量
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # (batch_size, 1, hidden_size)

        # context: (batch_size, hidden_size), attn_weights: (batch_size, seq_len)
        return context.squeeze(1), attn_weights



class DecoderGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout, pretrained_embeddings,
                 attention_type):
        super(DecoderGRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embeddings is not None:
            self.embedding.weight = nn.Parameter(pretrained_embeddings)
            self.embedding.weight.requires_grad = True
        self.gru = nn.GRU(embedding_dim + hidden_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.attention = Attention(hidden_size, attention_type)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, hidden, encoder_outputs):
        # tgt: (batch_size, 1) [single token]
        # hidden: (num_layers, batch_size, hidden_size)
        # encoder_outputs: (batch_size, seq_len, hidden_size)
        embedded = self.dropout(self.embedding(tgt))  # (batch_size, 1, embedding_dim)

        # Attention
        decoder_hidden = hidden[-1]  # 取最后一层隐藏状态 (batch_size, hidden_size)
        context, attn_weights = self.attention(decoder_hidden, encoder_outputs)  # context: (batch_size, hidden_size)

        # 拼接嵌入和上下文向量
        gru_input = torch.cat([embedded.squeeze(1), context], dim=-1)  # (batch_size, embedding_dim + hidden_size)
        gru_input = gru_input.unsqueeze(1)  # (batch_size, 1, embedding_dim + hidden_size)

        # GRU 解码
        output, hidden = self.gru(gru_input, hidden)  # output: (batch_size, 1, hidden_size)
        # 预测下一个词
        prediction = self.fc(output.squeeze(1))  # (batch_size, vocab_size)
        return prediction, hidden, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, teacher_forcing_ratio):
        # src: (batch_size, src_len)
        # tgt: (batch_size, tgt_len)
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        vocab_size = self.decoder.fc.out_features

        # 保存输出和注意力权重
        outputs = torch.zeros(batch_size, tgt_len, vocab_size).to(src.device)
        attn_weights_all = torch.zeros(batch_size, tgt_len, src.size(1)).to(src.device)

        # 编码
        encoder_outputs, hidden = self.encoder(src)

        # 解码器初始输入为 <sos>
        input = tgt[:, 0].unsqueeze(1)  # (batch_size, 1)

        for t in range(1, tgt_len):
            output, hidden, attn_weights = self.decoder(input, hidden, encoder_outputs)
            outputs[:, t] = output
            attn_weights_all[:, t] = attn_weights

            # Teacher Forcing 或 Free Running
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1).unsqueeze(1)  # (batch_size, 1)
            input = tgt[:, t].unsqueeze(1) if teacher_force else top1

        return outputs, attn_weights_all


def init_model(config, zh_vocab_size, en_vocab_size, zh_embeddings, en_embeddings):
    """Initialize Seq2Seq model with specified attention type."""
    hidden_size = config['model']['hidden_size']
    num_layers = config['model']['num_layers']
    dropout = config['model']['dropout']
    attention_type = config['model']['attention_type']

    encoder = EncoderGRU(
        vocab_size=zh_vocab_size,
        embedding_dim=config['model']['embedding_dim'],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        pretrained_embeddings=zh_embeddings
    )
    decoder = DecoderGRU(
        vocab_size=en_vocab_size,
        embedding_dim=config['model']['embedding_dim'],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        pretrained_embeddings=en_embeddings,
        attention_type=attention_type
    )
    model = Seq2Seq(encoder, decoder)
    return model
