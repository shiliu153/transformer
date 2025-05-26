import torch
import torch.nn as nn
import math


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_len, device):
        super().__init__()
        self.device = device
        self.d_model = d_model

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len)

        # Decoder-only 模型的核心是由多个 DecoderLayer 堆叠而成
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout, device)
            for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src_tokens, src_mask=None):
        # src_tokens: (batch_size, seq_len)

        token_embedded = self.token_embedding(src_tokens) * math.sqrt(self.d_model)
        # token_embedded: (batch_size, seq_len, d_model)

        positioned_embedded = self.positional_encoding(token_embedded)
        # positioned_embedded: (batch_size, seq_len, d_model)

        output = self.dropout(positioned_embedded)

        # Decoder-only 模型中，src_mask 通常是因果掩码 (causal mask)
        # 以确保在预测当前词时，模型只能关注到前面的词。
        if src_mask is None:
            # 创建因果掩码
            seq_len = src_tokens.size(1)
            # src_mask 的形状应该是 (batch_size, num_heads, seq_len, seq_len) 或 (batch_size, 1, seq_len, seq_len)
            # 对于 Decoder-only，通常是 (seq_len, seq_len) 然后广播
            # 或者直接在 MultiHeadAttentionLayer 内部处理
            # 这里我们先假设 MultiHeadAttentionLayer 会处理或接收一个 (seq_len, seq_len) 的掩码
            # 并且这个掩码会被应用到注意力分数上，使得未来的 token 被屏蔽。
            # 一个简单的因果掩码 (上三角为 True/1, 表示屏蔽)
            causal_mask = torch.triu(torch.ones((seq_len, seq_len), device=self.device), diagonal=1).bool()
            # 在 MultiHeadAttentionLayer 中，通常期望 mask 中为 True 的位置被填充为 -inf
            # 所以这里的 mask 传递方式和 Encoder-Decoder 中的 padding mask 可能略有不同
            # 具体取决于 MultiHeadAttentionLayer 的实现
            # 我们先传递一个概念上的 src_mask，具体实现可能需要调整
            # 对于 Decoder-only，这个 mask 主要用于 self-attention
            # 它的作用是防止看到未来的 token
            # (batch_size, 1, seq_len, seq_len)
            src_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # 扩展到 batch 和 head 维度

        for layer in self.decoder_layers:
            # 在 Decoder-only 模型中，DecoderLayer 的输入是前一层的输出
            # 并且它只进行 self-attention (没有 cross-attention)
            # 因此，DecoderLayer 的 forward 方法需要调整，只接收一个输入和对应的掩码
            output = layer(output, src_mask)  # 假设 DecoderLayer 调整为只接受一个输入和其掩码

        # output: (batch_size, seq_len, d_model)
        logits = self.fc_out(output)
        # logits: (batch_size, seq_len, vocab_size)
        return logits


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout_p, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)  # 不参与模型参数更新

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        # self.pe 的形状是 (1, max_len, d_model)
        # 我们需要截取到 x 的序列长度
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, device):
        super().__init__()
        self.device = device
        # Decoder-only 模型中的 DecoderLayer 只有一个 masked self-attention
        self.masked_self_attn = MultiHeadAttentionLayer(d_model, num_heads, dropout, device)
        self.norm1 = nn.LayerNorm(d_model)

        self.feed_forward = PositionwiseFeedforwardLayer(d_model, d_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, trg_mask):
        # trg: (batch_size, trg_len, d_model)
        # trg_mask: (batch_size, 1, trg_len, trg_len) or similar for causal masking

        # 1. Masked Self-Attention
        # _trg 是注意力层的输出, attn_weights 是注意力权重 (可选返回)
        _trg, _ = self.masked_self_attn(trg, trg, trg, trg_mask)  # Q, K, V 都是 trg

        # 2. Add & Norm
        trg = self.norm1(trg + self.dropout(_trg))

        # 3. Feed Forward
        _trg = self.feed_forward(trg)

        # 4. Add & Norm
        trg = self.norm2(trg + self.dropout(_trg))

        return trg


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout, device):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.device = device

        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        # query, key, value: (batch_size, seq_len, d_model)
        # mask: (batch_size, 1, seq_len, seq_len) or (batch_size, num_heads, seq_len, seq_len)
        # 对于 Decoder-only 的 self-attention, query_len == key_len == value_len

        batch_size = query.shape[0]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        # Q, K, V: (batch_size, seq_len, d_model)

        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # Q, K, V: (batch_size, num_heads, seq_len, head_dim)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        # energy: (batch_size, num_heads, seq_len, seq_len)

        if mask is not None:
            # 确保 mask 能够正确广播
            # 如果 mask 是 (batch_size, 1, seq_len, seq_len), 它会自动广播到 num_heads
            # 如果 mask 是 (seq_len, seq_len), 需要扩展
            if mask.dim() == 2:  # (seq_len, seq_len)
                mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
            energy = energy.masked_fill(mask == True, -1e10)  # 在 PyTorch 中, mask 为 True 的地方被填充

        attention = torch.softmax(energy, dim=-1)
        # attention: (batch_size, num_heads, seq_len, seq_len)

        attention = self.dropout(attention)

        x = torch.matmul(attention, V)
        # x: (batch_size, num_heads, seq_len, head_dim)

        x = x.permute(0, 2, 1, 3).contiguous()
        # x: (batch_size, seq_len, num_heads, head_dim)

        x = x.view(batch_size, -1, self.d_model)
        # x: (batch_size, seq_len, d_model)

        x = self.fc_o(x)
        # x: (batch_size, seq_len, d_model)

        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.fc_1 = nn.Linear(d_model, d_ff)
        self.fc_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()  # 或者 GELU

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = self.dropout(self.activation(self.fc_1(x)))
        # x: (batch_size, seq_len, d_ff)
        x = self.fc_2(x)
        # x: (batch_size, seq_len, d_model)
        return x


# --- 辅助函数和类 (可以从您现有的代码中复用或调整) ---
class Vocab:  # 假设您有类似的 Vocab 类
    def __init__(self, freq_threshold=0):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold
        # ... (build_vocab, numericalize 等方法)

    def __len__(self):
        return len(self.itos)

    def build_vocab(self, sentence_list, tokenizer_fn):
        frequencies = {}
        idx = len(self.itos)

        for sentence in sentence_list:
            for token in tokenizer_fn(sentence):
                frequencies[token] = frequencies.get(token, 0) + 1

        for token, freq in frequencies.items():
            if freq >= self.freq_threshold:
                if token not in self.stoi:
                    self.stoi[token] = idx
                    self.itos[idx] = token
                    idx += 1

    def numericalize(self, text, tokenizer_fn, add_sos=False, add_eos=False):
        tokenized_text = tokenizer_fn(text)
        numericalized_text = []
        if add_sos:
            numericalized_text.append(self.stoi["<SOS>"])
        for token in tokenized_text:
            numericalized_text.append(self.stoi.get(token, self.stoi["<UNK>"]))
        if add_eos:
            numericalized_text.append(self.stoi["<EOS>"])
        return numericalized_text


# 简单的分词器示例
def simple_tokenizer(text):
    return text.split()


# --- 主程序示例 ---
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 超参数
    VOCAB_SIZE = 1000  # 示例：根据您的数据调整
    D_MODEL = 256
    NUM_LAYERS = 3  # Decoder layers
    NUM_HEADS = 8
    D_FF = 512
    DROPOUT = 0.1
    MAX_LEN = 100  # 最大序列长度
    LEARNING_RATE = 0.0005
    BATCH_SIZE = 32
    NUM_EPOCHS = 10

    # 1. 准备数据 (这里用虚拟数据)
    # 对于 Decoder-only 模型，输入和目标通常是相同的序列，但目标会向左移动一位
    # 例如，输入: "<SOS> w1 w2 w3", 目标: "w1 w2 w3 <EOS>"
    dummy_sentences = [
                          "hello world how are you",
                          "this is a test sentence",
                          "another example for language model",
                          "pytorch is fun to use"
                      ] * 20  # 更多数据

    # 2. 构建词汇表
    vocab = Vocab(freq_threshold=1)  # 较低的阈值用于小数据集
    vocab.build_vocab(dummy_sentences, simple_tokenizer)
    VOCAB_SIZE = len(vocab)
    PAD_IDX = vocab.stoi["<PAD>"]
    print(f"Vocab size: {VOCAB_SIZE}")

    # 3. 创建模型实例
    model = DecoderOnlyTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        dropout=DROPOUT,
        max_len=MAX_LEN,
        device=device
    ).to(device)

    print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')

    # 4. 准备数据加载器
    # 对于语言模型，输入是 token_ids, 目标是 token_ids 左移一位
    input_data = []
    target_data = []
    for sentence in dummy_sentences:
        token_ids = vocab.numericalize(sentence, simple_tokenizer, add_sos=True, add_eos=True)
        if len(token_ids) > 1 and len(token_ids) <= MAX_LEN:  # 确保至少有两个token (SOS + token)
            input_seq = token_ids[:-1]
            target_seq = token_ids[1:]

            # Padding
            input_seq.extend([PAD_IDX] * (MAX_LEN - len(input_seq)))
            target_seq.extend([PAD_IDX] * (MAX_LEN - len(target_seq)))

            input_data.append(input_seq)
            target_data.append(target_seq)

    if not input_data:
        print("No data to train on after processing. Check MAX_LEN or data.")
        exit()

    input_tensor = torch.LongTensor(input_data).to(device)
    target_tensor = torch.LongTensor(target_data).to(device)

    from torch.utils.data import TensorDataset, DataLoader

    dataset = TensorDataset(input_tensor, target_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 5. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 6. 训练循环
    model.train()
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        for batch_idx, (src_batch, trg_batch) in enumerate(dataloader):
            # src_batch: (batch_size, MAX_LEN)
            # trg_batch: (batch_size, MAX_LEN)

            optimizer.zero_grad()

            # 前向传播
            # 对于 Decoder-only, 输入就是源序列 (src_batch)
            # 模型会内部生成因果掩码
            output_logits = model(src_batch)
            # output_logits: (batch_size, MAX_LEN, VOCAB_SIZE)

            # 计算损失
            # CrossEntropyLoss 期望 (N, C, d1, d2, ...) 和 (N, d1, d2, ...)
            # output_logits: (batch_size * MAX_LEN, VOCAB_SIZE)
            # trg_batch: (batch_size * MAX_LEN)
            loss = criterion(output_logits.view(-1, VOCAB_SIZE), trg_batch.view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 1 == 0:  # 调整打印频率
                print(
                    f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Batch [{batch_idx + 1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        print(f"Epoch {epoch + 1} finished. Average Loss: {epoch_loss / len(dataloader):.4f}")

    model.eval()


    def generate_text(prompt, max_new_tokens=20):
        tokens = vocab.numericalize(prompt, simple_tokenizer, add_sos=True, add_eos=False)
        generated_tokens = list(tokens)

        for _ in range(max_new_tokens):
            input_ids = torch.LongTensor([generated_tokens[-MAX_LEN:]]).to(device)  # 只取最后 MAX_LEN 个 token

            with torch.no_grad():
                logits = model(input_ids)  # (1, current_seq_len, vocab_size)

            # 取最后一个时间步的 logits 进行采样
            next_token_logits = logits[:, -1, :]  # (1, vocab_size)
            next_token_id = torch.argmax(next_token_logits, dim=-1).item()  # 贪心采样

            if next_token_id == vocab.stoi["<EOS>"]:
                break
            generated_tokens.append(next_token_id)

        return " ".join([vocab.itos.get(tok_id, "<UNK>") for tok_id in generated_tokens])


    print("\n--- Text Generation Example ---")
    test_prompt = "hello"
    generated_sequence = generate_text(test_prompt)
    print(f"Prompt: '{test_prompt}'")
    print(f"Generated: '{generated_sequence}'")

    test_prompt_2 = "this is a"
    generated_sequence_2 = generate_text(test_prompt_2)
    print(f"Prompt: '{test_prompt_2}'")
    print(f"Generated: '{generated_sequence_2}'")