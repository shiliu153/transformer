import math
import os
from collections import Counter
import jieba
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 0. 辅助函数：从文件加载句子
def load_sentences_from_file(filepath):
    sentences = []
    print(f"Loading sentences from {filepath}...")
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Reading {os.path.basename(filepath)}"):
            sentences.append(line.strip())
    print(f"Loaded {len(sentences)} sentences from {filepath}.")
    return sentences

# 1. 数据处理类
class Vocab:
    def __init__(self, freq_threshold=2):
        # 初始化特殊标记：PAD用于填充，SOS是序列开始，EOS是序列结束，UNK是未知词
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    def build_vocab(self, sentence_list, tokenizer):
        frequencies = Counter()
        idx = len(self.itos)  # 从特殊标记之后开始索引

        print("Building vocabulary...")
        for sentence in tqdm(sentence_list, desc="Processing sentences for vocab"):
            frequencies.update(tokenizer(sentence))

        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                if word not in self.stoi: # 确保不覆盖特殊标记
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1
        print(f"Vocabulary built. Size: {len(self.itos)}")

    def numericalize(self, text, tokenizer):
        tokenized_text = tokenizer(text)
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokenized_text]

def tokenize_zh(text):
    return jieba.lcut(text)

def tokenize_en(text):
    return text.lower().split()

def simple_space_tokenizer(text):
    return text.split()

class TranslationDataset(Dataset):
    def __init__(self, src_sentences, trg_sentences, src_vocab, trg_vocab,
                 src_tokenizer, trg_tokenizer, max_len=100):
        self.src_sentences = src_sentences
        self.trg_sentences = trg_sentences
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, index):
        src_sentence = self.src_sentences[index]
        trg_sentence = self.trg_sentences[index]

        src_tokens = [self.src_vocab.stoi["<SOS>"]] + \
                     self.src_vocab.numericalize(src_sentence, self.src_tokenizer) + \
                     [self.src_vocab.stoi["<EOS>"]]
        trg_tokens = [self.trg_vocab.stoi["<SOS>"]] + \
                     self.trg_vocab.numericalize(trg_sentence, self.trg_tokenizer) + \
                     [self.trg_vocab.stoi["<EOS>"]]

        src_tokens = src_tokens[:self.max_len]
        trg_tokens = trg_tokens[:self.max_len]

        # 填充
        src_padded = src_tokens + [self.src_vocab.stoi["<PAD>"]] * (self.max_len - len(src_tokens))
        trg_padded = trg_tokens + [self.trg_vocab.stoi["<PAD>"]] * (self.max_len - len(trg_tokens))

        return torch.tensor(src_padded), torch.tensor(trg_padded)

# 3. 模型组件
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.d_k])).to(device)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)
        x = torch.matmul(self.dropout(attention), V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.d_model)
        x = self.fc_o(x)
        return x, attention

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc_1 = nn.Linear(d_model, d_ff)
        self.fc_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(torch.relu(self.fc_1(x)))
        x = self.fc_2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        _src, _ = self.self_attn(src, src, src, src_mask)
        src = self.norm1(src + self.dropout(_src))
        _src = self.ff(src)
        src = self.norm2(src + self.dropout(_src))
        return src

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.enc_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        _trg, _ = self.self_attn(trg, trg, trg, trg_mask)
        trg = self.norm1(trg + self.dropout(_trg))
        _trg, attention = self.enc_attn(trg, enc_src, enc_src, src_mask)
        trg = self.norm2(trg + self.dropout(_trg))
        _trg = self.ff(trg)
        trg = self.norm3(trg + self.dropout(_trg))
        return trg, attention

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# 4. Transformer 模型
class Encoder(nn.Module):
    def __init__(self, input_vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_len, device):
        super().__init__()
        self.device = device
        self.d_model = d_model # Store d_model for PositionalEncoding scaling
        self.tok_embedding = nn.Embedding(input_vocab_size, d_model)
        self.pos_embedding = PositionalEncoding(d_model, dropout, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout)
                                     for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        src_emb = self.tok_embedding(src)
        src_emb = self.pos_embedding(src_emb)
        for layer in self.layers:
            src_emb = layer(src_emb, src_mask)
        return src_emb

class Decoder(nn.Module):
    def __init__(self, output_vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_len, device):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.tok_embedding = nn.Embedding(output_vocab_size, d_model)
        self.pos_embedding = PositionalEncoding(d_model, dropout, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout)
                                     for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, output_vocab_size)
        self.dropout = nn.Dropout(dropout)


    def forward(self, trg, enc_src, trg_mask, src_mask):
        trg_emb = self.tok_embedding(trg)
        trg_emb = self.pos_embedding(trg_emb)
        for layer in self.layers:
            trg_emb, attention = layer(trg_emb, enc_src, trg_mask, src_mask)
        output = self.fc_out(trg_emb)
        return output, attention

class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        # src_mask: [batch_size, 1, 1, src_len]
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output, attention

# 5. 训练和评估函数
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_epoch(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(tqdm(iterator, desc="Training Epoch", leave=False)):
        src, trg = batch
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output, _ = model(src, trg[:, :-1])
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg_y = trg[:, 1:].contiguous().view(-1)
        loss = criterion(output, trg_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def evaluate_epoch(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(iterator, desc="Evaluating Epoch", leave=False)):
            src, trg = batch
            src, trg = src.to(device), trg.to(device)
            output, _ = model(src, trg[:, :-1])
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg_y = trg[:, 1:].contiguous().view(-1)
            loss = criterion(output, trg_y)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

# 6. 翻译函数
def translate_sentence(sentence, src_vocab, trg_vocab, model, src_tokenizer, device, max_len=100):
    model.eval()

    if isinstance(sentence, str):
        tokens_numericalized = src_vocab.numericalize(sentence, src_tokenizer)
    elif isinstance(sentence, list):
        tokens_numericalized = [src_vocab.stoi.get(token, src_vocab.stoi["<UNK>"]) for token in sentence]
    else:
        raise ValueError("Input 'sentence' must be a string or a list of tokens.")

    tokens = [src_vocab.stoi["<SOS>"]] + tokens_numericalized + [src_vocab.stoi["<EOS>"]]
    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)
    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [trg_vocab.stoi["<SOS>"]]
    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        trg_mask = model.make_trg_mask(trg_tensor)
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        pred_token = output.argmax(2)[:, -1].item()
        trg_indexes.append(pred_token)
        if pred_token == trg_vocab.stoi["<EOS>"]:
            break

    trg_tokens = [trg_vocab.itos[i] for i in trg_indexes if i in trg_vocab.itos]
    # 移除 <SOS> 和 <EOS>
    if trg_tokens and trg_tokens[0] == "<SOS>":
        trg_tokens = trg_tokens[1:]
    if trg_tokens and trg_tokens[-1] == "<EOS>":
        trg_tokens = trg_tokens[:-1]
    return trg_tokens, attention


# 7. 主程序
if __name__ == '__main__':

    D_MODEL = 256
    NUM_HEADS = 8
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3
    D_FF = 512
    DROPOUT = 0.1
    MAX_LEN = 100
    LEARNING_RATE = 0.0005
    BATCH_SIZE = 64 # 根据您的 GPU 内存调整
    NUM_EPOCHS = 10 # 为了演示，减少 epoch 数量
    CLIP = 1
    FREQ_THRESHOLD = 5 # 词汇表构建的频率阈值
    DATA_SUBSET_FRACTION = 0.01 # 使用更小的数据子集进行快速测试，例如1%

    # --- 文件和目录路径 ---
    data_dir = './data/traindata'
    model_save_dir = './saved_models' # 建议将模型和词汇表保存在单独的目录
    os.makedirs(model_save_dir, exist_ok=True) # 创建保存目录

    train_zh_filepath = os.path.join(data_dir, 'train.zh')
    train_en_filepath = os.path.join(data_dir, 'train.en')
    val_zh_filepath = os.path.join(data_dir, 'valid.en-zh.zh.sgm')
    val_en_filepath = os.path.join(data_dir, 'valid.en-zh.en.sgm')

    # --- 词汇表和模型保存路径 ---
    src_vocab_path = os.path.join(model_save_dir, 'src_vocab.pth')
    trg_vocab_path = os.path.join(model_save_dir, 'trg_vocab.pth')
    model_save_path = os.path.join(model_save_dir, 'transformer-nmt-model-best.pt')


    train_data_zh_raw = load_sentences_from_file(train_zh_filepath)
    train_data_en_raw = load_sentences_from_file(train_en_filepath)

    val_data_zh_raw = load_sentences_from_file(val_zh_filepath)
    val_data_en_raw = load_sentences_from_file(val_en_filepath)

    # --- 数据子集处理 (可选，用于快速测试) ---
    if 0.0 < DATA_SUBSET_FRACTION < 1.0:
        print(f"Using {DATA_SUBSET_FRACTION*100:.0f}% of the loaded data.")

        train_zh_len = len(train_data_zh_raw)
        train_en_len = len(train_data_en_raw)
        # 确保两个列表长度一致
        min_train_len = min(train_zh_len, train_en_len)
        train_data_zh_raw = train_data_zh_raw[:min_train_len]
        train_data_en_raw = train_data_en_raw[:min_train_len]
        desired_train_subset_size = max(1, int(min_train_len * DATA_SUBSET_FRACTION))
        train_data_zh_raw = train_data_zh_raw[:desired_train_subset_size]
        train_data_en_raw = train_data_en_raw[:desired_train_subset_size]
        print(f"New training data size: {len(train_data_zh_raw)} sentence pairs.")

        val_zh_len = len(val_data_zh_raw)
        val_en_len = len(val_data_en_raw)
        min_val_len = min(val_zh_len, val_en_len)
        val_data_zh_raw = val_data_zh_raw[:min_val_len]
        val_data_en_raw = val_data_en_raw[:min_val_len]
        desired_val_subset_size = max(1, int(min_val_len * DATA_SUBSET_FRACTION))
        val_data_zh_raw = val_data_zh_raw[:desired_val_subset_size]
        val_data_en_raw = val_data_en_raw[:desired_val_subset_size]
        print(f"New validation data size: {len(val_data_zh_raw)} sentence pairs.")

    elif DATA_SUBSET_FRACTION == 1.0:
        print("Using 100% of the loaded data.")
    elif DATA_SUBSET_FRACTION <= 0.0:
        print("Error: DATA_SUBSET_FRACTION must be greater than 0. Exiting.")
        exit()
    if not train_data_zh_raw or not train_data_en_raw:
        print("Error: Training data is empty after subset selection or initial loading. Exiting.")
        exit()


    print("Initializing tokenizers...")
    try:
        jieba.lcut("预热jieba", cut_all=False)
        print("Jieba initialized.")
        src_tokenizer_fn = tokenize_zh
    except NameError:
        print("jieba not imported. Please ensure it's installed and imported.")
        print("Falling back to simple_space_tokenizer for Chinese. This is not recommended.")
        src_tokenizer_fn = simple_space_tokenizer
    except Exception as e:
        print(f"Error initializing jieba: {e}")
        print("Falling back to simple_space_tokenizer for Chinese. This is not recommended.")
        src_tokenizer_fn = simple_space_tokenizer


    trg_tokenizer_fn = tokenize_en

    print("Building vocabularies...")
    src_vocab = Vocab(freq_threshold=FREQ_THRESHOLD)
    trg_vocab = Vocab(freq_threshold=FREQ_THRESHOLD)

    # 使用训练数据构建词汇表
    # 确保 train_data_zh_raw 和 train_data_en_raw 在这里是非空的
    if train_data_zh_raw:
        src_vocab.build_vocab(train_data_zh_raw, src_tokenizer_fn)
        torch.save(src_vocab, src_vocab_path) # 保存源语言词汇表
        print(f"Source vocabulary saved to '{src_vocab_path}'")
    else:
        print("Skipping source vocab build and save due to empty training data.")
        # 如果没有训练数据，后续步骤可能会失败，这里应该考虑退出或使用预加载的词汇表

    if train_data_en_raw:
        trg_vocab.build_vocab(train_data_en_raw, trg_tokenizer_fn)
        torch.save(trg_vocab, trg_vocab_path) # 保存目标语言词汇表
        print(f"Target vocabulary saved to '{trg_vocab_path}'")
    else:
        print("Skipping target vocab build and save due to empty training data.")


    print(f"Source Vocab Size: {len(src_vocab)}")
    print(f"Target Vocab Size: {len(trg_vocab)}")


    SRC_PAD_IDX = src_vocab.stoi["<PAD>"]
    TRG_PAD_IDX = trg_vocab.stoi["<PAD>"]
    INPUT_DIM = len(src_vocab)
    OUTPUT_DIM = len(trg_vocab)

    min_vocab_size = 4 # <PAD>, <SOS>, <EOS>, <UNK>
    if INPUT_DIM <= min_vocab_size or OUTPUT_DIM <= min_vocab_size:
        print(f"Warning: Vocab size is very small (SRC: {INPUT_DIM}, TRG: {OUTPUT_DIM}).")
        print("This might be due to small dataset, high freq_threshold, or issues with data loading/subsetting.")
        print("Consider reducing freq_threshold or using more data.")
        if not train_data_zh_raw or not train_data_en_raw:
             print("Continuing with potentially small vocab due to empty or very small dataset.")
        # 可以考虑在这里添加一个更强的检查，如果词汇表真的太小，可能无法训练
        if INPUT_DIM <= min_vocab_size and len(train_data_zh_raw) > 0 : # 有数据但词汇表小
            print("Problem: Source vocab is too small despite having training data. Check tokenization and freq_threshold.")
        if OUTPUT_DIM <= min_vocab_size and len(train_data_en_raw) > 0: # 有数据但词汇表小
            print("Problem: Target vocab is too small despite having training data. Check tokenization and freq_threshold.")


    print("Creating datasets and dataloaders...")
    # 确保 val_data_zh_raw 和 val_data_en_raw 也是有效的，如果它们是空的，DataLoader 会出问题
    if not val_data_zh_raw or not val_data_en_raw:
        print("Warning: Validation data is empty. Evaluation will not be performed.")
        val_iterator = None # 或者创建一个空的 DataLoader
    else:
        val_dataset = TranslationDataset(val_data_zh_raw, val_data_en_raw, src_vocab, trg_vocab,
                                         src_tokenizer_fn, trg_tokenizer_fn, MAX_LEN)
        val_iterator = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


    train_dataset = TranslationDataset(train_data_zh_raw, train_data_en_raw, src_vocab, trg_vocab,
                                       src_tokenizer_fn, trg_tokenizer_fn, MAX_LEN)

    num_workers_to_use = 0
    if os.name == 'posix': # num_workers > 0 在 Windows 上有时会有问题
        num_workers_to_use = 2

    train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers_to_use)


    # 初始化模型
    print("Initializing model...")
    enc = Encoder(INPUT_DIM, D_MODEL, NUM_ENCODER_LAYERS, NUM_HEADS, D_FF, DROPOUT, MAX_LEN, device)
    dec = Decoder(OUTPUT_DIM, D_MODEL, NUM_DECODER_LAYERS, NUM_HEADS, D_FF, DROPOUT, MAX_LEN, device)
    model = Transformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
    model.apply(initialize_weights)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    best_valid_loss = float('inf')

    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_iterator, optimizer, criterion, CLIP)

        if val_iterator: # 仅当有验证数据时才进行评估
            valid_loss = evaluate_epoch(model, val_iterator, criterion)
            print(f'Epoch: {epoch + 1:02} | Train Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}')
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), model_save_path)
                print(f"Epoch {epoch + 1}: New best validation loss: {valid_loss:.3f}. Model saved to '{model_save_path}'")
        else: # 没有验证数据，只打印训练损失，并保存每轮的模型（或最后一轮）
            print(f'Epoch: {epoch + 1:02} | Train Loss: {train_loss:.3f} | (No validation)')
            # 如果没有验证集，可以考虑每轮都保存或只保存最后一轮的模型
            # torch.save(model.state_dict(), model_save_path.replace(".pt", f"_epoch{epoch+1}.pt"))
            if epoch == NUM_EPOCHS -1: # 保存最后一轮的模型
                 torch.save(model.state_dict(), model_save_path)
                 print(f"Training finished. Model saved to '{model_save_path}' (last epoch).")


    if not val_iterator and NUM_EPOCHS > 0: # 如果没有验证，确保模型被保存了
        if not os.path.exists(model_save_path): # 如果因为某种原因最后一轮没保存
            torch.save(model.state_dict(), model_save_path)
            print(f"Final model state saved to '{model_save_path}' as no validation was performed.")

    print("Training finished.")

    # 加载最佳模型进行测试 (如果之前有保存)
    try:
        if os.path.exists(model_save_path):
            model.load_state_dict(torch.load(model_save_path, map_location=device))
            print(f"Loaded model from '{model_save_path}' for translation.")
        else:
            print(f"Model file '{model_save_path}' not found. Using the current model state for translation (which might be the last epoch's).")
    except Exception as e:
        print(f"Error loading model: {e}. Using the current model state for translation.")

    test_sentences_zh = [
        "你好 世界",
        "我 想 了解 更多 信息",
        "今天 天气 不错"
    ]
    print("\n--- Starting Translation Tests ---")
    for test_sentence_zh in test_sentences_zh:
        print(f"\nTranslating (ZH): '{test_sentence_zh}'")
        if len(src_vocab) > min_vocab_size and len(trg_vocab) > min_vocab_size:
            translation, attention = translate_sentence(test_sentence_zh, src_vocab, trg_vocab, model,
                                                        src_tokenizer_fn, device, MAX_LEN)
            print(f"Translated (EN): {' '.join(translation)}")
        else:
            print("Skipping translation due to small/invalid vocabulary.")


    print("\n--- Translation Tests Finished ---")