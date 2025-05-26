import torch
import torch.nn as nn
import math
import jieba
import os
from collections import Counter
from tqdm import tqdm

# 0. 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1. 辅助函数：从文件加载句子 (用于重建词汇表)
def load_sentences_from_file(filepath):
    sentences = []
    print(f"Loading sentences from {filepath} for vocabulary...")
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}. Vocabulary might be incomplete or incorrect.")
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Reading {os.path.basename(filepath)} for vocab"):
            sentences.append(line.strip())
    print(f"Loaded {len(sentences)} sentences from {filepath}.")
    return sentences

# 2. 数据处理类
class Vocab:
    def __init__(self, freq_threshold=2):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    def build_vocab(self, sentence_list, tokenizer):
        frequencies = Counter()
        idx = len(self.itos)
        print("Building vocabulary...")
        for sentence in tqdm(sentence_list, desc="Processing sentences for vocab"):
            frequencies.update(tokenizer(sentence))
        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                if word not in self.stoi:
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

# 3. 模型组件 (与训练时定义一致)
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
        x = x * math.sqrt(self.d_model) # 与训练脚本中的 Encoder/Decoder 内部的 PositionalEncoding 应用方式一致
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

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

class Encoder(nn.Module):
    def __init__(self, input_vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_len, device):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.tok_embedding = nn.Embedding(input_vocab_size, d_model)
        self.pos_embedding = PositionalEncoding(d_model, dropout, max_len) # d_model is passed here
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout)
                                     for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        src_emb = self.tok_embedding(src)
        # PositionalEncoding's forward method handles the scaling by sqrt(d_model)
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
        self.pos_embedding = PositionalEncoding(d_model, dropout, max_len) # d_model is passed here
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout)
                                     for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, output_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        trg_emb = self.tok_embedding(trg)
        # PositionalEncoding's forward method handles the scaling by sqrt(d_model)
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

# 4. 翻译函数
def translate_sentence(sentence_text, src_vocab_obj, trg_vocab_obj, model_instance, src_tokenizer_fn,
                       device_to_use, max_output_len=100):
    model_instance.eval()

    if isinstance(sentence_text, str):
        tokens_numericalized = src_vocab_obj.numericalize(sentence_text, src_tokenizer_fn)
    elif isinstance(sentence_text, list): # 假设已经是分词后的列表
        tokens_numericalized = [src_vocab_obj.stoi.get(token, src_vocab_obj.stoi["<UNK>"]) for token in sentence_text]
    else:
        raise ValueError("Input 'sentence_text' must be a string or a list of tokens.")

    tokens = [src_vocab_obj.stoi["<SOS>"]] + tokens_numericalized + [src_vocab_obj.stoi["<EOS>"]]
    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device_to_use)
    src_mask = model_instance.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model_instance.encoder(src_tensor, src_mask)

    trg_indexes = [trg_vocab_obj.stoi["<SOS>"]]
    for i in range(max_output_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device_to_use)
        trg_mask = model_instance.make_trg_mask(trg_tensor)
        with torch.no_grad():
            output, attention = model_instance.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        pred_token = output.argmax(2)[:, -1].item()
        trg_indexes.append(pred_token)
        if pred_token == trg_vocab_obj.stoi["<EOS>"]:
            break

    trg_tokens = [trg_vocab_obj.itos[i] for i in trg_indexes if i in trg_vocab_obj.itos]
    # 移除 <SOS> 和 <EOS> (如果存在)
    if trg_tokens and trg_tokens[0] == "<SOS>":
        trg_tokens = trg_tokens[1:]
    if trg_tokens and trg_tokens[-1] == "<EOS>":
        trg_tokens = trg_tokens[:-1]
    return trg_tokens, attention


# 5. 主程序
if __name__ == '__main__':
    # --- 配置参数 (必须与训练时一致) ---
    D_MODEL = 256
    NUM_HEADS = 8
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3
    D_FF = 512
    DROPOUT = 0.1
    MAX_LEN = 100  # 用于 PositionalEncoding 和翻译函数中的 max_output_len
    FREQ_THRESHOLD = 5 # 用于重建词汇表

    # --- 文件路径 ---
    # !! 修改为你实际的模型路径 !!
    model_load_path = 'transformer-nmt-model-best-ai-challenger.pt'
    # !! 修改为你的原始训练数据路径 (用于重建词汇表) !!
    # 你只需要提供训练时用于构建词汇表的文件
    data_dir = './data/traindata' # 假设和训练脚本中的目录结构一致
    train_zh_filepath = os.path.join(data_dir, 'train.zh')
    train_en_filepath = os.path.join(data_dir, 'train.en')

    # --- 初始化分词器 ---
    print("Initializing tokenizers...")
    try:
        jieba.lcut("预热jieba", cut_all=False) # 尝试初始化jieba
        print("Jieba initialized.")
        src_tokenizer_fn = tokenize_zh
    except Exception as e:
        print(f"Error initializing jieba: {e}. Jieba is required for Chinese tokenization.")
        print("Please ensure jieba is installed and working.")
        exit()
    trg_tokenizer_fn = tokenize_en

    # --- 重建词汇表 ---
    print("Rebuilding vocabularies from training data...")
    # 加载原始训练数据以构建词汇表
    # 注意：这里只加载用于构建词汇表的数据，不需要全量加载进行训练
    # 如果你的训练数据非常大，可以考虑保存和加载 Vocab 对象本身，而不是每次重建
    train_data_zh_for_vocab = load_sentences_from_file(train_zh_filepath)
    train_data_en_for_vocab = load_sentences_from_file(train_en_filepath)

    if not train_data_zh_for_vocab or not train_data_en_for_vocab:
        print("Could not load training data for vocabulary building. Exiting.")
        exit()

    src_vocab = Vocab(freq_threshold=FREQ_THRESHOLD)
    trg_vocab = Vocab(freq_threshold=FREQ_THRESHOLD)
    src_vocab.build_vocab(train_data_zh_for_vocab, src_tokenizer_fn)
    trg_vocab.build_vocab(train_data_en_for_vocab, trg_tokenizer_fn)

    print(f"Source Vocab Size: {len(src_vocab)}")
    print(f"Target Vocab Size: {len(trg_vocab)}")

    SRC_PAD_IDX = src_vocab.stoi["<PAD>"]
    TRG_PAD_IDX = trg_vocab.stoi["<PAD>"]
    INPUT_DIM = len(src_vocab)
    OUTPUT_DIM = len(trg_vocab)

    if INPUT_DIM <= 4 or OUTPUT_DIM <= 4: # 检查词汇表大小是否合理
        print("Warning: Vocabulary size is very small. This might lead to poor translation.")
        print("Ensure FREQ_THRESHOLD and data paths for vocab building are correct.")

    # --- 初始化模型 ---
    print("Initializing model structure...")
    encoder = Encoder(INPUT_DIM, D_MODEL, NUM_ENCODER_LAYERS, NUM_HEADS, D_FF, DROPOUT, MAX_LEN, device)
    decoder = Decoder(OUTPUT_DIM, D_MODEL, NUM_DECODER_LAYERS, NUM_HEADS, D_FF, DROPOUT, MAX_LEN, device)
    model = Transformer(encoder, decoder, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)

    # --- 加载模型权重 ---
    if not os.path.exists(model_load_path):
        print(f"Model file not found at '{model_load_path}'. Exiting.")
        exit()
    try:
        print(f"Loading model weights from '{model_load_path}'...")
        model.load_state_dict(torch.load(model_load_path, map_location=device))
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        exit()

    model.eval() # 设置为评估模式

    # --- 进行翻译 ---
    print("\n--- Transformer Translation Service ---")
    print("Type 'quit' or 'exit' to stop.")
    while True:
        try:
            input_sentence = input("Enter Chinese sentence to translate: ")
            if input_sentence.lower() in ['quit', 'exit']:
                break
            if not input_sentence.strip():
                continue

            translated_tokens, attention = translate_sentence(
                input_sentence,
                src_vocab,
                trg_vocab,
                model,
                src_tokenizer_fn,
                device,
                max_output_len=MAX_LEN
            )
            print(f"Input (ZH): {input_sentence}")
            print(f"Translated (EN): {' '.join(translated_tokens)}")
            print("-" * 20)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"An error occurred during translation: {e}")

    print("Translation service stopped.")