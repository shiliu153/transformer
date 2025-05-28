import torch
import torch.nn as nn
import math
import os
import jieba



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 词汇表类 (与 transformer.py 中的定义相同) ---
class Vocab:
    def __init__(self, freq_threshold=2): # freq_threshold 在加载时无关紧要
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        # build_vocab 和 __len__ 在加载预训练词汇表时不需要完全一样
        # 但为了torch.load能正确反序列化，类结构应匹配

    def __len__(self):
        return len(self.itos)

    def numericalize(self, text, tokenizer):
        tokenized_text = tokenizer(text)
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokenized_text]

# --- 分词函数 (与 transformer.py 中的定义相同) ---
def tokenize_zh(text):
    return jieba.lcut(text)

def tokenize_en(text):
    return text.lower().split()

# --- 模型组件 (与 transformer.py 中的定义相同) ---
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
        x = x * math.sqrt(self.d_model) # 应用缩放
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
        self.pos_embedding = PositionalEncoding(d_model, dropout, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout)
                                     for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        src_emb = self.tok_embedding(src)
        src_emb = self.pos_embedding(src_emb) # 应用位置编码
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
        trg_emb = self.pos_embedding(trg_emb) # 应用位置编码
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

# --- 翻译函数 (与 transformer.py 中的定义相同) ---
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
    if trg_tokens and trg_tokens[0] == "<SOS>":
        trg_tokens = trg_tokens[1:]
    if trg_tokens and trg_tokens[-1] == "<EOS>":
        trg_tokens = trg_tokens[:-1]
    return trg_tokens, attention


if __name__ == '__main__':

    D_MODEL = 256
    NUM_HEADS = 8
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3
    D_FF = 512
    DROPOUT = 0.1
    MAX_LEN = 100

    # --- 文件路径 ---
    model_save_dir = './saved_models'
    src_vocab_path = os.path.join(model_save_dir, 'src_vocab.pth')
    trg_vocab_path = os.path.join(model_save_dir, 'trg_vocab.pth')
    model_path = os.path.join(model_save_dir, 'transformer-nmt-model-best.pt')

    print(f"Using device: {device}")

    # --- 加载模型和词汇表 ---
    print(f"Loading source vocabulary from {src_vocab_path}...")
    if not os.path.exists(src_vocab_path):
        print(f"Error: Source vocabulary file not found at {src_vocab_path}")
        exit()

    src_vocab = torch.load(src_vocab_path, map_location=device, weights_only=False)
    print(f"Source vocabulary loaded. Size: {len(src_vocab)}")

    print(f"Loading target vocabulary from {trg_vocab_path}...")
    if not os.path.exists(trg_vocab_path):
        print(f"Error: Target vocabulary file not found at {trg_vocab_path}")
        exit()

    trg_vocab = torch.load(trg_vocab_path, map_location=device, weights_only=False)
    print(f"Target vocabulary loaded. Size: {len(trg_vocab)}")

    INPUT_DIM = len(src_vocab)
    OUTPUT_DIM = len(trg_vocab)
    SRC_PAD_IDX = src_vocab.stoi["<PAD>"]
    TRG_PAD_IDX = trg_vocab.stoi["<PAD>"]

    min_vocab_size = 4
    if INPUT_DIM <= min_vocab_size or OUTPUT_DIM <= min_vocab_size:
        print(f"Error: Loaded vocabulary size is too small (SRC: {INPUT_DIM}, TRG: {OUTPUT_DIM}). Cannot proceed.")
        exit()

    print("Initializing model structure...")
    enc = Encoder(INPUT_DIM, D_MODEL, NUM_ENCODER_LAYERS, NUM_HEADS, D_FF, DROPOUT, MAX_LEN, device)
    dec = Decoder(OUTPUT_DIM, D_MODEL, NUM_DECODER_LAYERS, NUM_HEADS, D_FF, DROPOUT, MAX_LEN, device)
    model = Transformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)


    print(f"Loading model weights from {model_path}...")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        exit()
    try:

        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        print("Ensure that the model architecture in this script matches the saved model.")
        exit()

    model.eval() # 设置为评估模式

    src_tokenizer_fn_for_translation = None
    try:
        jieba.lcut("预热jieba", cut_all=False) # 确保jieba可用
        src_tokenizer_fn_for_translation = tokenize_zh
        print("Using Jieba for Chinese tokenization.")
    except NameError:
        print("jieba not available. Ensure it's installed and imported if translating from Chinese.")
        print("If your source language is not Chinese or uses a different tokenizer, adjust src_tokenizer_fn_for_translation.")
        exit()
    except Exception as e:
        print(f"Error initializing jieba: {e}. Cannot proceed with Chinese tokenization.")
        exit()

    test_sentences_zh = [
        "你好 世界",
        "我 想 看 看 这个 产品",
        "这 是 一个 测试"
    ]

    print("\n--- Starting Translation ---")
    for sentence_str in test_sentences_zh:
        print(f"\nOriginal (ZH): '{sentence_str}'")
        translated_tokens, attention = translate_sentence(
            sentence_str,
            src_vocab,
            trg_vocab,
            model,
            src_tokenizer_fn_for_translation,
            device,
            MAX_LEN
        )
        print(f"Translated (EN): {' '.join(translated_tokens)}")

    print("\n--- Translation Finished ---")

    while True:
        input_sentence = input("\nEnter a Chinese sentence to translate (or 'quit' to exit): ")
        if input_sentence.lower() == 'quit':
            break
        if not input_sentence.strip():
            continue
        translated_tokens, _ = translate_sentence(
            input_sentence,
            src_vocab,
            trg_vocab,
            model,
            src_tokenizer_fn_for_translation,
            device,
            MAX_LEN
        )
        print(f"Translated (EN): {' '.join(translated_tokens)}")