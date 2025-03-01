import streamlit as st
import torch
import pickle
import re
import numpy as np
import torch.nn as nn

# Special tokens
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'

# Tokenization function
def tokenize(text):
    return re.findall(r'\b\w+\b|[^\w\s]', text)

# Convert text to indices
def text_to_indices(text, vocab):
    tokens = tokenize(text)
    indices = [vocab.get(token, vocab[UNK_TOKEN]) for token in tokens]
    return [vocab[SOS_TOKEN]] + indices + [vocab[EOS_TOKEN]]

# Positional Encoding (paste the same class definition from above)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

# Multi-Head Attention (paste from above)
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        query = self.query_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention, value).transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        return self.output_linear(output)

# Feed-Forward Network (paste from above)
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x = self.dropout(torch.relu(self.linear1(x)))
        return self.linear2(x)

# Encoder Layer (paste from above)
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        attention_output = self.self_attention(x, x, x, mask)
        x = self.layer_norm1(x + self.dropout(attention_output))
        ff_output = self.feed_forward(x)
        return self.layer_norm2(x + self.dropout(ff_output))

# Decoder Layer (paste from above)
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask, tgt_mask):
        self_attention_output = self.self_attention(x, x, x, tgt_mask)
        x = self.layer_norm1(x + self.dropout(self_attention_output))
        cross_attention_output = self.cross_attention(x, enc_output, enc_output, src_mask)
        x = self.layer_norm2(x + self.dropout(cross_attention_output))
        ff_output = self.feed_forward(x)
        return self.layer_norm3(x + self.dropout(ff_output))

# Encoder (paste from above)
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x

# Decoder (paste from above)
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask, tgt_mask):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return x

# Transformer Model (paste from above)
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout)
        self.output_linear = nn.Linear(d_model, tgt_vocab_size)
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        return self.output_linear(dec_output)

# Causal mask function
def create_causal_mask(seq_len):
    return torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)

# Load vocabularies and model
with open('pseudocode_vocab.pkl', 'rb') as f:
    pseudocode_vocab = pickle.load(f)
with open('cpp_vocab.pkl', 'rb') as f:
    cpp_vocab = pickle.load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Transformer(len(pseudocode_vocab), len(cpp_vocab), d_model=256, num_heads=8, num_layers=3, d_ff=512, max_len=100, dropout=0.1)
model.load_state_dict(torch.load('transformer_model.pth'))
model.to(device)
model.eval()

# Inference function
def generate_code(model, pseudocode, pseudocode_vocab, cpp_vocab, max_len=100):
    with torch.no_grad():
        pseudo_indices = text_to_indices(pseudocode, pseudocode_vocab)
        src = torch.tensor([pseudo_indices]).to(device)
        src_mask = (src != pseudocode_vocab[PAD_TOKEN]).unsqueeze(1).unsqueeze(2).to(device)
        tgt = torch.tensor([[cpp_vocab[SOS_TOKEN]]]).to(device)
        generated = []
        for _ in range(max_len):
            tgt_seq_len = tgt.size(1)
            causal_mask = create_causal_mask(tgt_seq_len).to(device)
            output = model(src, tgt, src_mask, causal_mask)
            next_token = output[:, -1, :].argmax(dim=-1).item()
            generated.append(next_token)
            if next_token == cpp_vocab[EOS_TOKEN]:
                break
            tgt = torch.cat([tgt, torch.tensor([[next_token]]).to(device)], dim=1)
        generated_tokens = [list(cpp_vocab.keys())[list(cpp_vocab.values()).index(idx)] for idx in generated if idx != cpp_vocab[EOS_TOKEN]]
        return ' '.join(generated_tokens)

# Streamlit UI
st.title('Pseudocode to C++ Converter')
pseudocode_input = st.text_area('Enter pseudocode:', height=200)

if st.button('Generate C++ Code'):
    if pseudocode_input:
        generated_code = generate_code(model, pseudocode_input, pseudocode_vocab, cpp_vocab)
        st.code(generated_code, language='cpp')
    else:
        st.write('Please enter some pseudocode.')
