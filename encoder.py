from datasets import load_dataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import math
import copy

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            attn_scores = attn_scores + mask  # mask has -inf for masked positions
            
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class EncoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(EncoderOnlyTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])

        self.fc = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_causal_mask(self, seq):
        # Create causal mask to prevent looking ahead
        sz = seq.size(1)
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        mask = mask.to(seq.device)
        
        # Combine with padding mask
        padding_mask = (seq == 0).unsqueeze(1).unsqueeze(2)
        combined_mask = mask | padding_mask
        attention_mask = torch.zeros_like(combined_mask, dtype=torch.float)
        attention_mask.masked_fill_(combined_mask, float('-inf'))
        
        return attention_mask

    def forward(self, src):
        mask = self.generate_causal_mask(src)
        src_embedded = self.dropout(self.positional_encoding(self.embedding(src)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, mask)

        output = self.fc(enc_output)
        return output

class PTBDataset(data.Dataset):
    def __init__(self, sentences, vocab, max_len):
        self.sentences = sentences
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        tokens = ['<s>'] + self.sentences[idx]['sentence'].split() + ['</s>']
        tokens = tokens[:self.max_len]

        # Convert to indices
        indices = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]

        # Pad sequence
        if len(indices) < self.max_len:
            indices = indices + [self.vocab['<pad>']] * (self.max_len - len(indices))

        return torch.tensor(indices[:-1]), torch.tensor(indices[1:])

def build_vocab(sentences, min_freq=2):
    counter = {}
    for sentence in sentences:
        for token in sentence['sentence'].split():
            counter[token] = counter.get(token, 0) + 1

    vocab = {'<pad>': 0, '<unk>': 1, '<s>': 2, '</s>': 3}
    for token, freq in counter.items():
        if freq >= min_freq:
            vocab[token] = len(vocab)

    return vocab

def compute_perplexity(probs, targets):
    """Compute perplexity: exp(-1/T * sum(log(p(ti|t<i))))"""
    log_probs = torch.log(probs)
    # Exclude padding tokens from loss calculation
    non_pad_mask = (targets != 0).view(-1)
    predictions = log_probs.view(-1, log_probs.size(-1))[non_pad_mask]
    targets = targets.view(-1)[non_pad_mask]
    
    if len(targets) == 0:
        return float('inf')  # Return infinity for empty sequences
        
    loss = F.nll_loss(predictions, targets, reduction='mean')
    return math.exp(loss.item())

def train():
    # Hyperparameters
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_length = 50
    dropout = 0.1
    batch_size = 32
    num_epochs = 5
    learning_rate = 0.0001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    ptb = load_dataset('ptb-text-only/ptb_text_only')
    vocab = build_vocab(ptb['train'])
    vocab_size = len(vocab)

    # Create datasets
    train_dataset = PTBDataset(ptb['train'], vocab, max_seq_length)
    val_dataset = PTBDataset(ptb['validation'], vocab, max_seq_length)
    test_dataset = PTBDataset(ptb['test'], vocab, max_seq_length)
    print("Train Data: ", len(train_dataset))
    print("Val Data: ", len(val_dataset))
    print("Test Data: ", len(test_dataset))

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size)
    print("Train length: ", len(train_loader))
    print("Val length: ", len(val_loader))
    print("Test length: ", len(test_loader))

    # Initialize model
    model = EncoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_length=max_seq_length,
        dropout=dropout
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])

    # Training loop
    best_val_ppl = float('inf')
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)

            optimizer.zero_grad()
            output = model(src)  # Only need source sequence for encoder-only
            loss = criterion(output.view(-1, vocab_size), tgt.view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Add gradient clipping
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                output = model(src)
                val_loss += criterion(output.view(-1, vocab_size), tgt.view(-1)).item()

        val_ppl = math.exp(val_loss / len(val_loader))
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, Val PPL: {val_ppl:.4f}')

    # Generate predictions file
    predictions = []
    model.eval()
    with torch.no_grad():
        for i, (src, tgt) in enumerate(test_dataset):
            src = src.unsqueeze(0).to(device)
            tgt = tgt.unsqueeze(0).to(device)
            output = model(src)
            probs = F.softmax(output, dim=-1)
            ppl = compute_perplexity(probs, tgt)
            predictions.append({'ID': i, 'ppl': ppl})

            if i % 100 == 0:
                print(f"Processed {i}/{len(test_dataset)} test samples")

    # Save predictions
    import pandas as pd
    df = pd.DataFrame(predictions)
    print(f"Generated {len(df)} predictions")
    df.to_csv('predictions.csv', index=False)

if __name__ == "__main__":
    train()