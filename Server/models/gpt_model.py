
import torch
from torch import nn
import torch.nn.functional as F

class Head(nn.Module):
    """ a self attention head """ 

    def __init__(self, head_size, n_embd, block_size, dropout=0.2):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # (B, T, Head_size)
        q = self.query(x) # (B, T, Head_size)
        # compute attention scores
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 #(B, T, Head_size)  @ (B, Head_size, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) #set all future values to -inf , (B, T, T) 
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        v = self.value(x) #(B, T, Head_size)
        out = wei @ v # apply the value matrix (B, T, T) -> (B, T, Head_size)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout = 0.2):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)]) #create a list of heads
        self.proj = nn.Linear(head_size * num_heads, n_embd) #head_size * num_heads dimension to become n_embd dimension
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) #concatenate these heads. (B, T, Head_size) -> (B, T, Head_size * num_heads)
        out = self.dropout(self.proj(out)) #linear transformation + (B, T, Head_size * num_heads) -> (B, T, n_embd) for correct out dimensions
        return out

class FeedFoward(nn.Module):

    def __init__(self, n_embd, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    #one decoder block

    def __init__(self, n_embd, n_head, block_size, dropout=0.2):
        #n_head -> amt of heads
        super().__init__()
        head_size = n_embd // n_head 
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
         #define LayerNorm functions. 2 because they won't just reduce mean to 0 and variance to 1, they will be trainable
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        #adds & layernorms per transformer architecture. Here layernorm is applied before the sublayer as is more common nowadays
        x = x + self.sa(self.ln1(x)) #after attention
        x = x + self.ffwd(self.ln2(x)) #after forward layer
        return x

    
# super simple bigram model
class GPTLanguageModel(nn.Module):

    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, dropout=0.2):
        super().__init__()
        # each token directly reads off the logits for the next token from lookup table
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        #positional encoding table, for each value in the block. does not necessarily have to have same dimensionality, n_embd
        #as the token embedding table, but does here
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head, block_size = block_size, dropout=dropout) for _ in range(n_layer)]) # * because nn.Sequential takes separate arguments, not an iterable as input
        self.ln_f = nn.LayerNorm(n_embd)
        #linear transformation layer, also changes n_embd features to vocab_size features, which we need since our
        #token embedding table is n_embd features tall, but logits must be vocab_size features tall
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T, C=n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T, C=n_embd)
        x = tok_emb + pos_emb # (B, T, C=n_embd)
        x = self.blocks(x) #run decoder blocks (B, T, C=n_embd)
        x = self.ln_f(x) # (B, T, C=n_embd)
        logits = self.lm_head(x) # (B, T, C=vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            #crop idx to the last set of block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits, loss = self(idx_cond) #targets = None, so no loss is calculated
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C), and works because no loss is calcualted, so logits dimensions are (B, T, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
