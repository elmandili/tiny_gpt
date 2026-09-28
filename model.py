import torch
import torch.nn as nn
import torch.nn.functional as F

from config import BLOCK_SIZE, N_EMBD, N_HEAD, N_LAYER, DROPOUT


class AttentionHead(nn.Module):

    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
        )

        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):

        B, T, C = x.shape

        # Produce keys, queries and values
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # Q K^T
        attention = q @ k.transpose(-2, -1)

        # Scale
        attention = attention * (k.shape[-1] ** -0.5)

        # Causal mask
        attention = attention.masked_fill(
            self.mask[:T, :T] == 0,
            float("-inf")
        )

        # Convert scores into probabilities
        attention = F.softmax(attention, dim=-1)

        attention = self.dropout(attention)

        # Weighted combination of values
        output = attention @ v

        return output



class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()

        self.heads = nn.ModuleList([
            AttentionHead(head_size)
            for _ in range(num_heads)
        ])

        self.projection = nn.Linear(
            num_heads * head_size,
            N_EMBD
        )

        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):

        outputs = [
            head(x)
            for head in self.heads
        ]

        output = torch.cat(outputs, dim=-1)

        output = self.projection(output)

        return self.dropout(output)


class FeedForward(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(N_EMBD, 4 * N_EMBD),

            nn.GELU(),

            nn.Linear(4 * N_EMBD, N_EMBD),

            nn.Dropout(DROPOUT)
        )

    def forward(self, x):
        return self.network(x)

class TransformerBlock(nn.Module):

    def __init__(self):
        super().__init__()

        head_size = N_EMBD // N_HEAD

        self.attention = MultiHeadAttention(
            N_HEAD,
            head_size
        )

        self.feed_forward = FeedForward()

        self.norm1 = nn.LayerNorm(N_EMBD)
        self.norm2 = nn.LayerNorm(N_EMBD)

    def forward(self, x):

        # Pre-LayerNorm + residual connection
        x = x + self.attention(
            self.norm1(x)
        )

        x = x + self.feed_forward(
            self.norm2(x)
        )

        return x


class TinyGPT(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            N_EMBD
        )

        self.position_embedding = nn.Embedding(
            BLOCK_SIZE,
            N_EMBD
        )

        self.blocks = nn.Sequential(
            *[
                TransformerBlock()
                for _ in range(N_LAYER)
            ]
        )

        self.final_norm = nn.LayerNorm(N_EMBD)

        self.language_head = nn.Linear(
            N_EMBD,
            vocab_size
        )

    def forward(self, tokens, targets=None):

        B, T = tokens.shape

        # Token embeddings
        token_emb = self.token_embedding(tokens)

        # Position embeddings
        positions = torch.arange(
            T,
            device=tokens.device
        )

        position_emb = self.position_embedding(
            positions
        )

        x = token_emb + position_emb

        # Transformer
        x = self.blocks(x)

        x = self.final_norm(x)

        # Predict vocabulary
        logits = self.language_head(x)

        loss = None

        if targets is not None:

            B, T, V = logits.shape

            logits_flat = logits.view(B * T, V)
            targets_flat = targets.view(B * T)

            loss = F.cross_entropy(
                logits_flat,
                targets_flat
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        tokens,
        max_new_tokens,
        tokenizer,
        temperature=0.8
    ):
        for _ in range(max_new_tokens):

            # Only use the latest context
            context = tokens[:, -BLOCK_SIZE:]

            logits, _ = self(context)

            # Prediction for the next character
            logits = logits[:, -1, :]

            # Temperature
            logits = logits / temperature

            probabilities = F.softmax(
                logits,
                dim=-1
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            tokens = torch.cat(
                [tokens, next_token],
                dim=1
            )

            # Decode generated text
            generated_text = tokenizer.decode(
                tokens[0].tolist()
            )

            # Stop when model generates <END>
            if generated_text.endswith("<END>"):
                break

        return tokens