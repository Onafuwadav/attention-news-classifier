import math
import torch


class SelfAttention(torch.nn.Module):
    # Single head
    def __init__(self, embedDim):
        super().__init__()

        self.query = torch.nn.Linear(embedDim, embedDim)
        self.key = torch.nn.Linear(embedDim, embedDim)
        self.value = torch.nn.Linear(embedDim, embedDim)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attn_logits = torch.matmul(Q, K.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(Q.size(-1))

        att = torch.softmax(attn_logits, dim=-1)
        h = torch.matmul(att, V)

        return h, att


class SelfAttention_reduced(torch.nn.Module):
    # Bottleneck self attention
    def __init__(self, inputDim, embedDim):
        super().__init__()

        self.query = torch.nn.Linear(inputDim, embedDim)
        self.key = torch.nn.Linear(inputDim, embedDim)
        self.value = torch.nn.Linear(inputDim, embedDim)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attn_logits = torch.matmul(Q, K.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(Q.size(-1))

        att = torch.softmax(attn_logits, dim=-1)
        h = torch.matmul(att, V)

        return h, att


class MultiheadSelfAttention_reduced(torch.nn.Module):
    # Multi head self attention
    def __init__(self, inputDim, embedDim, noHeads):
        super().__init__()

        # embedDim must split evenly
        assert embedDim % noHeads == 0

        self.query = torch.nn.Linear(inputDim, embedDim)
        self.key = torch.nn.Linear(inputDim, embedDim)
        self.value = torch.nn.Linear(inputDim, embedDim)

        self.noHeads = noHeads
        self.embedDim = embedDim
        self.head = self.embedDim // self.noHeads


    # Padding mask so padded positions get no attention
    def forward(self, x, key_padding_mask=None):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        heads = []
        attentions = []

        for i in range(self.noHeads):
            start = self.head * i
            end = self.head * (i + 1)

            Q_head = Q[:, :, start:end]
            K_head = K[:, :, start:end]
            V_head = V[:, :, start:end]

            attn_logits = torch.matmul(Q_head, K_head.transpose(-2, -1))
            attn_logits = attn_logits / math.sqrt(Q_head.size(-1))

            # Block attention to padded keys
            if key_padding_mask is not None:
                attn_logits = attn_logits.masked_fill(key_padding_mask[:, None, :], float('-inf'))

            att = torch.softmax(attn_logits, dim=-1)
            heads.append(torch.matmul(att, V_head))
            attentions.append(att)

        h = torch.cat(heads, dim=-1)
        att = torch.stack(attentions, dim=1)

        return h, att
