import torch
from attention import MultiheadSelfAttention_reduced


class NeuralNetwork(torch.nn.Module):
    def __init__(self, textLength, vocabSize, wordEmbedDim=64, embedDim=None, noHeads=1,
                 hiddenDim=0, outputDim=0, dropout=0.3, padIdx=0, pretrainedEmbeddings=None):
        super().__init__()

        if embedDim is None:
            embedDim = wordEmbedDim

        self.padIdx = padIdx

        self.embedding = torch.nn.Embedding(vocabSize, wordEmbedDim, padding_idx=padIdx)

        if pretrainedEmbeddings is not None:
            with torch.no_grad():
                self.embedding.weight.copy_(pretrainedEmbeddings)

        self.attention = MultiheadSelfAttention_reduced(wordEmbedDim, embedDim, noHeads)

        self.flatten = torch.nn.Flatten(start_dim=1)

        # Dropout to reduce overfitting
        self.dropout = torch.nn.Dropout(dropout)

        self.linear = torch.nn.Linear(textLength * embedDim, hiddenDim)

        self.sigmoid = torch.nn.Sigmoid()

        self.output = torch.nn.Linear(hiddenDim, outputDim)

    def forward(self, x):
        pad_mask = x == self.padIdx
        x = self.embedding(x)
        x, attention = self.attention(x, key_padding_mask=pad_mask)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.linear(x)
        x = self.sigmoid(x)
        x = self.dropout(x)
        x = self.output(x)

        return x
