from torch import nn, optim
from torch.utils.data import Dataset
import torch


class EM_DATA():
    def __init__(self):
        embedding_dim = 100
        vocab_size = 10
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        word_to_ix = {"hello": 0, "world": 1}
        embeds = nn.Embedding(2, 5)  # 2 words in vocab, 5 dimensional embeddings
        lookup_tensor = torch.tensor([word_to_ix["hello"]], dtype=torch.long)
        hello_embed = embeds(lookup_tensor)
        print(hello_embed)
