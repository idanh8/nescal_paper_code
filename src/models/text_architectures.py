import torch
import torch.nn as nn
import torch.nn.functional as F

from models.factory import _package_generic

class GlobalPoolingCNN(nn.Module):
    def __init__(self, num_classes, vocab_size, embed_dim=100, num_filters=128, kernel_size=5,
                 dropout=0.3, embedding_matrix_path=None, pad_idx=0, freeze_embedding=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        if embedding_matrix_path is not None:
            mat = torch.load(embedding_matrix_path, map_location='cpu')
            assert tuple(mat.shape) == (vocab_size, embed_dim), \
                f'GloVe matrix {tuple(mat.shape)} != ({vocab_size}, {embed_dim})'
            self.embedding.weight.data.copy_(mat)
        if freeze_embedding:
            self.embedding.weight.requires_grad = False
        self.conv1 = nn.Conv1d(embed_dim, num_filters, kernel_size)
        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size)
        self.conv3 = nn.Conv1d(num_filters, num_filters, kernel_size)
        self.pool = nn.MaxPool1d(kernel_size, stride=1)
        self.fc1 = nn.Linear(num_filters, num_filters)
        self.feat_dim = num_filters
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(num_filters, num_classes)

    def forward(self, x, return_embeddings=False, embedding_layer=None):
        e = self.embedding(x).transpose(1, 2)
        h = self.pool(F.relu(self.conv1(e)))
        h = self.pool(F.relu(self.conv2(h)))
        h = F.relu(self.conv3(h))
        g = h.max(dim=2).values
        pooled_final = F.relu(self.fc1(g))
        logits = self.linear(self.dropout(pooled_final))
        return _package_generic(logits, {}, pooled_final, return_embeddings, embedding_layer)


def create_text_model(arch, num_classes, **kwargs):
    arch = arch.lower()
    if arch == 'gpcnn':
        keys = ('vocab_size', 'embed_dim', 'num_filters', 'kernel_size', 'dropout',
                'embedding_matrix_path', 'pad_idx', 'freeze_embedding')
        return GlobalPoolingCNN(num_classes=num_classes, **{k: kwargs[k] for k in keys if k in kwargs})
    raise ValueError(f"Unrecognized text arch: '{arch}'")
