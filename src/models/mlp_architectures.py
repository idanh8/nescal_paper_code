import torch.nn as nn
import torch.nn.functional as F


def _package(logits, stage_maps, pooled_final, return_embeddings, embedding_layer):
    if embedding_layer is not None:
        if embedding_layer == 'final':
            return logits, pooled_final
        return logits, stage_maps[embedding_layer]
    if not return_embeddings:
        return logits
    emb = dict(stage_maps); emb['final'] = pooled_final
    return logits, emb


class MLP(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden_dims=(64, 64), embed_dim: int = 16,
                 input_channels: int = None):
        super().__init__()

        assert len(hidden_dims) >= 2, "MLP expects >=2 hidden layers so stage_minus_1/2 exist"
        self.in_dim = in_dim
        h1, h2 = hidden_dims[-2], hidden_dims[-1]
        self.fc1 = nn.Linear(in_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.embed = nn.Linear(h2, embed_dim)
        self.linear = nn.Linear(embed_dim, num_classes)
        self.feat_dim = embed_dim

    def forward(self, x, return_embeddings=False, embedding_layer=None):
        if x.dim() > 2:
            x = x.reshape(x.size(0), -1)
        a1 = F.relu(self.fc1(x))
        a2 = F.relu(self.fc2(a1))
        z = F.relu(self.embed(a2))
        logits = self.linear(z)
        stage_maps = {'stage_minus_1': a2, 'stage_minus_2': a1}
        return _package(logits, stage_maps, z, return_embeddings, embedding_layer)


def _activation(name):
    return {'relu': nn.ReLU(), 'gelu': nn.GELU(), 'silu': nn.SiLU(),
            'leakyrelu': nn.LeakyReLU(0.01)}[name.lower()]


class FrozenEncoderMLP(nn.Module):
    def __init__(self, in_dim, num_classes, hidden_dims=(256,), dropout: float = 0.0,
                 norm_type=None, activation: str = 'relu', input_channels=None):
        super().__init__()
        if in_dim is None:
            raise ValueError("FrozenEncoderMLP requires in_dim (encoder embedding dimension)")
        self.in_dim = in_dim
        dims = [in_dim] + list(hidden_dims)
        self.blocks = nn.ModuleList()
        for i in range(len(dims) - 1):
            layer = [nn.Linear(dims[i], dims[i + 1])]
            if norm_type == 'layer': layer.append(nn.LayerNorm(dims[i + 1]))
            elif norm_type == 'batch': layer.append(nn.BatchNorm1d(dims[i + 1]))
            layer.append(_activation(activation))
            if dropout > 0: layer.append(nn.Dropout(dropout))
            self.blocks.append(nn.Sequential(*layer))
        self.linear = nn.Linear(dims[-1], num_classes)
        self.feat_dim = in_dim

    def forward(self, x, return_embeddings=False, embedding_layer=None):
        if x.dim() > 2:
            x = x.reshape(x.size(0), -1)
        h = x
        for blk in self.blocks:
            h = blk(h)
        logits = self.linear(h)

        if embedding_layer is not None:
            return logits, x
        if not return_embeddings:
            return logits
        return logits, {'final': x, 'input': x, 'stage_minus_1': x, 'stage_minus_2': x}

class TabularMLP(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden_dims=(256, 256, 256),
                 dropout: float = 0.1, input_channels: int = None):
        super().__init__()
        assert len(hidden_dims) >= 3, 'TabularMLP expects >=3 hidden layers so stage_minus_1/2 exist'
        self.in_dim = in_dim
        dims = [in_dim] + list(hidden_dims)
        self.blocks = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(hidden_dims))])
        self.dropout = nn.Dropout(dropout)
        self.feat_dim = hidden_dims[-1]
        self.linear = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x, return_embeddings=False, embedding_layer=None):
        acts = []
        h = x
        for blk in self.blocks:
            h = self.dropout(F.relu(blk(h)))
            acts.append(h)
        pooled_final = acts[-1]
        stage_maps = {'stage_minus_1': acts[-2], 'stage_minus_2': acts[-3]}
        return _package(self.linear(pooled_final), stage_maps, pooled_final,
                        return_embeddings, embedding_layer)


def create_mlp_model(arch: str, num_classes: int, in_dim: int = None, input_channels: int = 3,
                     hidden_dims=(64, 64), embed_dim: int = 16, **kwargs):
    arch = arch.lower()
    if arch == 'mlp':
        if in_dim is None:
            raise ValueError("arch='mlp' requires model_params['in_dim'] (blob feature dimension)")
        return MLP(in_dim=in_dim, num_classes=num_classes, hidden_dims=hidden_dims, embed_dim=embed_dim)
    if arch == 'frozen_enc_mlp':


        return FrozenEncoderMLP(in_dim=in_dim, num_classes=num_classes, hidden_dims=hidden_dims,
                                dropout=kwargs.get('dropout', 0.0), norm_type=kwargs.get('norm_type'),
                                activation=kwargs.get('activation', 'relu'))

    if arch == 'tabular_mlp':
        if in_dim is None:
            raise ValueError("arch='tabular_mlp' requires model_params['in_dim']")
        return TabularMLP(in_dim=in_dim, num_classes=num_classes,
                          hidden_dims=kwargs.get('hidden_dims', (256, 256, 256)),
                          dropout=kwargs.get('dropout', 0.1))

    raise ValueError(f"create_mlp_model got unexpected arch '{arch}'")
