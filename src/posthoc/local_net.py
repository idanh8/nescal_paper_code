"""
LoCal Nets (Barbera, Perini, De Toni, Passerini, Pugnana; AISTATS 2026; arXiv 2510.26566). 
Auxiliary head + losses, reimplemented at the authors' published CIFAR-10/100 configuration.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoCalNet(nn.Module):

    def __init__(self, feature_dim, num_classes, similarity_dim=50, hidden_dim=64,
                 dropout=0.3, init_alpha_sim=1.0, init_alpha_cls=1.0):
        super().__init__()
        self.similarity_dim = similarity_dim
        self.num_classes = num_classes

        self.dense1 = nn.Linear(feature_dim, hidden_dim)
        self.dropout1 = nn.Dropout(p=dropout)
        self.similarity_head = nn.Linear(hidden_dim, similarity_dim)
        self.classification_head = nn.Linear(hidden_dim, num_classes)

        nn.init.normal_(self.dense1.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.similarity_head.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.classification_head.weight, mean=0.0, std=0.01)

        # Authors use scalar alpha/beta for <=10 classes and per-dimension

        if num_classes <= 10:
            self.alpha_sim = nn.Parameter(torch.tensor(float(init_alpha_sim)))
            self.beta_sim = nn.Parameter(torch.empty(1).normal_(mean=0.0, std=0.01))
            self.alpha_cls = nn.Parameter(torch.tensor(float(init_alpha_cls)))
            self.beta_cls = nn.Parameter(torch.empty(1).normal_(mean=0.0, std=0.01))
        else:
            self.alpha_sim = nn.Parameter(torch.full((similarity_dim,), float(init_alpha_sim)))
            self.beta_sim = nn.Parameter(torch.empty(similarity_dim).normal_(mean=0.0, std=0.01))
            self.alpha_cls = nn.Parameter(torch.full((num_classes,), float(init_alpha_cls)))
            self.beta_cls = nn.Parameter(torch.empty(num_classes).normal_(mean=0.0, std=0.01))

    def forward(self, feats, logits, pca):
        h = self.dropout1(F.relu(self.dense1(feats)))
        phi = self.similarity_head(h) + self.alpha_sim * pca + self.beta_sim
        new_logits = self.classification_head(h) + self.alpha_cls * logits + self.beta_cls
        return new_logits, phi


def neighborhood_frequencies(phi, y_onehot, sigma, eps=1e-6, exclude_self=True):
    sq = (phi ** 2).sum(dim=1)
    d2 = torch.clamp(sq.unsqueeze(1) + sq.unsqueeze(0) - 2.0 * (phi @ phi.t()), min=0.0)
    w = torch.exp(-d2 / (2.0 * sigma + eps))
    if exclude_self:
        w = w * (1.0 - torch.eye(phi.size(0), device=phi.device, dtype=phi.dtype))
    w = w / (w.sum(dim=1, keepdim=True) + eps)
    w = torch.clamp(w, eps, 1.0 - eps)
    return w @ y_onehot


def js_distance(p, q, eps=1e-4):
    p = torch.clamp(p, eps, 1.0 - eps)
    q = torch.clamp(q, eps, 1.0 - eps)
    m = torch.clamp(0.5 * (p + q), eps, 1.0 - eps)
    kl_p = p * torch.log(p / m) + (1.0 - p) * torch.log((1.0 - p) / (1.0 - m))
    kl_q = q * torch.log(q / m) + (1.0 - q) * torch.log((1.0 - q) / (1.0 - m))
    js = torch.clamp(0.5 * (kl_p + kl_q), min=eps)
    return torch.sqrt(js).mean()


def categorical_cross_entropy(probs, targets, eps=1e-8):
    probs = torch.clamp(probs, eps, 1.0 - eps)
    return -torch.sum(targets * torch.log(probs), dim=1).mean()


def locals_loss(new_logits, phi, y_onehot, sigma, lambda_js=1.0, alpha_ce=1.0):
    p2 = F.softmax(new_logits, dim=1)
    p1 = neighborhood_frequencies(phi, y_onehot, sigma)
    js = js_distance(p2, p1)
    ce = categorical_cross_entropy(p1, y_onehot)
    return lambda_js * js + alpha_ce * ce, js, ce
