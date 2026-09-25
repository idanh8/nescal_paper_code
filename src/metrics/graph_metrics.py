import numpy as np
import torch
import torch.nn.functional as F

@torch.no_grad()
def _embed_probs_labels(model, loader, device, layer):
    model.eval(); Z, P, Y = [], [], []
    for images, labels, idx in loader:
        logits, emb = model(images.to(device), return_embeddings=True)
        Z.append(emb[layer].float().cpu()); P.append(torch.softmax(logits, 1).float().cpu()); Y.append(labels)
    return torch.cat(Z).numpy(), torch.cat(P).numpy(), torch.cat(Y).numpy()

def _cwcc(Z, P, k):
    Zn = F.normalize(Z.float(), dim=1); S = Zn @ Zn.T; S.fill_diagonal_(-1e30)
    idx = S.topk(k, 1).indices; N = Z.shape[0]
    i = torch.arange(N, device=Z.device).repeat_interleave(k); j = idx.reshape(-1)
    conf = P.max(1).values
    cross = (P[i].argmax(1) != P[j].argmax(1)).float(); w = conf[i] * conf[j]
    return float((w * cross).sum() / (w.sum() + 1e-12))

def _knn_jaccard(emb_now, emb_prev, k):
    if emb_prev is None: return float('nan')
    def nbrs(z):
        zt = F.normalize(torch.as_tensor(z, dtype=torch.float32), dim=1)
        S = zt @ zt.T; S.fill_diagonal_(-1e30)
        return S.topk(k, 1).indices.cpu().numpy()
    a, b = nbrs(emb_now), nbrs(emb_prev)
    return float(np.mean([len(set(a[i]) & set(b[i])) / len(set(a[i]) | set(b[i])) for i in range(len(a))]))
