import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def js_divergence_torch(p: torch.Tensor, q: torch.Tensor, epsilon: float = 1e-10) -> torch.Tensor:
    """
    Batched Jensen-Shannon divergence between rows of p and q, in bits
    (log base 2) so the result is bounded in [0, 1] .
    Moved here from spatial_metrics.py (now deleted) since this was the only
    """
    p = torch.clamp(p, epsilon, 1.0)
    p = p / p.sum(dim=1, keepdim=True)
    q = torch.clamp(q, epsilon, 1.0)
    q = q / q.sum(dim=1, keepdim=True)
    m = 0.5 * (p + q)

    kl_pm = (p * (torch.log2(p) - torch.log2(m))).sum(dim=1)
    kl_qm = (q * (torch.log2(q) - torch.log2(m))).sum(dim=1)
    return 0.5 * kl_pm + 0.5 * kl_qm


class PreNeighborhoodConsistencyLoss(nn.Module):
    """
    Neighborhood Consistency regularization loss (secondary loss module).
    Penalizes inconsistency between predicted distributions of neighboring
    samples in feature space.
    Returns only the NCL reg term as a scalar. Primary loss and lambda
    scaling are handled by PrimarySecondaryLoss in losses.py.
    """
    requires_activation   = True
    requires_global_probs = True
    default_lambda_reg    = 1.0

    def __init__(self, lambda_reg: float = 1.0, tau: float = 1.0, divergence_type: str = 'js', weight_type: str = 'topk', k: int = 10, epsilon: float = 1e-10, mask_mode: str = 'none', graph_reducer=None, eps_fisher: float = 1e-2):
        """
        Args:
            lambda_reg:        regularization strength
            tau:               temperature for exponential neighbor weighting
            divergence_type:   'l2' or 'js'
            weight_type:       'exponential' or 'topk'
            k:                 number of neighbors (required)
            epsilon:           numerical stability constant
            eps_fisher:        numerical stability constant for Fisher information
        """
        super().__init__()

        self.lambda_reg = lambda_reg
        self.tau = tau
        self.divergence_type = divergence_type
        self.weight_type = weight_type
        self.k = k
        self.active = False
        self.mask_mode = mask_mode
        self.node_labels = None
        self.graph_reducer = graph_reducer
        self.eps_fisher = eps_fisher

    def activate(self, features, initial_probs, chunk_size=512, node_labels=None):
        self.precompute_matrices(features, chunk_size=chunk_size)
        self.update_global_probs(initial_probs)
        with torch.no_grad():
            nb = self.neighbors.to(initial_probs.device)
            pen_sum, js_sum, n_edges = 0.0, 0.0, 0
            for s in range(0, nb.shape[0], 4096):
                nb_c = nb[s:s + 4096]
                pj_flat = self.global_probs[nb_c].reshape(-1, self.global_probs.shape[1])
                pi_flat = self.global_probs[s:s + 4096].repeat_interleave(nb_c.shape[1], dim=0)
                pen_sum += float(self._edge_divergence(pi_flat, pj_flat).sum().item())
                js_sum += float(js_divergence_torch(pi_flat, pj_flat).sum().item())
                n_edges += pi_flat.shape[0]
            self.mean_edge_penalty_at_activation = pen_sum / n_edges
            self.mean_edge_js_at_activation = js_sum / n_edges
        if node_labels is not None:
            self.node_labels = torch.as_tensor(node_labels, dtype=torch.long)
        self.active = True

    def _reduce_graph_features(self, features):
        gr = self.graph_reducer
        if callable(gr):
            return gr(features)
        x = features.detach().cpu().numpy()
        if gr.startswith('pca'):
            from sklearn.decomposition import PCA
            k = int(gr[3:]) if gr[3:] else 50
            red = PCA(n_components=min(k, x.shape[1]), random_state=0).fit_transform(x)
        elif gr.startswith('umap'):
            import umap
            k = int(gr[4:]) if gr[4:] else 30
            red = umap.UMAP(n_components=k, random_state=0).fit_transform(x)
        else:
            raise ValueError(f'Unknown graph_reducer spec: {gr}')
        return torch.tensor(red, dtype=features.dtype, device=features.device)

    def precompute_matrices(self, features, chunk_size=512):
        assert self.k is not None, "k must be set for PreNCL — full matrix is infeasible at scale"

        if self.graph_reducer is not None:
            features = self._reduce_graph_features(features)

        N = features.shape[0]
        features_norm = F.normalize(features, p=2, dim=1)

        all_topk_indices = torch.zeros(N, self.k, dtype=torch.long)
        all_topk_weights = torch.zeros(N, self.k, dtype=torch.float32)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk = features_norm[start:end]


            sim_chunk = torch.mm(chunk, features_norm.T)


            for i in range(end - start):
                sim_chunk[i, start + i] = -float('inf')


            topk_vals, topk_idx = torch.topk(sim_chunk, self.k, dim=1)

            if self.weight_type == 'exponential':
                topk_weights = torch.exp(topk_vals / self.tau)
            else:
                topk_weights = torch.ones_like(topk_vals)


            all_topk_indices[start:end] = topk_idx.cpu()
            all_topk_weights[start:end] = topk_weights.cpu()

            del sim_chunk, topk_vals, topk_idx, topk_weights
            torch.cuda.empty_cache()


        self.neighbors = all_topk_indices
        self.neighbor_weights = all_topk_weights

    def update_global_probs(self, probs, indices=None):
        with torch.no_grad():
            if indices is not None:
                self.global_probs[indices] = probs.detach()
            else:
                self.global_probs = probs.detach()

    def forward(self, logits, labels, node_indices, **kwargs):
        if not self.active:
            return torch.tensor(0.0, device=logits.device)

        probs = F.softmax(logits, dim=1)
        self.update_global_probs(probs, node_indices)
        return self._space_aware_regularization_loss(probs, node_indices)

    def _edge_divergence(self, pi_flat: torch.Tensor, pj_flat: torch.Tensor) -> torch.Tensor:
        if self.divergence_type == 'js':
            return js_divergence_torch(pi_flat, pj_flat)
        elif self.divergence_type == 'fisher':
            p_src = pi_flat.clamp_min(self.eps_fisher)
            return ((pi_flat - pj_flat) ** 2 / (8.0 * math.log(2.0) * p_src)).sum(dim=1)
        elif self.divergence_type == 'kl':
            pic = pi_flat.clamp_min(1e-10); pjc = pj_flat.clamp_min(1e-10)
            return (pic * (torch.log2(pic) - torch.log2(pjc))).sum(dim=1)
        else:
            return ((pi_flat - pj_flat) ** 2).sum(dim=1)

    def _space_aware_regularization_loss(self, probs: torch.Tensor, node_indices: torch.Tensor) -> torch.Tensor:
        device = probs.device
        neighbor_idx = self.neighbors[node_indices].to(device)
        W = self.neighbor_weights[node_indices].to(device)

        pi = probs
        pj = self.global_probs[neighbor_idx]

        B, k, num_classes = pj.shape

        pi_expanded = pi.unsqueeze(1).expand(B, k, num_classes).reshape(-1, num_classes)
        pj_flat = pj.reshape(-1, num_classes)

        div = self._edge_divergence(pi_expanded, pj_flat)

        div = div.reshape(B, k)

        if self.mask_mode == 'hard':

            yi = pi.argmax(dim=1, keepdim=True)
            yj = pj.argmax(dim=-1)
            W = W * (yi == yj).to(W.dtype)
        elif self.mask_mode == 'oracle':

            assert self.node_labels is not None, "set .node_labels for oracle masking"
            yi = self.node_labels[node_indices.cpu()].to(div.device).unsqueeze(1)
            yj = self.node_labels[neighbor_idx.cpu()].to(div.device)
            W = W * (yi == yj).to(W.dtype)

        self.last_keep_frac = float((W > 0).float().mean().item())
        return (W * div).sum() / (W.sum() + 1e-8)

