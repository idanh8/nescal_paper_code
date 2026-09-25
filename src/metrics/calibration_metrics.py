import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.calibration import calibration_curve
import relplot as rp
from viz.vizlib import plot_calibration_analysis
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Multi-class Brier score:\
    """
    N, C = probs.shape
    one_hot = np.eye(C)[labels]
    return np.mean(np.sum((probs - one_hot) ** 2, axis=1))


def ece_mce(probabilities, labels, num_bins: int = 50):
    predictions = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)

    bins = np.linspace(0, 1, num_bins + 1)
    bin_indices = np.clip(np.digitize(confidences, bins) - 1, 0, num_bins - 1)
    bin_accuracies = np.zeros(num_bins)
    bin_confidences = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)

    for i in range(num_bins):
        mask = bin_indices == i
        bin_counts[i] = np.sum(mask)
        if bin_counts[i] > 0:
            bin_accuracies[i] = np.mean(predictions[mask] == labels[mask])
            bin_confidences[i] = np.mean(confidences[mask])

    ece_value = np.sum((bin_counts / len(labels)) * np.abs(bin_accuracies - bin_confidences))
    mce_value = np.max(np.abs(bin_accuracies - bin_confidences)) if np.any(bin_counts > 0) else 0
    return ece_value, mce_value, bin_confidences, bin_accuracies, bin_counts


def ece_mce_classwise(all_probs, all_labels, num_bins: int = 50):
    ece_list, mce_list = [], []
    all_bin_confidences, all_bin_accuracies, all_bin_counts = [], [], []

    for i in range(all_probs.shape[1]):
        p_i, y_i = all_probs[:, i], (all_labels == i).astype(float)
        ids = np.clip(np.digitize(p_i, np.linspace(0., 1., num_bins + 1)) - 1, 0, num_bins - 1)
        cnt = np.bincount(ids, minlength=num_bins); nz = cnt > 0
        prob_true = np.bincount(ids, weights=y_i, minlength=num_bins)[nz] / cnt[nz]
        prob_pred = np.bincount(ids, weights=p_i, minlength=num_bins)[nz] / cnt[nz]
        hist = cnt[nz]
        ece = np.sum(np.abs(prob_true - prob_pred) * hist) / len(all_probs)
        mce = np.max(np.abs(prob_true - prob_pred)) if len(prob_true) > 0 else 0.0

        ece_list.append(ece)
        mce_list.append(mce)
        all_bin_confidences.append(prob_pred)
        all_bin_accuracies.append(prob_true)

        mask = all_labels == i
        hist_true_class, _ = np.histogram(all_probs[mask, i], bins=np.linspace(0., 1., num_bins + 1))
        all_bin_counts.append(hist_true_class)

    return np.mean(ece_list), np.mean(mce_list), all_bin_confidences, all_bin_accuracies, all_bin_counts, ece_list, mce_list


def ece_mce_full_prob(probabilities, labels, num_bins: int = 50, class_index: int = 1):
    f = probabilities[:, class_index]
    y = (labels == class_index).astype(float)

    bins = np.linspace(0, 1, num_bins + 1)
    bin_indices = np.clip(np.digitize(f, bins) - 1, 0, num_bins - 1)
    bin_accuracies = np.zeros(num_bins)
    bin_confidences = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)

    for i in range(num_bins):
        mask = bin_indices == i
        bin_counts[i] = np.sum(mask)
        if bin_counts[i] > 0:
            bin_accuracies[i] = np.mean(y[mask])
            bin_confidences[i] = np.mean(f[mask])

    ece_value = np.sum((bin_counts / len(labels)) * np.abs(bin_accuracies - bin_confidences))
    mce_value = np.max(np.abs(bin_accuracies - bin_confidences)) if np.any(bin_counts > 0) else 0
    return ece_value, mce_value, bin_confidences, bin_accuracies, bin_counts


def ece_hist_equal_mass(p, label, n_bins: int = 15, order: int = 1):
    p = np.clip(p, 1e-256, 1 - 1e-256)
    N = p.shape[0]
    label_index = label if label.ndim == 1 else np.array([np.where(r == 1)[0][0] for r in label])

    with torch.no_grad():
        if p.shape[1] != 2:
            preds_new = torch.from_numpy(p)
            preds_b = torch.zeros(N, 1)
            label_binary = np.zeros((N, 1))
            for i in range(N):
                pred_label = int(torch.argmax(preds_new[i]).numpy())
                if pred_label == label_index[i]:
                    label_binary[i] = 1
                preds_b[i] = preds_new[i, pred_label] / torch.sum(preds_new[i, :])
        else:
            preds_b = torch.from_numpy((p / np.sum(p, 1)[:, None])[:, 1])
            label_binary = label_index

        confidences = preds_b
        accuracies = torch.from_numpy(label_binary)
        x = np.sort(confidences.numpy(), axis=0)
        bin_count = int(len(x) / n_bins)
        bins = np.array([x[min((i + 1) * bin_count, x.shape[0] - 1)] for i in range(n_bins)])
        bin_boundaries = torch.zeros(len(bins) + 1, 1)
        bin_boundaries[1:] = torch.from_numpy(bins).reshape(-1, 1)
        bin_boundaries[0], bin_boundaries[-1] = 0.0, 1.0
        bin_lowers, bin_uppers = bin_boundaries[:-1], bin_boundaries[1:]

        ece_avg = torch.zeros(1)
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece_avg += torch.abs(avg_confidence_in_bin - accuracy_in_bin) ** order * prop_in_bin
    return ece_avg

class DegenerateKernelError(RuntimeError):
    """Raised when the LCE kernel carries no local information."""


def fit_lce_projection(features, n_components: int = 50, seed: int = 0):
    """
    Adapted from Barbera et al.'s implementation (2026). Same holds for all LCE functions here.
    """
    n_components = min(n_components, np.asarray(features).shape[1])
    features = np.asarray(features, dtype=np.float32)
    scaler = StandardScaler().fit(features)
    pca = PCA(n_components=n_components, svd_solver='full', random_state=seed).fit(scaler.transform(features))
    return {'scaler': scaler, 'pca': pca, 'n_components': n_components}


def apply_lce_projection(projection, features):
    features = np.asarray(features, dtype=np.float32)
    return projection['pca'].transform(projection['scaler'].transform(features)).astype(np.float32)

def compute_lce(probs: torch.Tensor, y_true: torch.Tensor, features: torch.Tensor, n_bins: int = 15,
                 gamma: float = 10.0, filter_thr: int = 20, return_per_class: bool = False,
                 strict: bool = True):
    """
    Local Calibration Error (Luo et al. 2022, Implementation from Barbera et al. 2026)
    """
    device = probs.device
    N = probs.shape[0]
    eps = 1e-12

    if probs.dim() == 1 or probs.shape[1] == 1:
        p_pos = probs.view(-1).to(device)
        n_classes = 1
    elif probs.shape[1] == 2:
        p_pos = probs[:, 1].to(device)
        n_classes = 1
    else:
        n_classes = probs.shape[1]

    features = features.to(device).float()
    if features.dim() == 1:
        features = features.view(N, 1)

    per_class_avg, per_class_max = [], []
    ess_all = []

    if n_classes == 1:
        labels_bin = y_true.float() if y_true.dtype != torch.float32 else y_true
        p_class = p_pos
        classes_to_process = [0]
    else:
        classes_to_process = range(n_classes)

    for c in classes_to_process:
        if n_classes > 1:
            labels_bin = (y_true == c).float()
            p_class = probs[:, c]

        bin_edges = torch.linspace(0.0, 1.0, n_bins + 1, device=device)
        bin_idx = torch.bucketize(p_class.contiguous(), bin_edges, right=True).clamp(1, n_bins)
        lce_vals = torch.full((N,), float('nan'), device=device)

        for b in range(1, n_bins + 1):
            idx = (bin_idx == b).nonzero(as_tuple=True)[0]
            if idx.numel() <= filter_thr:
                continue

            pca_bin = features[idx]
            e_sub = (p_class[idx] - labels_bin[idx])
            s = idx.numel()
            chunk_size = 1024
            lce_sub = torch.zeros(s, device=device)
            sq_norms_all = (pca_bin ** 2).sum(dim=1)

            for start in range(0, s, chunk_size):
                end = min(start + chunk_size, s)
                X = pca_bin[start:end]
                sq_norms_X = (X ** 2).sum(dim=1, keepdim=True)
                D = torch.clamp(sq_norms_X + sq_norms_all.unsqueeze(0) - 2 * (X @ pca_bin.t()), min=0.0)
                K = torch.exp(-D / (2.0 * gamma ** 2))
                numer = K @ e_sub
                denom = K.sum(dim=1)
                denom_safe = denom.clone()
                zero_mask = denom_safe <= eps
                denom_safe[zero_mask] = 1.0
                lce_chunk = numer / (denom_safe + eps)
                lce_chunk[zero_mask] = 0.0
                lce_sub[start:end] = lce_chunk


                rs = denom - 1.0
                ssq = (K ** 2).sum(dim=1) - 1.0
                ess_chunk = torch.where(rs > eps, rs.pow(2) / (ssq + eps),
                                        torch.ones_like(rs))
                ess_all.append(ess_chunk)


            lce_vals[idx] = lce_sub

        valid = torch.isfinite(lce_vals)
        if valid.any():
            lce_abs = torch.abs(lce_vals[valid])
            per_class_avg.append(lce_abs.mean().item())
            per_class_max.append(lce_abs.max().item())
        else:
            per_class_avg.append(0.0)
            per_class_max.append(0.0)

    med_ess = float(torch.cat(ess_all).median().item()) if ess_all else 1.0
    if strict and med_ess < 2.0:
        raise DegenerateKernelError(
            f'LCE kernel is degenerate: median self-excluded ESS = {med_ess:.2f} (< 2). '
            f'dim={features.shape[1]}, gamma={gamma}. The kernel has collapsed to the identity '
            f'and LCE would equal the per-sample absolute residual. Project features with '
            f'fit_lce_projection/apply_lce_projection before calling compute_lce.')

    avg_lce = float(np.mean(per_class_avg))
    max_lce = float(np.max(per_class_max))
    if return_per_class:
        return avg_lce, max_lce, med_ess, per_class_avg, per_class_max
    return avg_lce, max_lce, med_ess


def local_freq_error(probs, labels, features, lce_projection=None, gamma=10.0, device='cpu', chunk=2048):
    import torch
    p = torch.as_tensor(np.asarray(probs), dtype=torch.float32, device=device); N, C = p.shape
    Y = torch.nn.functional.one_hot(torch.as_tensor(np.asarray(labels), dtype=torch.long, device=device), C).float()
    phi = apply_lce_projection(lce_projection, features) if lce_projection is not None else np.asarray(features, np.float32)
    phi = torch.as_tensor(phi, dtype=torch.float32, device=device); sig = float(gamma) ** 2
    sq = (phi ** 2).sum(1); js_sum = topl_sum = 0.0
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        d2 = torch.clamp(sq[s:e, None] + sq[None, :] - 2 * (phi[s:e] @ phi.T), min=0.0)
        w = torch.exp(-d2 / (2 * sig)); w[torch.arange(e - s), torch.arange(s, e)] = 0.0
        w = w / (w.sum(1, keepdim=True) + 1e-12); q = w @ Y
        pi = p[s:e]; mid = 0.5 * (pi + q)
        kl = lambda a, b: (a.clamp_min(1e-12) * (a.clamp_min(1e-12).log() - b.clamp_min(1e-12).log())).sum(1)
        js_sum += (0.5 * kl(pi, mid) + 0.5 * kl(q, mid)).clamp_min(0).sqrt().sum().item()
        conf, pred = pi.max(1); topl_sum += (conf - q.gather(1, pred[:, None]).squeeze(1)).abs().sum().item()
    return {'lfe_js': js_sum / N, 'lfe_topl_abs': topl_sum / N}

def proper_score_decomposition(probs, labels, num_bins=15):
    """Calibration-refinement decomposition of the top-label Brier score """
    probs = np.asarray(probs); labels = np.asarray(labels); N = len(labels)
    conf = probs.max(1); correct = (probs.argmax(1) == labels).astype(float)
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, num_bins - 1)
    cal = ref = 0.0
    for b in range(num_bins):
        mask = idx == b; nb = int(mask.sum())
        if nb == 0: continue
        conf_b = float(conf[mask].mean()); acc_b = float(correct[mask].mean()); w = nb / N
        cal += w * (conf_b - acc_b) ** 2
        ref += w * acc_b * (1.0 - acc_b)
    return {'brier_toplabel_cal': float(cal), 'brier_toplabel_ref': float(ref),
            'brier_toplabel': float(cal + ref)}


def calibration_metrics(all_softmax_probs, all_labels, all_features, num_bins: int = 15,
                         plots: bool = False, plot_params: dict = None,
                         lce_projection=None, lce_gamma: float = 10.0, lce_device: str = 'cpu'):
    if lce_projection is None:
        raise ValueError(
            'calibration_metrics requires lce_projection (see fit_lce_projection). '
            'Passing raw features to the LCE kernel produces a silently degenerate metric.')
    lce_feats = apply_lce_projection(lce_projection, all_features)
    predictions = all_softmax_probs.argmax(axis=1)
    confidences = all_softmax_probs.max(axis=1)
    probs_t = torch.tensor(all_softmax_probs, dtype=torch.float32)
    labels_t = torch.tensor(all_labels, dtype=torch.long)

    bs = brier_score(all_softmax_probs, all_labels)
    ece, mce, bcnfs, baccss, bcnts = ece_mce(all_softmax_probs, all_labels, num_bins=num_bins)
    cwece, cwmce, cw_bin_confidences, cw_bin_accuracies, cw_bin_counts, cw_ece_list, cw_mce_list = \
        ece_mce_classwise(all_softmax_probs, all_labels, num_bins=num_bins)

    correct = [1 if predictions[i] == all_labels[i] else 0 for i in range(len(all_labels))]

    emece = ece_hist_equal_mass(all_softmax_probs, all_labels, n_bins=num_bins).detach().cpu().item()


    nll = F.nll_loss(torch.log(probs_t.clamp(min=1e-12)), labels_t).item()
    _dev = torch.device(lce_device)
    lce, mlce, med_ess = compute_lce(probs_t.to(_dev), labels_t.to(_dev), torch.from_numpy(lce_feats).to(_dev), num_bins, gamma=lce_gamma)

    results = {
        'ece': ece, 'mce': mce, 'avg_cw_ece': cwece, 'avg_cw_mce': cwmce, 'brier_score': bs, 'nll': nll,
        'equal_mass_ece': emece,
        'lce': lce.item() if isinstance(lce, torch.Tensor) else lce,
        'mlce': mlce.item() if isinstance(mlce, torch.Tensor) else mlce,
        'median_ess': med_ess,

    }
    results.update(proper_score_decomposition(all_softmax_probs, all_labels, num_bins=num_bins))

    if plots:
        original_style = plt.rcParams.copy()
        if len(plot_params['class_names']) == 2:
            _, __, bcnfs, baccss, bcnts = ece_mce_full_prob(all_softmax_probs, all_labels, num_bins=num_bins, class_index=1)
            fig, ax = rp.rel_diagram(all_softmax_probs[:, 1], all_labels, plot_density=True, plot_density_ticks=False, plot_labels=False)
        else:
            fig, ax = rp.rel_diagram(confidences, correct, plot_density=True, plot_density_ticks=False, plot_labels=False)
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        ax.set_title('smECE Reliability Diagram')
        fig.savefig(os.path.join(plot_params['save_path'], f"{plot_params['savename']}_smooth_reliability_diagram.png"),
                    dpi=300, bbox_inches='tight')
        plt.close('all')
        plt.rcParams.update(original_style)

        cw_plots = len(plot_params['class_names']) <= 10
        plot_calibration_analysis(bcnfs, baccss, bcnts, cw_bin_confidences, cw_bin_accuracies, cw_bin_counts,
                                   cw_ece_list, cw_mce_list, plot_params['class_names'], plot_params['save_path'],
                                   savename=plot_params['savename'], classwise=cw_plots, residual_analysis=False)

    return results
