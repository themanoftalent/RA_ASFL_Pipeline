"""Client reliability and aggregation weights."""
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F


def effective_support(quality):
    w = np.asarray(quality, dtype=float)
    return float(w.sum() ** 2 / max(float(np.dot(w, w)), 1e-9))


def jensen_shannon(p, q):
    p = np.asarray(p, float) + 1e-9
    q = np.asarray(q, float) + 1e-9
    p /= p.sum()
    q /= q.sum()
    mid = (p + q) / 2
    return float((np.sum(p * np.log(p / mid)) + np.sum(q * np.log(q / mid))) / (2 * np.log(2)))


def mmd_rbf(x, y, max_points=128):
    if x is None or y is None or len(x) < 2 or len(y) < 2:
        return 0.0
    x = torch.as_tensor(x[:max_points], dtype=torch.float32)
    y = torch.as_tensor(y[:max_points], dtype=torch.float32)
    combined = torch.cat((x, y))
    with torch.no_grad():
        distances = torch.pdist(combined).square()
        bandwidth = distances[distances > 0].median().clamp_min(1e-6) if (distances > 0).any() else 1.
        kxx = torch.exp(-torch.cdist(x, x).square() / bandwidth).mean()
        kyy = torch.exp(-torch.cdist(y, y).square() / bandwidth).mean()
        kxy = torch.exp(-torch.cdist(x, y).square() / bandwidth).mean()
    return float((kxx + kyy - 2 * kxy).clamp_min(0))


def mc_uncertainty(model, signals, multilabel=False, passes=10):
    """Estimate uncertainty with dropout and fixed batch normalization."""
    original = model.training
    model.eval()
    for layer in model.modules():
        if isinstance(layer, torch.nn.Dropout):
            layer.train()
    with torch.no_grad():
        outputs = torch.stack([model(signals).sigmoid() if multilabel else
                               model(signals).softmax(-1) for _ in range(passes)])
        mean = outputs.mean(0).clamp(1e-7, 1-1e-7)
        if multilabel:
            entropy = lambda x: -(x * x.log() + (1-x) * (1-x).log()).mean(-1) / np.log(2)
        else:
            entropy = lambda x: -(x * x.log()).sum(-1) / np.log(mean.shape[-1])
        predictive = entropy(mean)
        expected = entropy(outputs.clamp(1e-7, 1-1e-7)).mean(0)
        epistemic = (predictive - expected).clamp_min(0)
        uncertainty = (predictive + epistemic).div(2).clamp(0, 1)
    model.train(original)
    return float(uncertainty.mean()), float(epistemic.mean())


def consistency(delta, previous_delta):
    if previous_delta is None:
        return 0.5
    a = torch.cat([v.float().flatten().cpu() for v in delta.values()])
    b = torch.cat([previous_delta[k].float().flatten().cpu() for k in delta])
    if a.norm() < 1e-12 or b.norm() < 1e-12:
        return 0.5
    return float(((F.cosine_similarity(a[None], b[None]) + 1) / 2).item())


@dataclass
class ClientState:
    reliability: float = 0.5
    val_loss_ema: float = 0.5
    uncertainty: float = 0.5
    heterogeneity: float = 0.5
    split_utility: tuple = (0.5, 0.5, 0.5, 0.5, 0.5)
    split_uncertainty: tuple = (0.5, 0.5, 0.5, 0.5, 0.5)
    previous_loss: float = None


def update_reliability(state, quality, uncertainty, heterogeneity, consistency_score,
                       predictive_quality, validation_loss, support, support_scale,
                       smoothing=0.9):
    stability = np.exp(-abs(validation_loss - state.val_loss_ema))
    normalized_support = np.log1p(support) / max(np.log1p(support_scale), 1e-9)
    normalized_support = float(np.clip(normalized_support, 0, 1))
    # Fixed weights give each factor its intended direction.
    predictors = np.array([predictive_quality, quality, 1-uncertainty, 1-heterogeneity,
                           consistency_score, stability, normalized_support]) - 0.5
    logit = float(predictors.mean() * 4)
    instantaneous = 1 / (1 + np.exp(-logit))
    state.reliability = smoothing * state.reliability + (1-smoothing) * instantaneous
    state.val_loss_ema = smoothing * state.val_loss_ema + (1-smoothing) * validation_loss
    state.uncertainty = uncertainty
    state.heterogeneity = heterogeneity
    state.previous_loss = validation_loss
    return state.reliability


def contribution_weights(summaries, temperature=1.0, reliability_aware=True):
    if not reliability_aware:
        n = np.asarray([s['n'] for s in summaries], dtype=float)
        return n / n.sum()
    scores = []
    supports = [s['support'] for s in summaries]
    top = max(supports)
    for s in summaries:
        support = np.log1p(s['support']) / max(np.log1p(top), 1e-9)
        scores.append(2*s['reliability'] + support + s['consistency'] -
                      s['uncertainty'] - 0.5*s['heterogeneity'])
    scores = np.asarray(scores) / max(temperature, 1e-8)
    scores -= scores.max()
    probs = np.exp(scores)
    return probs / probs.sum()


def personalization_gate(reliability, uncertainty, heterogeneity):
    value = 2*(reliability-0.5) - (uncertainty-0.5) - (heterogeneity-0.5)
    return float(1 / (1 + np.exp(-value)))
