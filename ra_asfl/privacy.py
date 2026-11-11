"""Optional empirical attack diagnostics; do not interpret as differential privacy.

Reconstruction attacks need auxiliary raw signals accessible to the evaluator.
This module never claims that activation clipping is end-to-end DP.
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


class ReconstructionDecoder(nn.Module):
    def __init__(self, activation_channels, signal_length):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose1d(activation_channels, 64, 3, stride=2, padding=1,
                               output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(32, 1, 3, stride=2, padding=1, output_padding=1))
        self.signal_length = signal_length

    def forward(self, smashed):
        return F.interpolate(self.model(smashed), size=self.signal_length,
                             mode='linear', align_corners=False)


def train_reconstruction_decoder(auxiliary_activations, auxiliary_signals,
                                 epochs=100, batch_size=64, lr=0.001, device='cpu'):
    """Fit only on an auxiliary population disjoint from target-model members."""
    x = torch.as_tensor(auxiliary_activations, dtype=torch.float32, device=device)
    y = torch.as_tensor(auxiliary_signals, dtype=torch.float32, device=device)
    if y.shape[1] != 1:
        raise ValueError('The reference decoder reconstructs one ECG lead')
    decoder = ReconstructionDecoder(x.shape[1], y.shape[-1]).to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    for _ in range(epochs):
        for idx in torch.randperm(len(x), device=device).split(batch_size):
            optimizer.zero_grad()
            loss = F.mse_loss(decoder(x[idx]), y[idx])
            loss.backward()
            optimizer.step()
    return decoder


def reconstruction_metrics(reference, reconstruction):
    reference, reconstruction = np.asarray(reference), np.asarray(reconstruction)
    if reference.shape != reconstruction.shape:
        raise ValueError('Reconstructed and reference signals must have equal shape')
    error = reference-reconstruction
    rmse = np.sqrt(np.mean(error**2, axis=(-1, -2)))
    ranges = np.ptp(reference, axis=(-1, -2)).clip(min=1e-8)
    centered = reference-reference.mean(axis=-1, keepdims=True)
    prd = 100*np.sqrt(np.sum(error**2, axis=(-1, -2)) /
                      np.maximum(np.sum(centered**2, axis=(-1, -2)), 1e-8))
    return {'nrmse': float(np.mean(rmse/ranges)), 'prd_percent': float(np.mean(prd))}


def membership_attack(train_features, train_membership, test_features, test_membership):
    """Evaluate a separately fitted attack classifier on a disjoint attack test set."""
    clf = LogisticRegression(max_iter=1000)
    clf.fit(np.asarray(train_features), np.asarray(train_membership))
    prob = clf.predict_proba(np.asarray(test_features))[:, 1]
    return {'attack_auroc': float(roc_auc_score(test_membership, prob))}


def conditional_rdp_bound(clip_radius, noise_sd, affected_releases, delta, orders=None):
    """Restricted Gaussian activation-channel analytical bound only.

    Replacement adjacency has L2 sensitivity at most 2C. The encoder, split,
    output dimension and release schedule must satisfy the manuscript's fixed or
    previously accounted conditions. It excludes gradients, labels and updates.
    """
    if noise_sd <= 0:
        return {'epsilon': None, 'reason': 'No Gaussian privacy guarantee at zero noise'}
    if not 0 < delta < 1:
        raise ValueError('delta must lie strictly between zero and one')
    orders = np.asarray(orders if orders is not None else np.arange(2, 129), float)
    rdp = affected_releases*orders*(2*clip_radius)**2/(2*noise_sd**2)
    eps = rdp + np.log(1/delta)/(orders-1)
    i = int(np.argmin(eps))
    return {'epsilon': float(eps[i]), 'delta': delta, 'renyi_order': float(orders[i]),
            'scope': 'conditional activation channel only, not end-to-end training'}
