"""Score ECG quality using training data."""
import numpy as np


def descriptors(x):
    # Input: records, leads, samples. Higher penalties mean lower quality.
    x = np.asarray(x, dtype=np.float32)
    base = np.abs(np.mean(x[:, :, :max(2, x.shape[-1]//8)], axis=-1) -
                  np.mean(x[:, :, -max(2, x.shape[-1]//8):], axis=-1)).mean(-1)
    hf = np.mean(np.diff(x, n=2, axis=-1)**2, axis=(1, 2))**0.5
    spectrum = np.abs(np.fft.rfft(x, axis=-1)).mean(1)
    narrow = spectrum.max(axis=-1) / np.maximum(spectrum.mean(axis=-1), 1e-8)
    missing = (~np.isfinite(x)).mean(axis=(1, 2))
    clipped = (np.abs(x) > 8).mean(axis=(1, 2))
    morphology = np.mean(np.abs(x - np.median(x, axis=-1, keepdims=True)), axis=(1, 2))
    return np.stack((base, hf, narrow, missing+clipped, morphology), axis=1)


class QualityScorer:
    def fit(self, training_signals):
        d = descriptors(training_signals)
        self.low = np.percentile(d, 5, axis=0)
        self.high = np.percentile(d, 95, axis=0)
        return self

    def transform(self, signals):
        d = descriptors(signals)
        penalties = np.clip((d-self.low)/np.maximum(self.high-self.low, 1e-8), 0, 1)
        return np.maximum(0.1, 1-penalties.mean(axis=1)).astype(np.float32)
