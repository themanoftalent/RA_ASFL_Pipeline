"""Load local ECG files and assign records to clients."""
import ast
import pathlib
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


MIT_LABELS = ('A', 'L', 'N', 'R', 'V')
PTB_LABELS = ('NORM', 'MI', 'STTC', 'CD', 'HYP')


class ECGDataset(Dataset):
    def __init__(self, x, y, quality=None):
        self.x = np.asarray(x, dtype=np.float32)
        self.y = np.asarray(y)
        self.quality = (np.ones(len(x), dtype=np.float32) if quality is None else
                        np.asarray(quality, dtype=np.float32))
        assert len(self.x) == len(self.y) == len(self.quality)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        import torch
        y = torch.tensor(self.y[index], dtype=(torch.float32 if self.y.ndim == 2 else torch.long))
        return torch.from_numpy(self.x[index]), y, torch.tensor(self.quality[index])


def zscore_per_beat(x):
    return ((x - x.mean(axis=-1, keepdims=True)) /
            np.maximum(x.std(axis=-1, keepdims=True), 1e-6)).astype(np.float32)


def grouped_subject_split(subjects, seed=42, train_fraction=0.7, valid_fraction=0.15):
    unique = np.unique(subjects)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_train = int(len(unique) * train_fraction)
    n_val = int(len(unique) * valid_fraction)
    groups = [unique[:n_train], unique[n_train:n_train+n_val], unique[n_train+n_val:]]
    output = [np.flatnonzero(np.isin(subjects, group)) for group in groups]
    if any(not len(group) for group in output):
        raise ValueError('Too few distinct subjects for non-empty train/validation/test splits')
    return output


def load_mitbih(root, seed=42, beat_length=250):
    """Load selected MIT-BIH beats, keeping records 201 and 202 together."""
    import wfdb
    root = pathlib.Path(root)
    record_ids = [line.strip() for line in (root / 'RECORDS').read_text().splitlines() if line.strip()]
    beats, labels, subjects = [], [], []
    half = beat_length // 2
    for record_id in record_ids:
        rec = wfdb.rdrecord(str(root / record_id))
        ann = wfdb.rdann(str(root / record_id), 'atr')
        waveform = np.asarray(rec.p_signal[:, 0], dtype=np.float32)
        for position, symbol in zip(ann.sample, ann.symbol):
            if symbol not in MIT_LABELS or position < half or position + beat_length-half > len(waveform):
                continue
            beat = waveform[position-half:position-half+beat_length]
            if len(beat) != beat_length or not np.all(np.isfinite(beat)):
                continue
            beats.append(beat[None, :])
            labels.append(MIT_LABELS.index(symbol))
            subjects.append('201_202' if record_id in {'201', '202'} else record_id)
    if not beats:
        raise ValueError('No annotated beats found. Check the MIT-BIH directory.')
    x = zscore_per_beat(np.stack(beats))
    y = np.asarray(labels, dtype=np.int64)
    subjects = np.asarray(subjects)
    splits = grouped_subject_split(subjects, seed)
    return x, y, subjects, splits


def load_ptbxl(root):
    """Load PTB-XL with its diagnostic labels and official folds."""
    import wfdb
    root = pathlib.Path(root)
    meta = pd.read_csv(root / 'ptbxl_database.csv', index_col='ecg_id')
    statements = pd.read_csv(root / 'scp_statements.csv', index_col=0)
    diagnostic = statements[statements['diagnostic'] == 1]
    x, y, groups, folds = [], [], [], []
    for _, row in meta.iterrows():
        codes = ast.literal_eval(row['scp_codes'])
        labels = np.zeros(5, dtype=np.float32)
        for code in codes:
            if code in diagnostic.index:
                superclass = diagnostic.loc[code, 'diagnostic_class']
                if superclass in PTB_LABELS:
                    labels[PTB_LABELS.index(superclass)] = 1
        if not labels.any():
            continue
        signal, _ = wfdb.rdsamp(str(root / row['filename_lr']))
        if signal.shape != (1000, 12) or not np.isfinite(signal).all():
            continue
        x.append(signal.T.astype(np.float32))
        y.append(labels)
        groups.append(int(row['patient_id']))
        folds.append(int(row['strat_fold']))
    x = np.stack(x)
    y = np.stack(y)
    groups, folds = np.asarray(groups), np.asarray(folds)
    train_idx, val_idx, test_idx = [np.flatnonzero(np.isin(folds, k)) for k in
                                     (range(1, 9), [9], [10])]
    mu = x[train_idx].mean(axis=(0, 2), keepdims=True)
    sd = x[train_idx].std(axis=(0, 2), keepdims=True)
    x = ((x - mu) / np.maximum(sd, 1e-6)).astype(np.float32)
    assert not (set(groups[train_idx]) & set(groups[val_idx]))
    assert not (set(groups[train_idx]) & set(groups[test_idx]))
    assert not (set(groups[val_idx]) & set(groups[test_idx]))
    return x, y, groups, [train_idx, val_idx, test_idx]


def allocate_clients(train_indices, y, groups, clients, alpha=0.5, seed=42,
                     quantity_sigma=0.0):
    """Dirichlet label skew, with each training subject assigned only once.

    For multilabel data, the first active superclass is an allocation key only.
    All original target labels are preserved for training.
    """
    rng = np.random.default_rng(seed)
    group_values = np.unique(groups[train_indices])
    label_keys = {}
    for group in group_values:
        subset = train_indices[groups[train_indices] == group]
        yy = y[subset]
        counts = yy.sum(0) if yy.ndim == 2 else np.bincount(yy.astype(int), minlength=5)
        label_keys[group] = int(np.argmax(counts))
    allocation = [[] for _ in range(clients)]
    quantity = rng.lognormal(mean=0, sigma=quantity_sigma, size=clients)
    quantity /= quantity.sum()
    for cls in range(5):
        class_groups = [g for g in group_values if label_keys[g] == cls]
        rng.shuffle(class_groups)
        proportions = rng.dirichlet(np.full(clients, alpha)) * quantity
        proportions /= proportions.sum()
        assignments = rng.choice(clients, len(class_groups), p=proportions)
        for group, client_id in zip(class_groups, assignments):
            allocation[int(client_id)].extend(train_indices[groups[train_indices] == group].tolist())
    # Move whole subject groups when balancing clients.
    while any(not items for items in allocation):
        empty = next(i for i, items in enumerate(allocation) if not items)
        donor = max(range(clients), key=lambda i: len(set(groups[allocation[i]])))
        donor_groups = np.unique(groups[allocation[donor]])
        if len(donor_groups) < 2:
            raise ValueError('Too few subject groups to populate all clients')
        move_group = donor_groups[-1]
        moved = [item for item in allocation[donor] if groups[item] == move_group]
        allocation[donor] = [item for item in allocation[donor] if groups[item] != move_group]
        allocation[empty].extend(moved)
    flat = [int(i) for part in allocation for i in part]
    assert len(flat) == len(train_indices) and len(set(flat)) == len(flat)
    return [np.asarray(items, dtype=int) for items in allocation]
