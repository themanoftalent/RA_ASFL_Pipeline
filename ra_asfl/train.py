"""Run split federated training in one process."""
import copy
import json
import math
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .model import ECGNet
from .data import ECGDataset, allocate_clients, load_mitbih, load_ptbxl
from .quality import QualityScorer
from .metrics import evaluate, choose_thresholds
from .reliability import (ClientState, effective_support, jensen_shannon, consistency,
                          update_reliability, contribution_weights, personalization_gate,
                          mc_uncertainty)
from .controller import PROFILES, cut_costs, select_split, update_split_state


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def private_keys(state):
    """Keep the classifier and BN state local."""
    return [key for key in state if key.startswith('head.4.') or
            ('.running_' in key or '.num_batches_tracked' in key or
             ('.net.1.' in key and key.startswith('blocks.0.')) or
             ('.main.2.' in key and key.startswith('blocks.')) or
             ('.skip.1.' in key and key.startswith('blocks.')) or
             ('.project.1.' in key and key.startswith('blocks.')))]


def clip_activation(tensor, radius=1.0, noise=0.0):
    if radius is None:
        return tensor
    flat = tensor.flatten(1)
    scale = (radius / flat.norm(dim=1, keepdim=True).clamp_min(1e-9)).clamp(max=1)
    output = (flat*scale).reshape_as(tensor)
    if noise > 0:
        output = output + torch.randn_like(output)*noise
    return output


def local_train(model, loader, split, task, cfg, device, round_number, global_state):
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    # Detach at the cut to pass gradients back explicitly.
    loss_sum, observed = 0., 0
    for xb, yb, qb in loader:
        xb, yb, qb = xb.to(device), yb.to(device), qb.to(device)
        optim.zero_grad(set_to_none=True)
        smashed = model.prefix(xb, split)
        interface = clip_activation(smashed, cfg['clip_radius'], cfg['noise_scale'])
        bridge = interface.detach().requires_grad_(True)
        logits = model.suffix(bridge, split)
        if task == 'ptbxl':
            per_sample = F.binary_cross_entropy_with_logits(logits, yb, reduction='none').mean(1)
        else:
            per_sample = F.cross_entropy(logits, yb, reduction='none')
        loss = (per_sample*qb).sum()/qb.sum().clamp_min(1e-8)
        if cfg.get('calibration_weight', 0) > 0:
            probs = logits.sigmoid() if task == 'ptbxl' else logits.softmax(-1)
            target = yb if task == 'ptbxl' else F.one_hot(yb, probs.size(-1)).float()
            loss = loss + cfg['calibration_weight']*(probs-target).square().mean()
        if cfg.get('personalization_regularizer', 0) > 0:
            reg = sum((p-global_state[n].to(device)).square().sum()
                      for n, p in model.named_parameters() if n.startswith('head.4.'))
            loss = loss + cfg['personalization_regularizer']*reg
        loss.backward()
        # Backpropagate through the cut and client layers.
        torch.autograd.backward(interface, bridge.grad)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optim.step()
        loss_sum += float(loss.detach())*len(xb)
        observed += len(xb)
    return loss_sum/max(observed, 1)


@torch.no_grad()
def predict(model, dataset, device, task, batch_size=128):
    model.eval()
    targets, probabilities, losses = [], [], []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for xb, yb, _ in loader:
        logits = model(xb.to(device))
        p = logits.sigmoid() if task == 'ptbxl' else logits.softmax(-1)
        loss = (F.binary_cross_entropy_with_logits(logits, yb.to(device), reduction='none').mean(-1)
                if task == 'ptbxl' else F.cross_entropy(logits, yb.to(device), reduction='none'))
        targets.append(yb.numpy())
        probabilities.append(p.cpu().numpy())
        losses.extend(loss.cpu().tolist())
    return np.concatenate(targets), np.concatenate(probabilities), float(np.mean(losses))


def model_delta(local, global_state):
    return {k: (v.detach().cpu().float()-global_state[k].float()) for k, v in
            local.state_dict().items() if torch.is_floating_point(v)}


def aggregate(global_model, local_models, weights):
    original = global_model.state_dict()
    updates = [m.state_dict() for m in local_models]
    combined = {}
    for key, old in original.items():
        if torch.is_floating_point(old):
            combined[key] = sum(float(w)*state[key].to(old.device) for w, state in zip(weights, updates))
        else:
            combined[key] = old
    global_model.load_state_dict(combined)


def profile_assignment(clients, seed):
    rng = np.random.default_rng(seed)
    return rng.choice(len(PROFILES), size=clients, p=(.3, .4, .3))


def evaluate_model(model, dataset, task, device, thresholds=None):
    targets, probs, loss = predict(model, dataset, device, task)
    metrics = evaluate(targets, probs, task == 'ptbxl', thresholds)
    metrics['loss'] = loss
    return metrics


def run(config, output_dir):
    cfg = dict(config)
    seed = cfg['seed']
    set_seed(seed)
    output = Path(output_dir)
    if any((output / name).exists() for name in ('results.json', 'history.json', 'checkpoint.pt')):
        raise FileExistsError(f'Experiment output already exists: {output}. Choose a new output directory to preserve the existing run.')
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.get('device', 'cpu'))
    task = cfg['dataset']
    if task == 'mitbih':
        x, y, subjects, (tr, va, te) = load_mitbih(cfg['data_root'], seed)
    elif task == 'ptbxl':
        x, y, subjects, (tr, va, te) = load_ptbxl(cfg['data_root'])
    elif task == 'synthetic_smoke':
        rng = np.random.default_rng(seed)
        subjects = np.repeat(np.arange(50), 10)
        y = np.tile(np.arange(5), 100)
        x = rng.normal(size=(500, 1, 250)).astype(np.float32)
        # Random inputs are for testing the code only.
        from .data import grouped_subject_split
        tr, va, te = grouped_subject_split(subjects, seed)
        task = 'mitbih'
    else:
        raise ValueError(f'Unknown dataset: {task}')
    scorer = QualityScorer().fit(x[tr])
    quality = scorer.transform(x)
    allocation = allocate_clients(tr, y, subjects, cfg['clients'], cfg['alpha'], seed,
                                  cfg.get('quantity_sigma', 0.0))
    validation = ECGDataset(x[va], y[va], quality[va])
    test = ECGDataset(x[te], y[te], quality[te])
    channels = x.shape[1]
    global_model = ECGNet(channels).to(device)
    states = [ClientState() for _ in allocation]
    retained = [copy.deepcopy(global_model.state_dict()) for _ in allocation]
    profiles = profile_assignment(len(allocation), seed)
    rng = np.random.default_rng(seed + 987)
    history = []
    previous_delta = None
    patience = 0
    best_score = -float('inf')
    best_state = None
    previous_loss = [None]*len(allocation)
    labels_global = y[tr].sum(0) if y.ndim == 2 else np.bincount(y[tr], minlength=5)
    method = cfg.get('method', 'ra_asfl')
    allowed = {'ra_asfl', 'fedavg', 'splitfed', 'adaptsfl'}
    if method not in allowed:
        raise ValueError(f'Implemented methods: {sorted(allowed)}')
    adaptive = method in {'ra_asfl', 'adaptsfl'}
    reliable = method == 'ra_asfl'
    for rnd in range(cfg['rounds']):
        completed = [k for k in range(len(allocation)) if rng.random() >= cfg['dropout']]
        if len(completed) < 2:
            completed = rng.choice(len(allocation), size=2, replace=False).tolist()
        current_global = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
        local_models, summaries, trace = [], [], []
        for k in completed:
            indices = allocation[k]
            profile = PROFILES[int(profiles[k])]
            local = copy.deepcopy(global_model)
            if reliable:
                initialized = local.state_dict()
                for key in private_keys(initialized):
                    initialized[key] = retained[k][key].to(initialized[key].device).clone()
                local.load_state_dict(initialized)
            costs = cut_costs(local, x.shape[1:], cfg['batch_size'], profile, str(device))
            for cost in costs:
                cost['budget_mb'] = profile.memory_mb
            split = (5 if method == 'fedavg' else
                     select_split(costs, states[k], adaptive, reliable, cfg['fixed_split']))
            train_data = ECGDataset(x[indices], y[indices], quality[indices] if reliable else None)
            loader = DataLoader(train_data, batch_size=cfg['batch_size'], shuffle=True)
            train_loss = local_train(local, loader, split, task, cfg, device, rnd, current_global)
            _, val_prob, validation_loss = predict(local, validation, device, task, cfg['batch_size'])
            val_metrics = evaluate(y[va], val_prob, task == 'ptbxl')
            score = val_metrics['macro_auroc'] if task == 'ptbxl' else val_metrics['balanced_accuracy']
            if score is None:
                score = 0.
            subset = y[indices]
            local_labels = subset.sum(0) if subset.ndim == 2 else np.bincount(subset, minlength=5)
            heterogeneity = jensen_shannon(local_labels, labels_global)
            sample_idx = indices[:min(len(indices), cfg.get('uncertainty_samples', 32))]
            signals = torch.as_tensor(x[sample_idx], device=device)
            unc, epistemic = mc_uncertainty(local, signals, task == 'ptbxl', cfg['mc_passes'])
            delta = model_delta(local, current_global)
            align = consistency(delta, previous_delta)
            support = effective_support(quality[indices])
            if reliable:
                rel = update_reliability(states[k], float(quality[indices].mean()), unc,
                                         heterogeneity, align, score, validation_loss,
                                         support, max(len(p) for p in allocation), cfg['reliability_ema'])
            else:
                rel = 0.5
            update_split_state(states[k], split, unc, float(np.clip(score, 0, 1)),
                               cfg['split_ema'])
            local_models.append(local)
            summaries.append({'client': k, 'n': len(indices), 'support': support,
                              'reliability': rel, 'uncertainty': unc,
                              'heterogeneity': heterogeneity, 'consistency': align})
            trace.append({'client': k, 'profile': profile.name, 'split': split,
                          'train_loss': train_loss, 'validation_loss': validation_loss,
                          'uncertainty': unc, 'epistemic': epistemic,
                          'heterogeneity': heterogeneity, 'reliability': rel,
                          'traffic_bytes_proxy': costs[split-1]['bytes']})
        weights = contribution_weights(summaries, cfg['temperature'], reliable)
        aggregate(global_model, local_models, weights)
        latest = global_model.state_dict()
        previous_delta = {k: latest[k].cpu().float()-current_global[k].float() for k in
                          current_global if torch.is_floating_point(current_global[k])}
        if reliable:
            for local, summary in zip(local_models, summaries):
                k = summary['client']
                gate = personalization_gate(summary['reliability'], summary['uncertainty'],
                                            summary['heterogeneity'])
                state = local.state_dict()
                for key in private_keys(state):
                    if torch.is_floating_point(state[key]):
                        retained[k][key] = ((1-gate)*state[key].cpu() +
                                            gate*latest[key].detach().cpu()).clone()
                    else:
                        retained[k][key] = state[key].detach().cpu().clone()
        labels_val, probability_val, _ = predict(global_model, validation, device, task)
        metrics = evaluate(labels_val, probability_val, task == 'ptbxl')
        primary = metrics['macro_auroc'] if task == 'ptbxl' else metrics['macro_f1']
        entry = {'round': rnd+1, 'validation': metrics, 'clients': trace,
                 'weights': {str(s['client']): float(w) for s, w in zip(summaries, weights)}}
        history.append(entry)
        if primary is not None and primary > best_score + cfg.get('min_delta', 1e-5):
            best_score, best_state, patience = primary, copy.deepcopy(global_model.state_dict()), 0
        else:
            patience += 1
        if patience >= cfg['patience']:
            break
    if best_state is None:
        raise RuntimeError('Training yielded no valid validation metric')
    global_model.load_state_dict(best_state)
    labels_val, probabilities_val, _ = predict(global_model, validation, device, task)
    thresholds = choose_thresholds(labels_val, probabilities_val) if task == 'ptbxl' else None
    final = evaluate_model(global_model, test, task, device, thresholds)
    metadata = {'dataset': cfg['dataset'], 'method': method, 'seed': seed,
                'train_records': len(tr), 'validation_records': len(va),
                'test_records': len(te), 'classes': 5, 'config': cfg,
                'test_metrics': final, 'selection_validation_score': best_score,
                'best_round': int(np.argmax([h['validation']['macro_auroc'] if task == 'ptbxl'
                                              else h['validation']['macro_f1'] for h in history]))+1,
                'ptbxl_validation_thresholds': thresholds.tolist() if thresholds is not None else None,
                'simulation_notice': 'Software simulation, not physical IoMT or clinical evaluation.'}
    (output/'history.json').write_text(json.dumps(history, indent=2))
    (output/'results.json').write_text(json.dumps(metadata, indent=2))
    torch.save({'model': best_state, 'metadata': metadata}, output/'checkpoint.pt')
    return metadata
