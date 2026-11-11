"""Choose splits using simulated device resources."""
from dataclasses import dataclass
import numpy as np
import torch


@dataclass(frozen=True)
class DeviceProfile:
    name: str
    compute: float
    memory_mb: float
    up_mbps: float
    down_mbps: float
    rtt_ms: float


PROFILES = (
    DeviceProfile('low', 0.5, 256, 2, 5, 100),
    DeviceProfile('medium', 1., 512, 10, 20, 50),
    DeviceProfile('high', 2., 1024, 50, 100, 20),
)


def cut_costs(model, shape, batch, profile, device='cpu'):
    """Estimate traffic, memory, latency, and energy for each split."""
    model.eval()
    with torch.no_grad():
        x = torch.zeros((1, *shape), device=device)
        details = []
        cumulative = 0
        for idx, block in enumerate(model.blocks):
            x = block(x)
            cumulative += sum(p.numel() for p in block.parameters())
            elements = x.numel() * batch
            traffic = (2 * elements + 2 * cumulative) * 4
            communication = 8*elements*4/(profile.up_mbps*1e6) + 8*elements*4/(profile.down_mbps*1e6)
            communication += profile.rtt_ms/1000
            # Include model, optimizer, and activation memory.
            memory = (12*cumulative + 16*elements*4)/1048576
            compute_proxy = cumulative / (1e7 * profile.compute)
            latency = communication + compute_proxy
            energy_proxy = compute_proxy + 0.1*communication
            details.append({'split': idx+1, 'bytes': traffic, 'memory_mb': memory,
                            'latency_proxy': latency, 'energy_proxy': energy_proxy,
                            'privacy_proxy': (idx+1)/5})
    return details


def select_split(costs, state, adaptive=True, reliability_aware=True, fixed=3,
                 privacy=None):
    if not adaptive:
        return fixed
    feasible = [c for c in costs if c['memory_mb'] <= c.get('budget_mb', float('inf'))]
    if not feasible:
        raise RuntimeError('No split meets the simulated client memory budget')
    keys = ['bytes', 'latency_proxy', 'memory_mb', 'energy_proxy', 'privacy_proxy']
    values = {k: np.asarray([r[k] for r in feasible], dtype=float) for k in keys}
    normalized = {k: (v-v.min())/(np.ptp(v)+1e-9) for k, v in values.items()}
    score = np.zeros(len(feasible))
    for k in keys[:4]:
        score += normalized[k]
    if reliability_aware:
        score += normalized['privacy_proxy']
        score += np.array([state.split_uncertainty[r['split']-1] for r in feasible])
        score += np.array([1-state.split_utility[r['split']-1] for r in feasible])
        score /= 7
    else:
        score /= 4
    return feasible[int(np.argmin(score))]['split']


def update_split_state(state, split, uncertainty, validation_utility, decay=0.9):
    u, v = list(state.split_uncertainty), list(state.split_utility)
    i = split - 1
    u[i] = decay*u[i] + (1-decay)*uncertainty
    v[i] = decay*v[i] + (1-decay)*validation_utility
    state.split_uncertainty, state.split_utility = tuple(u), tuple(v)
