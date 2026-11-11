import numpy as np
import torch
from ra_asfl.model import ECGNet
from ra_asfl.data import grouped_subject_split, allocate_clients
from ra_asfl.metrics import evaluate
from ra_asfl.reliability import contribution_weights
from ra_asfl.controller import PROFILES, cut_costs, select_split
from ra_asfl.reliability import ClientState
from ra_asfl.train import clip_activation


def test_all_cuts_have_same_logits_in_eval_mode():
    torch.manual_seed(3)
    for channels, length in [(1, 250), (12, 1000)]:
        network = ECGNet(channels).eval()
        x = torch.randn(2, channels, length)
        with torch.no_grad():
            reference = network(x)
            for cut in range(1, 6):
                result = network.suffix(network.prefix(x, cut), cut)
                assert torch.allclose(result, reference, atol=1e-6)


def test_no_group_leakage_and_complete_allocation():
    groups = np.repeat(np.arange(50), 5)
    y = np.tile(np.arange(5), 50)
    train, valid, test = grouped_subject_split(groups)
    assert not (set(groups[train]) & set(groups[valid]))
    assert not (set(groups[valid]) & set(groups[test]))
    clients = allocate_clients(train, y, groups, 3, alpha=10, seed=9)
    assert set(np.concatenate(clients)) == set(train)
    assert sum(map(len, clients)) == len(train)
    for i, a in enumerate(clients):
        for b in clients[i+1:]:
            assert not (set(groups[a]) & set(groups[b]))


def test_aggregation_weights_sum_to_one():
    clients = [{'n': 20, 'support': 15, 'reliability': .7, 'uncertainty': .2,
                'heterogeneity': .4, 'consistency': .6},
               {'n': 50, 'support': 40, 'reliability': .5, 'uncertainty': .4,
                'heterogeneity': .3, 'consistency': .5}]
    for flag in [True, False]:
        w = contribution_weights(clients, reliability_aware=flag)
        assert np.isclose(w.sum(), 1) and np.all(w > 0)


def test_clip_and_feasible_split():
    x = torch.randn(3, 10, 20)
    clipped = clip_activation(x, radius=1.0)
    assert torch.all(clipped.flatten(1).norm(dim=1) <= 1.000001)
    model = ECGNet()
    costs = cut_costs(model, (1, 250), 16, PROFILES[0])
    for row in costs:
        row['budget_mb'] = 256
    assert select_split(costs, ClientState()) in range(1, 6)


def test_multilabel_accuracy_definitions():
    labels = np.array([[1, 0, 1, 0, 0], [0, 1, 0, 0, 1]])
    probs = labels * .8 + (1-labels)*.2
    result = evaluate(labels, probs, multilabel=True)
    assert result['label_accuracy'] == 1 and result['exact_match_accuracy'] == 1
