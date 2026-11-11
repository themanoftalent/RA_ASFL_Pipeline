"""Run methods with matched random seeds."""
import argparse
import copy
from pathlib import Path
import yaml
from ra_asfl.train import run


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/mitbih.yaml')
    p.add_argument('--out', default='results/comparison')
    p.add_argument('--seeds', type=int, nargs='+', default=[11, 22, 33, 44, 55])
    p.add_argument('--methods', nargs='+', default=['fedavg', 'splitfed', 'adaptsfl', 'ra_asfl'])
    args = p.parse_args()
    original = yaml.safe_load(Path(args.config).read_text())
    for seed in args.seeds:
        for method in args.methods:
            cfg = copy.deepcopy(original)
            cfg.update(seed=seed, method=method)
            destination = Path(args.out)/f'seed_{seed}'/method
            if (destination/'results.json').exists():
                import json
                existing = json.loads((destination/'results.json').read_text())
                if existing.get('seed') == seed and existing.get('method') == method and existing.get('config') == cfg:
                    print('Already complete:', destination)
                    continue
                raise RuntimeError(f'Existing results use different settings: {destination}')
            print('Training', seed, method, flush=True)
            run(cfg, destination)


if __name__ == '__main__':
    main()
