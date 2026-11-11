"""Run one experiment."""
import argparse
import json
from pathlib import Path
import yaml
from ra_asfl.train import run


def main():
    parser = argparse.ArgumentParser(description='Run a documented RA-ASFL simulation')
    parser.add_argument('--config', default='configs/mitbih.yaml')
    parser.add_argument('--out', default='results/run')
    parser.add_argument('--method', choices=['ra_asfl', 'fedavg', 'splitfed', 'adaptsfl'])
    parser.add_argument('--seed', type=int)
    parser.add_argument('--smoke', action='store_true', help='Only a synthetic software integration test')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if args.method:
        config['method'] = args.method
    if args.seed is not None:
        config['seed'] = args.seed
    if args.smoke:
        config.update({'dataset': 'synthetic_smoke', 'rounds': 1, 'clients': 3,
                       'batch_size': 32, 'mc_passes': 2})
    result = run(config, args.out)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
