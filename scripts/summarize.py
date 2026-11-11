"""Summarize completed experiments."""
import argparse
import csv
from pathlib import Path
from ra_asfl.analysis import collect_runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('folder')
    p.add_argument('--metric', default='macro_f1')
    p.add_argument('--out', default='results/summary.csv')
    args = p.parse_args()
    records = collect_runs(args.folder, args.metric)
    if not records:
        raise SystemExit('No results.json files found')
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f'Saved {len(records)} actual runs to {args.out}')


if __name__ == '__main__':
    main()
