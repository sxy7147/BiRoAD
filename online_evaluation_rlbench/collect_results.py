"""Summarize paired-role success rates for the eight-task benchmark."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_processing.task_config import TASK_PAIRS


def pair_metrics(base, swapped):
    return (
        (base + swapped) / 2,
        2 * base * swapped / (base + swapped) if base + swapped else 0.0,
        min(base, swapped),
        abs(base - swapped),
    )


def collect_results(folder):
    rows = []
    for base_task, swapped_task in TASK_PAIRS:
        rates = []
        for task in (base_task, swapped_task):
            path = folder / task / "eval.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing evaluation result: {path}")
            with path.open() as stream:
                rates.append(100 * json.load(stream)[task]["mean"])
        base, swapped = rates
        rows.append((base_task, base, swapped, *pair_metrics(base, swapped)))
    averages = [sum(row[i] for row in rows) / len(rows) for i in range(1, 7)]
    return rows + [("Average", *averages)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--output_file', type=Path)
    args = parser.parse_args()
    rows = collect_results(args.folder)
    lines = ['Task\tBase\tSwapped\tMean\tHM\tWorst\tGap']
    lines.extend(
        '\t'.join([row[0]] + [f'{value:.2f}' for value in row[1:]])
        for row in rows
    )
    text = '\n'.join(lines) + '\n'
    print(text, end='')
    output = args.output_file or args.folder / 'all_results.tsv'
    output.write_text(text)


if __name__ == '__main__':
    main()
