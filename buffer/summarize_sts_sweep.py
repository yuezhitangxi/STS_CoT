import argparse
import csv
import json
from pathlib import Path


def read_status(path):
    values = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            key, _, value = line.partition('\t')
            values[key] = value
    return values


def read_norm(path):
    values = {}
    if not path.is_file():
        return values
    with path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            values[row['tensor_name']] = row
    return values


def mean_position(summary, key):
    values = summary.get(key, [])
    return sum(values) / len(values) if values else ''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('result_dir', type=Path)
    args = parser.parse_args()

    rows = []
    for run_dir in sorted(path for path in args.result_dir.iterdir() if path.is_dir()):
        status = read_status(run_dir / 'status.tsv')
        result_path = run_dir / 'eval_results.json'
        if not result_path.is_file():
            rows.append({
                'run': run_dir.name,
                'state': status.get('state', 'PENDING'),
                'bank_size': status.get('bank_size', ''),
                'temperature': status.get('temperature', ''),
            })
            continue

        result = json.loads(result_path.read_text())
        summary = result['summary']
        norms = read_norm(run_dir / 'eval_norm_stats_summary.csv')
        norm_value = lambda name: norms.get(name, {}).get('avg_mean_norm', '')
        rows.append({
            'run': run_dir.name,
            'state': status.get('state', 'DONE'),
            'bank_size': result['args']['sts_bank_size'],
            'temperature': result['args']['sts_temperature'],
            'accuracy': summary['accuracy'],
            'correct': summary['correct'],
            'total': summary['total'],
            'train_seconds': status.get('train_seconds', ''),
            'eval_seconds': summary['evaluation_seconds'],
            'input_embedding_norm': norm_value('llm_input_token_embedding.raw_prompt'),
            'hidden_norm': norm_value('projection_input.hidden_state'),
            'query_norm': norm_value('sts_query'),
            'bank_norm': norm_value('sts_soft_token_bank'),
            'sts_output_norm': norm_value('sts_output'),
            'final_prompt_norm': norm_value('llm_input_token_embedding.final_prompt'),
            'attention_entropy': mean_position(summary, 'mean_attention_entropy_by_position'),
            'attention_effective_tokens': mean_position(summary, 'mean_attention_effective_tokens_by_position'),
            'attention_top1': mean_position(summary, 'mean_attention_top1_by_position'),
        })

    columns = [
        'run', 'state', 'bank_size', 'temperature', 'accuracy', 'correct', 'total',
        'train_seconds', 'eval_seconds', 'input_embedding_norm', 'hidden_norm',
        'query_norm', 'bank_norm', 'sts_output_norm', 'final_prompt_norm',
        'attention_entropy', 'attention_effective_tokens', 'attention_top1',
    ]
    with (args.result_dir / 'summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

    completed = [row for row in rows if row.get('accuracy', '') != '']
    completed.sort(key=lambda row: float(row['accuracy']), reverse=True)
    lines = [
        '# STS Sweep Summary',
        '',
        '| Run | State | N | Tau | Accuracy | Train hours | Eval hours | STS norm | Entropy | Top-1 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in completed:
        train_hours = float(row['train_seconds']) / 3600 if row['train_seconds'] else 0.0
        eval_hours = float(row['eval_seconds']) / 3600
        lines.append(
            f"| {row['run']} | {row['state']} | {row['bank_size']} | {row['temperature']} | "
            f"{float(row['accuracy']):.2f}% | {train_hours:.2f} | {eval_hours:.2f} | "
            f"{float(row['sts_output_norm']):.4f} | {float(row['attention_entropy']):.4f} | "
            f"{float(row['attention_top1']):.4f} |"
        )
    (args.result_dir / 'summary.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
