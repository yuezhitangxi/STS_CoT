
import csv
import json
import math
import os
import time
from collections import defaultdict

import torch
import torch.nn.functional as F


class NormMonitor:
    def __init__(self, jsonl_path, max_records=2000):
        self.jsonl_path = jsonl_path
        self.max_records = int(max_records) if max_records is not None else 2000
        self.records_written = 0
        self.disabled = not jsonl_path
        self.aggregates = defaultdict(lambda: {
            'count': 0,
            'mean_norm_sum': 0.0,
            'std_norm_sum': 0.0,
            'min_norm': math.inf,
            'max_norm': -math.inf,
            'mean_abs_sum': 0.0,
            'std_abs_sum': 0.0,
            'nan_count': 0,
            'inf_count': 0,
        })
        self._fh = None
        if not self.disabled:
            os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
            self._fh = open(jsonl_path, 'a', buffering=1)

    def _active(self):
        return (not self.disabled) and (self.records_written < self.max_records)

    def record_tensor(self, name, tensor, meta=None):
        if not self._active() or tensor is None:
            return
        with torch.no_grad():
            t = tensor.detach().float()
            nan_mask = torch.isnan(t)
            inf_mask = torch.isinf(t)
            clean = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
            if clean.numel() == 0:
                return
            if clean.dim() >= 1:
                norms = clean.norm(dim=-1).reshape(-1)
            else:
                norms = clean.reshape(-1).abs()
            rec = {
                'time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'tensor_name': name,
                'shape': list(t.shape),
                'numel': int(t.numel()),
                'mean_norm': float(norms.mean().item()),
                'std_norm': float(norms.std(unbiased=False).item()) if norms.numel() > 1 else 0.0,
                'min_norm': float(norms.min().item()),
                'max_norm': float(norms.max().item()),
                'mean_abs': float(clean.abs().mean().item()),
                'std_abs': float(clean.abs().std(unbiased=False).item()) if clean.numel() > 1 else 0.0,
                'nan_count': int(nan_mask.sum().item()),
                'inf_count': int(inf_mask.sum().item()),
            }
            if meta:
                rec.update(meta)
            self._fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
            self.records_written += 1
            agg = self.aggregates[name]
            agg['count'] += 1
            agg['mean_norm_sum'] += rec['mean_norm']
            agg['std_norm_sum'] += rec['std_norm']
            agg['min_norm'] = min(agg['min_norm'], rec['min_norm'])
            agg['max_norm'] = max(agg['max_norm'], rec['max_norm'])
            agg['mean_abs_sum'] += rec['mean_abs']
            agg['std_abs_sum'] += rec['std_abs']
            agg['nan_count'] += rec['nan_count']
            agg['inf_count'] += rec['inf_count']

    def record_cosine(self, name, left, right, meta=None):
        if not self._active() or left is None or right is None:
            return
        with torch.no_grad():
            a = torch.nan_to_num(left.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
            b = torch.nan_to_num(right.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
            if a.shape != b.shape or a.numel() == 0:
                return
            cos = F.cosine_similarity(a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1]), dim=-1)
            rec = {
                'time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'tensor_name': name,
                'shape': list(cos.shape),
                'numel': int(cos.numel()),
                'mean_norm': float(cos.mean().item()),
                'std_norm': float(cos.std(unbiased=False).item()) if cos.numel() > 1 else 0.0,
                'min_norm': float(cos.min().item()),
                'max_norm': float(cos.max().item()),
                'mean_abs': float(cos.abs().mean().item()),
                'std_abs': float(cos.abs().std(unbiased=False).item()) if cos.numel() > 1 else 0.0,
                'nan_count': int(torch.isnan(cos).sum().item()),
                'inf_count': int(torch.isinf(cos).sum().item()),
            }
            if meta:
                rec.update(meta)
            self._fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
            self.records_written += 1
            agg = self.aggregates[name]
            agg['count'] += 1
            agg['mean_norm_sum'] += rec['mean_norm']
            agg['std_norm_sum'] += rec['std_norm']
            agg['min_norm'] = min(agg['min_norm'], rec['min_norm'])
            agg['max_norm'] = max(agg['max_norm'], rec['max_norm'])
            agg['mean_abs_sum'] += rec['mean_abs']
            agg['std_abs_sum'] += rec['std_abs']
            agg['nan_count'] += rec['nan_count']
            agg['inf_count'] += rec['inf_count']

    def close(self):
        if self.disabled:
            return
        summary_path = self.jsonl_path.replace('.jsonl', '_summary.csv')
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'tensor_name', 'records', 'avg_mean_norm', 'avg_std_norm',
                'global_min_norm', 'global_max_norm', 'avg_mean_abs', 'avg_std_abs',
                'nan_count', 'inf_count',
            ])
            writer.writeheader()
            for name, agg in sorted(self.aggregates.items()):
                c = max(agg['count'], 1)
                writer.writerow({
                    'tensor_name': name,
                    'records': agg['count'],
                    'avg_mean_norm': agg['mean_norm_sum'] / c,
                    'avg_std_norm': agg['std_norm_sum'] / c,
                    'global_min_norm': agg['min_norm'] if agg['min_norm'] != math.inf else '',
                    'global_max_norm': agg['max_norm'] if agg['max_norm'] != -math.inf else '',
                    'avg_mean_abs': agg['mean_abs_sum'] / c,
                    'avg_std_abs': agg['std_abs_sum'] / c,
                    'nan_count': agg['nan_count'],
                    'inf_count': agg['inf_count'],
                })
        if self._fh:
            self._fh.close()
            self._fh = None
