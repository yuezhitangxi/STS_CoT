import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _load_cached_centroids(cache_path, bank_size, hidden_size):
    if not cache_path or not Path(cache_path).is_file():
        return None
    payload = torch.load(cache_path, map_location='cpu', weights_only=True)
    centroids = payload['centroids'] if isinstance(payload, dict) else payload
    expected_shape = (bank_size, hidden_size)
    if tuple(centroids.shape) != expected_shape:
        raise ValueError(
            f'Cached STS bank has shape {tuple(centroids.shape)}, expected {expected_shape}.'
        )
    return centroids.float()


def _run_faiss_kmeans(vectors, bank_size, seed, niter, device):
    try:
        import faiss
    except ImportError as exc:
        raise ImportError(
            'FAISS is required to initialize a new STS bank. Install faiss-gpu '
            'for GPU clustering or faiss-cpu for CPU clustering, or provide an '
            'existing --sts_bank_cache.'
        ) from exc

    gpu_available = hasattr(faiss, 'StandardGpuResources')
    if device == 'gpu' and not gpu_available:
        raise RuntimeError('GPU KMeans was requested, but this FAISS build has no GPU support.')
    use_gpu = gpu_available if device == 'auto' else device == 'gpu'
    max_points = max(256, math.ceil(vectors.shape[0] / bank_size))
    kmeans = faiss.Kmeans(
        vectors.shape[1],
        bank_size,
        niter=niter,
        nredo=1,
        seed=seed,
        verbose=True,
        gpu=use_gpu,
        min_points_per_centroid=1,
        max_points_per_centroid=max_points,
    )
    kmeans.train(vectors)
    return torch.from_numpy(np.asarray(kmeans.centroids).copy()).float(), use_gpu


@torch.no_grad()
def initialize_kmeans_bank(
    embedding_weight,
    bank_size,
    excluded_token_ids=None,
    cache_path=None,
    seed=42,
    niter=20,
    device='auto',
    bank_norm_scale=1.0,
):
    if not math.isfinite(bank_norm_scale) or bank_norm_scale <= 0:
        raise ValueError('bank_norm_scale must be finite and positive')
    if niter <= 0:
        raise ValueError('niter must be positive')
    if device not in {'auto', 'cpu', 'gpu'}:
        raise ValueError(f'Unknown KMeans device: {device}')
    hidden_size = embedding_weight.size(1)
    cached = _load_cached_centroids(cache_path, bank_size, hidden_size)
    if cached is not None:
        return cached.mul(bank_norm_scale)

    excluded = set(excluded_token_ids or [])
    candidate_ids = [i for i in range(embedding_weight.size(0)) if i not in excluded]
    if bank_size > len(candidate_ids):
        raise ValueError('bank_size exceeds the number of available vocabulary tokens')

    ids = torch.tensor(candidate_ids, device=embedding_weight.device)
    vectors = embedding_weight.detach().index_select(0, ids).float().cpu().numpy()
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    centroids, used_gpu = _run_faiss_kmeans(vectors, bank_size, seed, niter, device)

    if cache_path:
        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
        torch.save({
            'centroids': centroids,
            'bank_size': bank_size,
            'hidden_size': hidden_size,
            'seed': seed,
            'niter': niter,
            'excluded_token_ids': sorted(excluded),
            'faiss_gpu': used_gpu,
        }, temporary)
        os.replace(temporary, path)
    return centroids.mul(bank_norm_scale)


class SoftTokenQuery(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.query = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False,
            dtype=torch.bfloat16,
        )
        nn.init.xavier_uniform_(self.query.weight)
        self.last_query = None
        self.last_attention = None
        self.last_output = None

    def forward(self, hidden_state, soft_token_bank, temperature):
        query = self.query(hidden_state)
        logits = F.linear(query.float(), soft_token_bank.float()) / temperature
        attention = torch.softmax(logits, dim=-1)
        output = torch.matmul(attention, soft_token_bank.float()).to(hidden_state.dtype)
        self.last_query = query.detach()
        self.last_attention = attention.detach()
        self.last_output = output.detach()
        return output


class SharedSoftTokenSelector(nn.Module):
    """Position-specific queries over one shared learnable soft token bank."""

    def __init__(self, hidden_size, num_positions, bank_size, temperature, initial_bank):
        super().__init__()
        if bank_size <= 0:
            raise ValueError('bank_size must be positive')
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('temperature must be finite and positive')
        expected_shape = (bank_size, hidden_size)
        if tuple(initial_bank.shape) != expected_shape:
            raise ValueError(
                f'Initial STS bank has shape {tuple(initial_bank.shape)}, expected {expected_shape}.'
            )

        self.hidden_size = int(hidden_size)
        self.bank_size = int(bank_size)
        self.temperature = float(temperature)
        self.soft_token_bank = nn.Parameter(initial_bank.to(dtype=torch.bfloat16))
        self.queries = nn.ModuleList([
            SoftTokenQuery(hidden_size) for _ in range(num_positions)
        ])

    def forward(self, hidden_state, position):
        return self.queries[position](
            hidden_state,
            self.soft_token_bank,
            self.temperature,
        )

    def detached_selection_stats(self, position):
        module = self.queries[position]
        attention = module.last_attention.float().reshape(-1)
        entropy = -(attention * attention.clamp_min(1e-12).log()).sum()
        return {
            'query_norm': float(module.last_query.float().norm().item()),
            'bank_norm': float(self.soft_token_bank.detach().float().norm(dim=-1).mean().item()),
            'output_norm': float(module.last_output.float().norm().item()),
            'attention_entropy': float(entropy.item()),
            'attention_effective_tokens': float(entropy.exp().item()),
            'attention_top1': float(attention.max().item()),
            'attention_argmax': int(attention.argmax().item()),
        }

    @torch.no_grad()
    def bank_diagnostics(self):
        bank = torch.nan_to_num(self.soft_token_bank.detach().float())
        gram = bank @ bank.transpose(0, 1)
        singular_values = torch.linalg.eigvalsh(gram).clamp_min(0).sqrt()
        mass = singular_values.sum()
        if mass > 0:
            probabilities = singular_values / mass
            effective_rank = torch.exp(
                -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
            )
        else:
            effective_rank = mass.new_zeros(())

        normalized = F.normalize(bank, dim=-1)
        cosine = normalized @ normalized.transpose(0, 1)
        off_diagonal = cosine[~torch.eye(
            self.bank_size, device=cosine.device, dtype=torch.bool,
        )]
        if off_diagonal.numel() == 0:
            off_diagonal = cosine.new_zeros(1)
        return {
            'effective_rank': float(effective_rank.item()),
            'cosine_mean': float(off_diagonal.mean().item()),
            'cosine_std': float(off_diagonal.std(unbiased=False).item()),
            'cosine_min': float(off_diagonal.min().item()),
            'cosine_max': float(off_diagonal.max().item()),
        }
