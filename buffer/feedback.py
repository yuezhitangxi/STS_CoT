import json
import math
from collections import Counter
from pathlib import Path

import torch


def match_feedback_norm(vectors, target_norm):
    """Rescale nonzero vectors in float32, preserving their directions."""
    if target_norm is None:
        return vectors
    values = vectors.float()
    norms = values.norm(dim=-1, keepdim=True)
    nonzero = norms > 0
    denominator = torch.where(nonzero, norms, torch.ones_like(norms))
    # Zero-initialized projectors keep their identity gradient for the first update.
    scales = torch.where(nonzero, target_norm / denominator, torch.ones_like(norms))
    return (values * scales).to(vectors.dtype)


@torch.no_grad()
def estimate_embedding_norm(model, tokenizer, train_ds, preprocess_fn):
    """Measure token-frequency-weighted embedding norms on training prompts."""
    counts = Counter()
    special_ids = set(tokenizer.all_special_ids)
    for ins in train_ds:
        inputs = preprocess_fn(
            ins, tokenizer, num_thought_tokens=model.num_thought_tokens,
            split='test', device=None,
        )
        start, end = inputs['thought_index'][:2]
        counts.update(
            token for position, token in enumerate(inputs['input_ids'])
            if not start <= position < end and token not in special_ids
        )
    if not counts:
        raise ValueError('Training prompts contain no ordinary input tokens.')

    weight = model.model.get_input_embeddings().weight
    items = list(counts.items())
    norm_sum = 0.0
    for start in range(0, len(items), 4096):
        chunk = items[start:start + 4096]
        ids = torch.tensor([token for token, _ in chunk], device=weight.device)
        frequencies = torch.tensor(
            [count for _, count in chunk], device=weight.device, dtype=torch.float64,
        )
        norms = weight.index_select(0, ids).float().norm(dim=-1).double()
        norm_sum += (norms * frequencies).sum().item()
    num_tokens = sum(counts.values())
    return {
        'embedding_norm': norm_sum / num_tokens,
        'calibration_split': 'train',
        'calibration_source': 'prompt_tokens_excluding_special_and_thought_tokens',
        'calibration_prompts': len(train_ds),
        'calibration_tokens': num_tokens,
    }


def configure_feedback(model, tokenizer, train_ds, preprocess_fn, *, mode=None,
                       norm=None, scale=1.0, checkpoint=None):
    """Resolve the experiment scale or restore a trained checkpoint's scale."""
    saved_config = None
    if checkpoint and checkpoint != 'None':
        config_path = Path(checkpoint).resolve().parent.parent / 'run_config.json'
        if config_path.is_file():
            saved_config = json.loads(config_path.read_text()).get('feedback')
    if mode is None and norm is None and scale == 1.0 and saved_config is not None:
        model.feedback_norm = saved_config['target_norm']
        return saved_config

    mode = mode or ('scale_match' if norm is not None or scale != 1.0 else 'vanilla')
    if mode == 'vanilla':
        if norm is not None or scale != 1.0:
            raise ValueError('A feedback norm or scale requires scale_match mode.')
        model.feedback_norm = None
        return {'mode': mode, 'target_norm': None}
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('feedback_scale must be finite and positive.')
    if norm is None:
        calibration = estimate_embedding_norm(model, tokenizer, train_ds, preprocess_fn)
        reference_norm = calibration['embedding_norm']
    else:
        calibration = {'reference_norm_source': 'explicit'}
        reference_norm = norm
    target_norm = reference_norm * scale
    if not math.isfinite(target_norm) or target_norm <= 0:
        raise ValueError('The feedback target norm must be finite and positive.')
    model.feedback_norm = target_norm
    return {
        'mode': mode, 'target_norm': target_norm,
        'reference_norm': reference_norm, 'scale': scale, **calibration,
    }
