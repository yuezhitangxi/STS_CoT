import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftTokenSelector(nn.Module):
    """Map one hidden state to a convex combination of learned token vectors."""

    def __init__(
        self,
        hidden_size,
        bank_size,
        temperature,
        embedding_weight,
        excluded_token_ids=None,
        bank_norm_scale=1.0,
        init_seed=0,
    ):
        super().__init__()
        if bank_size <= 0:
            raise ValueError("bank_size must be positive")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not math.isfinite(bank_norm_scale) or bank_norm_scale <= 0:
            raise ValueError("bank_norm_scale must be finite and positive")

        self.hidden_size = int(hidden_size)
        self.bank_size = int(bank_size)
        self.temperature = float(temperature)
        self.bank_norm_scale = float(bank_norm_scale)

        self.query = nn.Linear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
            dtype=torch.bfloat16,
        )
        with torch.no_grad():
            self.query.weight.zero_()
            self.query.weight.diagonal().fill_(1.0)

        excluded = set(excluded_token_ids or [])
        candidates = [i for i in range(embedding_weight.size(0)) if i not in excluded]
        if self.bank_size > len(candidates):
            raise ValueError("bank_size exceeds the number of available vocabulary tokens")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(init_seed))
        selected = torch.randperm(len(candidates), generator=generator)[:self.bank_size]
        token_ids = torch.tensor([candidates[i] for i in selected.tolist()], device=embedding_weight.device)
        initial_bank = embedding_weight.detach().index_select(0, token_ids).clone().float()
        initial_bank.mul_(self.bank_norm_scale)
        self.soft_token_bank = nn.Parameter(initial_bank.to(dtype=torch.bfloat16))
        self.register_buffer("initial_token_ids", token_ids.detach().cpu(), persistent=True)

        self.last_query = None
        self.last_attention = None
        self.last_output = None

    def forward(self, hidden_state):
        query = self.query(hidden_state)
        logits = F.linear(query.float(), self.soft_token_bank.float())
        logits = logits / (math.sqrt(self.hidden_size) * self.temperature)
        attention = torch.softmax(logits, dim=-1)
        output = torch.matmul(attention, self.soft_token_bank.float()).to(hidden_state.dtype)

        self.last_query = query.detach()
        self.last_attention = attention.detach()
        self.last_output = output.detach()
        return output

    def detached_stats(self):
        if self.last_attention is None:
            return {}
        attention = self.last_attention.float().reshape(-1)
        entropy = -(attention * attention.clamp_min(1e-12).log()).sum()
        return {
            "query_norm": float(self.last_query.float().norm().item()),
            "bank_norm": float(self.soft_token_bank.detach().float().norm(dim=-1).mean().item()),
            "output_norm": float(self.last_output.float().norm().item()),
            "attention_entropy": float(entropy.item()),
            "attention_effective_tokens": float(entropy.exp().item()),
            "attention_top1": float(attention.max().item()),
            "attention_argmax": int(attention.argmax().item()),
        }
