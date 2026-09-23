import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'buffer'))

from sts import SharedSoftTokenSelector, initialize_kmeans_bank


class SharedSoftTokenSelectorTest(unittest.TestCase):
    def test_kmeans_initialization_is_cached(self):
        class FakeKMeans:
            def __init__(self, dimension, clusters, **kwargs):
                self.clusters = clusters
                self.centroids = None

            def train(self, vectors):
                self.centroids = vectors[:self.clusters].copy()

        fake_faiss = types.SimpleNamespace(Kmeans=FakeKMeans)
        embeddings = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / 'bank.pt'
            with mock.patch.dict(sys.modules, {'faiss': fake_faiss}):
                first = initialize_kmeans_bank(
                    embeddings,
                    bank_size=2,
                    excluded_token_ids=[0],
                    cache_path=cache,
                    seed=42,
                    niter=3,
                    device='cpu',
                )
            second = initialize_kmeans_bank(
                torch.zeros_like(embeddings),
                bank_size=2,
                excluded_token_ids=[0],
                cache_path=cache,
                bank_norm_scale=2.0,
            )
            self.assertTrue(torch.equal(first, embeddings[1:3]))
            self.assertTrue(torch.equal(second, first * 2))

    def test_scaling_has_no_sqrt_hidden_size_factor(self):
        bank = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        selector = SharedSoftTokenSelector(2, 1, 2, 2.0, bank)
        with torch.no_grad():
            selector.queries[0].query.weight.copy_(torch.eye(2))

        output = selector(torch.tensor([1.0, 0.0], dtype=torch.bfloat16), 0)
        expected_attention = torch.softmax(torch.tensor([0.5, 0.0]), dim=0)
        self.assertTrue(torch.allclose(output.float(), expected_attention, atol=2e-3))

    def test_positions_share_bank_and_receive_gradients(self):
        torch.manual_seed(7)
        selector = SharedSoftTokenSelector(
            hidden_size=4,
            num_positions=2,
            bank_size=3,
            temperature=1.0,
            initial_bank=torch.randn(3, 4),
        )
        hidden = torch.randn(2, 4, dtype=torch.bfloat16)
        loss = selector(hidden[0], 0).float().sum() + selector(hidden[1], 1).float().sum()
        loss.backward()

        self.assertIsNotNone(selector.soft_token_bank.grad)
        self.assertIsNotNone(selector.queries[0].query.weight.grad)
        self.assertIsNotNone(selector.queries[1].query.weight.grad)
        self.assertFalse(torch.equal(
            selector.queries[0].query.weight,
            selector.queries[1].query.weight,
        ))

    def test_bank_diagnostics_for_orthogonal_bank(self):
        selector = SharedSoftTokenSelector(
            hidden_size=2,
            num_positions=2,
            bank_size=2,
            temperature=1.0,
            initial_bank=torch.eye(2),
        )
        diagnostics = selector.bank_diagnostics()
        self.assertAlmostEqual(diagnostics['effective_rank'], 2.0, places=5)
        self.assertAlmostEqual(diagnostics['cosine_mean'], 0.0, places=5)
        self.assertAlmostEqual(diagnostics['cosine_std'], 0.0, places=5)
        self.assertAlmostEqual(diagnostics['cosine_min'], 0.0, places=5)
        self.assertAlmostEqual(diagnostics['cosine_max'], 0.0, places=5)
        self.assertTrue(math.isfinite(diagnostics['effective_rank']))


if __name__ == '__main__':
    unittest.main()
