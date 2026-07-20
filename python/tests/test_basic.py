"""Lean functional tests for hypercube_cnn (local / full pytest).

Kept deliberately short for developer iteration. Do **not** add multi-epoch
accuracy races or large DIMs — those belong in examples/, not the test suite.
cibuildwheel uses test_wheel.py only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import hypercube_cnn as hc


@pytest.fixture
def tiny_cls(tmp_path):
    """DIM=5 classification net, randomized once."""
    net = hc.HCNNConfig(
        dim=5,
        num_outputs=3,
        num_threads=1,
        layers=[
            hc.LayerSpec.conv(4, bn=False),
            hc.LayerSpec.pool("max"),
            hc.LayerSpec.conv(4),
        ],
        weight_seed=3,
    ).build()
    return net, tmp_path


class TestConstruction:
    def test_dim_and_capacity(self):
        net = hc.HCNN(dim=5, num_outputs=2, num_threads=1)
        assert net.dim == 5
        assert net.N == 32
        assert net.capacity == 32
        assert not net.weights_initialized

    def test_invalid_dim(self):
        with pytest.raises(ValueError, match="dim must be"):
            hc.HCNN(dim=2)
        with pytest.raises(ValueError, match="dim must be"):
            hc.HCNN(dim=31)

    def test_uninitialized_predict(self):
        net = hc.HCNN(dim=5, num_outputs=2, num_threads=1)
        net.add_conv(4)
        with pytest.raises(RuntimeError):
            net.predict(np.zeros(32, dtype=np.float32))


class TestInferTrain:
    def test_predict_shapes(self, tiny_cls):
        net, _ = tiny_cls
        x = np.random.default_rng(0).standard_normal(net.N).astype(np.float32)
        logits = net.predict(x)
        assert logits.shape == (net.num_outputs,)
        assert isinstance(net.predict_class(x), int)
        batch = np.stack([x, x], axis=0)
        out = net.forward_batch(batch)
        assert out.shape == (2, net.num_outputs)

    def test_train_step_and_epoch_once(self, tiny_cls):
        net, _ = tiny_cls
        rng = np.random.default_rng(1)
        B = 16
        X = rng.standard_normal((B, net.N), dtype=np.float32)
        y = rng.integers(0, net.num_outputs, size=B, dtype=np.int32)
        net.train_step(X[0], target=int(y[0]), params=hc.TrainParams(learning_rate=1e-2))
        net.train_epoch(X, y, batch_size=8, params=hc.TrainParams(learning_rate=1e-2, shuffle_seed=1))

    def test_regression_step(self):
        net = hc.HCNNConfig(
            dim=5,
            num_outputs=2,
            task=hc.TaskType.Regression,
            num_threads=1,
            layers=[hc.LayerSpec.conv(4)],
            weight_seed=2,
        ).build()
        x = np.zeros(net.N, dtype=np.float32)
        t = np.array([0.1, -0.2], dtype=np.float32)
        net.train_step(x, target=t, params=hc.TrainParams(learning_rate=1e-2))
        pred = net.predict(x)
        assert pred.shape == (2,)


class TestArchAndWeights:
    def test_summarize_matches_weight_count(self, tiny_cls):
        net, _ = tiny_cls
        s = hc.summarize_arch(net.dim, net.num_outputs, net.input_channels, net.layers)
        assert s.total == net.weight_count

    def test_export_from_arch_set_weights(self, tiny_cls):
        net, _ = tiny_cls
        x = np.random.default_rng(4).standard_normal(net.N).astype(np.float32)
        logits0 = net.predict(x)
        arch = net.export_arch()
        # JSON round-trip
        arch = json.loads(json.dumps(arch))
        w = net.get_weights()
        net2 = hc.HCNN.from_arch(arch, num_threads=1, weight_seed=99)
        net2.set_weights(w)
        np.testing.assert_array_equal(logits0, net2.predict(x))

    def test_future_arch_version_rejected(self, tiny_cls):
        net, _ = tiny_cls
        arch = net.export_arch()
        arch["version"] = 999
        with pytest.raises(ValueError, match="Upgrade hypercube-cnn"):
            hc.HCNN.from_arch(arch)

    def test_hcnw_save_load(self, tiny_cls):
        net, tmp = tiny_cls
        x = np.random.default_rng(5).standard_normal(net.N).astype(np.float32)
        logits0 = net.predict(x)
        stem = tmp / "m"
        net.save(stem)
        assert (tmp / "m.hcnw").is_file()
        assert (tmp / "m.arch.json").is_file()
        net2 = hc.HCNN.load(stem, num_threads=1)
        np.testing.assert_array_equal(logits0, net2.predict(x))

    def test_hcnw_arch_mismatch(self, tiny_cls):
        net, tmp = tiny_cls
        path = tmp / "w.hcnw"
        net.save_weights(path)
        other = hc.HCNNConfig(
            dim=5,
            num_outputs=3,
            num_threads=1,
            layers=[hc.LayerSpec.conv(4), hc.LayerSpec.conv(4)],
            weight_seed=1,
        ).build()
        with pytest.raises(RuntimeError, match="mismatch"):
            other.load_weights(path)
