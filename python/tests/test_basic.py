"""Lean functional tests for hypercube_cnn (local / full pytest).

Kept deliberately short for developer iteration. Do **not** add multi-epoch
accuracy races or large DIMs — those belong in examples/, not the test suite.
cibuildwheel uses test_wheel.py only.
"""

from __future__ import annotations

import json
import pickle
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


class TestSessionDefaults:
    def test_set_train_defaults_honored_when_params_none(self):
        """Zero LR via set_train_defaults must freeze weights on train_step."""
        net = hc.HCNNConfig(
            dim=5,
            num_outputs=2,
            num_threads=1,
            layers=[hc.LayerSpec.conv(4)],
            weight_seed=1,
        ).build()
        x = np.zeros(net.N, dtype=np.float32)
        x[0] = 1.0
        w0 = net.get_weights().copy()
        net.set_train_defaults(hc.TrainParams(learning_rate=0.0))
        net.train_step(x, target=0, params=None)
        np.testing.assert_array_equal(w0, net.get_weights())


class TestWeightSeed64:
    """Full uint64 weight seeds: no silent low-32 truncation (matches CoreSmokeTest)."""

    _WIDE = 7934791766227647176  # high half nonzero
    _LOW = _WIDE & 0xFFFFFFFF  # 816937672

    @staticmethod
    def _weights(seed) -> np.ndarray:
        net = hc.HCNN(dim=5, num_outputs=2, num_threads=1)
        net.add_conv(4)
        net.randomize_weights(scale=0.0, seed=seed)
        return net.get_weights()

    def test_small_seed_bit_identical(self):
        w_a = self._weights(3)
        w_b = self._weights(3)
        np.testing.assert_array_equal(w_a, w_b)

    def test_wide_seed_differs_from_low32_truncation(self):
        w_wide = self._weights(self._WIDE)
        w_trunc = self._weights(self._LOW)
        assert w_wide.shape == w_trunc.shape
        assert not np.array_equal(w_wide, w_trunc)

    def test_numpy_uint64_accepted(self):
        w_py = self._weights(self._WIDE)
        w_np = self._weights(np.uint64(self._WIDE))
        np.testing.assert_array_equal(w_py, w_np)

    def test_config_build_wide_seed(self):
        net = hc.HCNNConfig(
            dim=5,
            num_outputs=2,
            num_threads=1,
            layers=[hc.LayerSpec.conv(4)],
            weight_seed=self._WIDE,
        ).build()
        w_cfg = net.get_weights()
        np.testing.assert_array_equal(w_cfg, self._weights(self._WIDE))

    def test_rejects_negative_and_overflow(self):
        net = hc.HCNN(dim=5, num_outputs=2, num_threads=1)
        net.add_conv(4)
        with pytest.raises(ValueError, match=r"\[0, 2\*\*64-1\]"):
            net.randomize_weights(seed=-1)
        with pytest.raises(ValueError, match=r"\[0, 2\*\*64-1\]"):
            net.randomize_weights(seed=(1 << 64))
        with pytest.raises(TypeError, match="integer"):
            net.randomize_weights(seed=1.5)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="bool"):
            net.randomize_weights(seed=True)  # type: ignore[arg-type]


class TestArchAndWeights:
    def test_summarize_matches_weight_count(self, tiny_cls):
        net, _ = tiny_cls
        s = hc.summarize_arch(net.dim, net.num_outputs, net.input_channels, net.layers)
        assert s.total == net.weight_count

    def test_apply_layers_incremental_summary(self):
        """apply_layers on a non-empty body must summarize the full stack."""
        net = hc.HCNN(dim=5, num_outputs=2, num_threads=1)
        net.add_conv(4)
        net.add_pool()
        summary = net.apply_layers([hc.LayerSpec.conv(8)])
        net.randomize_weights(seed=1)
        assert summary.total == net.weight_count
        full = hc.summarize_arch(
            net.dim, net.num_outputs, net.input_channels, net.layers
        )
        assert summary.total == full.total

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

    def test_pickle_roundtrip(self, tiny_cls):
        net, _ = tiny_cls
        x = np.random.default_rng(6).standard_normal(net.N).astype(np.float32)
        logits0 = net.predict(x)
        loaded = pickle.loads(pickle.dumps(net))
        np.testing.assert_array_equal(logits0, loaded.predict(x))
        assert loaded.dim == net.dim
        assert loaded.num_conv == net.num_conv

    def test_load_cpp_written_hcnw_fixture(self):
        """C++ save_weights → Python load; logits match C++ Predict (gap 2)."""
        root = Path(__file__).resolve().parents[2] / "tests" / "data" / "hcnw_interop"
        hcnw = root / "model.hcnw"
        arch = root / "model.arch.json"
        assert hcnw.is_file() and arch.is_file(), f"missing fixture under {root}"
        x = np.loadtxt(root / "input.txt", dtype=np.float32)
        expected = np.loadtxt(root / "expected_logits.txt", dtype=np.float32)
        net = hc.HCNN.load(root / "model", num_threads=1)
        got = net.predict(x)
        # Same weight blob + same forward; allow tiny float noise across toolchains.
        np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-5)


class TestMetrics:
    def test_cosine_lr_endpoints(self):
        assert hc.cosine_lr(1e-3, 1e-4, 0, 10) == pytest.approx(1e-3)
        assert hc.cosine_lr(1e-3, 1e-4, 9, 10) == pytest.approx(1e-4)

    def test_evaluate_classification(self, tiny_cls):
        net, _ = tiny_cls
        rng = np.random.default_rng(7)
        B = 8
        X = rng.standard_normal((B, net.N), dtype=np.float32)
        y = rng.integers(0, net.num_outputs, size=B, dtype=np.int32)
        r = hc.evaluate_classification(net, X, y)
        assert r.count == B
        assert 0.0 <= r.accuracy <= 100.0
        assert r.correct >= 0

    def test_evaluate_regression(self):
        net = hc.HCNNConfig(
            dim=5,
            num_outputs=2,
            task=hc.TaskType.Regression,
            num_threads=1,
            layers=[hc.LayerSpec.conv(4)],
            weight_seed=2,
        ).build()
        X = np.zeros((4, net.N), dtype=np.float32)
        T = np.zeros((4, 2), dtype=np.float32)
        r = hc.evaluate_regression(net, X, T)
        assert r.count == 4
        assert r.mse >= 0.0


class TestSpatial:
    def test_pad_low_preserves_pad_value(self):
        emb = hc.SpatialEmbedder(
            dim=6, mode=hc.SpatialEmbedMode.PadLow, pad_value=-1.0
        )
        assert emb.capacity == 64
        img = np.ones((4, 4), dtype=np.float32)
        out = emb.embed(img)
        assert out.shape == (64,)
        np.testing.assert_array_equal(out[:16], 1.0)
        np.testing.assert_array_equal(out[16:], -1.0)

    def test_pad_low_center_mnist_dim10(self):
        emb = hc.SpatialEmbedder(
            dim=10, mode=hc.SpatialEmbedMode.PadLowCenter, pad_value=-1.0
        )
        assert emb.capacity == 1024
        plan = emb.plan(28, 28)
        assert plan.crop_h == 15 and plan.crop_w == 16
        assert plan.crop_row0 == 6 and plan.crop_col0 == 6
        assert plan.pattern_length == 1024
        img = np.arange(28 * 28, dtype=np.float32).reshape(28, 28)
        out = emb.embed(img)
        assert out.shape == (1024,)
        np.testing.assert_array_equal(out[:784], img.reshape(-1))
        crop = img[6 : 6 + 15, 6 : 6 + 16].reshape(-1)
        np.testing.assert_array_equal(out[784:], crop)

    def test_embed_batch_and_aug_identity(self):
        emb = hc.SpatialEmbedder(
            dim=6, mode=hc.SpatialEmbedMode.ResizeToFit, pad_value=-1.0
        )
        imgs = np.random.default_rng(0).standard_normal((2, 8, 8)).astype(np.float32)
        packed = emb.embed_batch(imgs)
        assert packed.shape == (2, emb.N)
        # identity aug (defaults) should copy
        aug = hc.SpatialAugmenter(enabled=False)
        out = aug.apply(imgs[0], seed=1)
        np.testing.assert_array_equal(out, imgs[0])

    def test_dual_plane_fits(self):
        emb = hc.SpatialEmbedder(
            dim=9, mode=hc.SpatialEmbedMode.DualPlaneResize, pad_value=-1.0
        )
        # N=512, dual S=16, 2*S*S=512
        img = np.zeros((28, 28), dtype=np.float32)
        img[10:18, 10:18] = 1.0
        out = emb.embed(img)
        assert out.shape == (512,)
        plan = emb.plan(28, 28)
        assert plan.pattern_length == 2 * plan.plane_side * plan.plane_side
