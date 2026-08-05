"""Ultra-fast smoke for cibuildwheel (seconds, not training loops).

Keep this file tiny: it runs once per Python × platform matrix cell.
Full coverage lives in test_basic.py (local / non-wheel CI).
"""

import numpy as np

import hypercube_cnn as hc


def test_import_version():
    assert isinstance(hc.__version__, str)
    assert len(hc.__version__) > 0


def test_construct_predict_one_step():
    net = hc.HCNNConfig(
        dim=5,
        num_outputs=2,
        num_threads=1,
        layers=[hc.LayerSpec.conv(4), hc.LayerSpec.conv(4)],
        weight_seed=1,
    ).build()
    x = np.zeros(net.N, dtype=np.float32)
    x[0] = 1.0
    out = net.predict(x)
    assert out.shape == (2,)
    assert out.dtype == np.float32
    net.train_step(x, target=0, params=hc.TrainParams(learning_rate=1e-2))
    # Touch optional products so a broken spatial/helpers link fails the wheel.
    emb = hc.SpatialEmbedder(
        dim=5, mode=hc.SpatialEmbedMode.PadLow, pad_value=-1.0
    )
    assert emb.embed(np.ones((2, 2), dtype=np.float32)).shape == (emb.N,)
    assert abs(hc.cosine_lr(1e-3, 1e-4, 0, 5) - 1e-3) < 1e-6
