# Smoke: build sdist, install into a clean venv, run a minimal API check.
# Run from repo root with CLion MinGW + Ninja on PATH (see python/README.md).
# Usage: powershell -File python/scripts/sdist_smoke.ps1

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Root

$py = if (Get-Command python -ErrorAction SilentlyContinue) { "python" } else { "py -3" }

Write-Host "== build sdist =="
if (Test-Path dist) { Remove-Item dist -Recurse -Force }
& $py -m pip install build -q
& $py -m build --sdist
$sdist = Get-ChildItem dist\*.tar.gz | Select-Object -First 1
if (-not $sdist) { throw "no sdist produced" }
Write-Host "sdist: $($sdist.FullName)"

$venv = Join-Path $Root "_sdist_smoke_venv"
if (Test-Path $venv) { Remove-Item $venv -Recurse -Force }
& $py -m venv $venv
$vp = Join-Path $venv "Scripts\python.exe"

Write-Host "== install from sdist (build isolation) =="
& $vp -m pip install --upgrade pip -q
& $vp -m pip install $sdist.FullName

Write-Host "== import / predict / spatial smoke =="
& $vp -c @"
import numpy as np
import hypercube_cnn as hc
net = hc.HCNNConfig(dim=5, num_outputs=2, num_threads=1,
    layers=[hc.LayerSpec.conv(4)], weight_seed=1).build()
x = np.zeros(net.N, dtype=np.float32); x[0] = 1.0
assert net.predict(x).shape == (2,)
emb = hc.SpatialEmbedder(dim=6, mode=hc.SpatialEmbedMode.PadLow, pad_value=-1.0)
out = emb.embed(np.ones((4, 4), dtype=np.float32))
assert out.shape == (64,) and float(out[16]) == -1.0
print('sdist_smoke: OK', hc.__version__)
"@

Write-Host "PASS"
