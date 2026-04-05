# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

cmake/g++ are bundled with CLion and are **not** on the system PATH. Bash cannot capture g++ stderr — you **must** use the PowerShell heredoc pattern.

**Release build:**
```bash
powershell.exe -File - <<'PS1'
$cmake = 'C:\Program Files\JetBrains\CLion 2024.3.2\bin\cmake\win\x64\bin\cmake.exe'
$env:PATH = "C:\Program Files\JetBrains\CLion 2024.3.2\bin\mingw\bin;" + $env:PATH
& $cmake --build C:\CLion\HypercubeCNNStaging\cmake-build-release 2>&1
PS1
```

**Debug build:** Replace `cmake-build-release` with `cmake-build-debug`.

**Run executable** (needs MinGW on PATH for libgomp-1.dll):
```bash
powershell.exe -File - <<'PS1'
$env:PATH = "C:\Program Files\JetBrains\CLion 2024.3.2\bin\mingw\bin;" + $env:PATH
& "C:\CLion\HypercubeCNNStaging\cmake-build-release\HypercubeCNN.exe" 2>&1
PS1
```

**Never reconfigure cmake-build-\* directories** (no `cmake -B` with `-G` flags). CLion owns those. If broken, delete and reload CMake from CLion.

Prefer Release mode for tests and diagnostics (Debug has different float behavior with -ffast-math).

## Architecture

HypercubeCNN performs convolutions on binary hypercubes using Hamming distance instead of spatial grids. All geometry is bitwise — no adjacency lists, no padding.

### Core pipeline

`HCNNNetwork` orchestrates the full forward pass:

1. **Input embedding** — maps flat scalar arrays onto `N = 2^DIM` hypercube vertices (Direct Linear Assignment). Values must be in [-1.0, 1.0].
2. **Conv layers (`HCNN`)** — sparse-vertex convolution using K = 2*DIM - 2 fixed XOR masks per vertex. Each output vertex = weighted sum of K specific neighbors (DIM nearest-neighbor single-bit flips + DIM-2 cumulative shell masks from HypercubeRC Reservoir). Kernel shape: `[c_out * c_in * K]`.
3. **Pool layers (`HCNNPool`)** — reduces DIM by `reduce_by` via MAX or AVG over subcubes (bit-shift grouping).
4. **Readout (`HCNNReadout`)** — global average per channel → linear layer → class logits.

Memory layout is channel-major: `activations[c * N + v]`.

### Build targets

All executables link against `HypercubeCNNCore` (static library). Sources live in the library; executables are thin wrappers. This is intentional — the library is the future C++ SDK surface.

| CMake target | Purpose |
|---|---|
| `HypercubeCNNCore` | Static library with all core classes |
| `HypercubeCNN` | Demo executable (main.cpp) |

Diagnostic, example, and test targets (empty placeholders in `diagnostics/`, `examples/`, `tests/`) will follow the same pattern: link `HypercubeCNNCore`, never compile core sources directly.

### Key constraints

- **OpenMP stays out of the core library.** Only diagnostic/sweep targets may use it.
- **No external dependencies** in core. Everything is flat arrays and standard C++23.
- The `dataloader/` directory holds dataset implementations (currently a toy MNIST stub). Its include path is exported by the library.
