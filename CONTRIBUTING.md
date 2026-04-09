# Contributing to HypercubeCNN

Thanks for your interest. HypercubeCNN is currently maintained by a single
author as a research project; contributions are welcome but please open an
issue to discuss substantial changes before opening a PR.

## Building from source

Requirements: a C++23 compiler (GCC 13+, Clang 17+, MSVC 2022+) and CMake 3.21+.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

The build pulls in zero external dependencies beyond the C++ standard library
and your platform's threading support.

### Useful build options

| Option | Default | Description |
|--------|---------|-------------|
| `HCNN_NATIVE_ARCH` | `ON`  | `-march=native -mtune=native` for the build host. Disable when building a binary that must run on a different CPU. |
| `HCNN_FAST_MATH`   | `ON`  | `-ffast-math`. Disable when downstream code depends on strict IEEE NaN/Inf semantics. |
| `HCNN_WERROR`      | `OFF` | Treat library warnings as errors. |
| `HCNN_BUILD_EXAMPLES` | `ON` (when standalone) | Build `HypercubeCNN`, `MNISTTrain`, and `CoreSmokeTest` executables. |

## Running the smoke test

```bash
ctest --test-dir build --output-on-failure
```

The smoke test (`tests/CoreSmokeTest.cpp`) covers every documented HCNN code
path: construction, forward / batch inference, single-sample and mini-batch
training, all readout types, all pool types, batch normalization, LeakyReLU,
Adam, FLATTEN + Adam, weight decay, class weights, embed truncation/padding,
batch_size validation, and BN training-mode preservation. Any new feature
should land with a corresponding assertion in this file (until a real unit
test framework is added).

## Coding style

- C++23. Standard-library-only — no external dependencies in the library.
- Public API is in `namespace hcnn` and uses **PascalCase** method names
  (matching the canonical SDK front door `HCNN`).
- Internal classes use **snake_case** method names (e.g.
  `HCNNNetwork::train_batch`).
- Channel-major activation layout: `data[c * N + v]` for channel `c`, vertex
  `v`. This convention is universal across the pipeline.
- No hidden allocations in hot paths. Persistent scratch on the network or
  caller-owned buffers, never per-call `std::vector` instantiation in
  forward/training methods.
- RAII for state-restoring guards (training-mode flags, thread-pool
  pointers, BN running-stats suppression).

## Pull request checklist

- [ ] Builds clean on at least Linux GCC + Linux Clang (CI will check
      macOS Clang and Windows MSVC).
- [ ] `ctest` passes.
- [ ] New public symbols have a class brief and per-method doc comments.
- [ ] No new external dependencies in `HypercubeCNNCore`.
- [ ] No commits include large binary blobs (datasets, model weights,
      build artifacts).
- [ ] Commit messages are imperative and explain *why*, not just *what*.

## Reporting bugs

Open a GitHub issue with:

1. A minimal reproducer (the smaller the better — synthetic data is fine).
2. Compiler + version + OS.
3. The exact `cmake` configure line you used.
4. Expected vs. observed behavior.

## License

By contributing, you agree that your contributions will be licensed under
the project's [Apache License 2.0](LICENSE).
