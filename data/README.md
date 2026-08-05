# MNIST data (HypercubeCNN)

`MNISTTrain` loads from **this repo’s** `data/` directory. It discovers that folder
from the process cwd, the executable path, or the source tree — not from
HypercubeWTF or any other project.

## Required files (uncompressed IDX)

```text
train-images-idx3-ubyte
train-labels-idx1-ubyte
t10k-images-idx3-ubyte
t10k-labels-idx1-ubyte
```

These are **not** committed (see root `.gitignore`). Populate locally once.

### Download (example)

From this `data/` directory:

```text
curl -L -O https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
curl -L -O https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
curl -L -O https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
curl -L -O https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz
gunzip *.gz
```

(On Windows you can use any tool that fetches and gunzips those four files into this folder.)

### Discovery order

`MNISTTrain` searches for a `data/` directory that contains all four IDX files under:

1. Current working directory and parents
2. Executable directory and parents (CLion build dirs)
3. Source tree next to `examples/`

No other project path. No env override. See `examples/find_data_dir.h`.
