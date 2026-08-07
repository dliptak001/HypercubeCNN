#pragma once

// MNIST data location for demos. Fixed local deploy root only:
//   C:\HypercubeCNN\data
// (same tree as a typical local MNISTTrain.exe deploy). Never walks the
// CLion / source-tree clone. Example-only helper.

#include <filesystem>
#include <stdexcept>
#include <string>

namespace hcnn_ex {

inline bool DirHasMnistIdx(const std::filesystem::path& dir)
{
    namespace fs = std::filesystem;
    return fs::is_directory(dir)
           && fs::exists(dir / "train-images-idx3-ubyte")
           && fs::exists(dir / "train-labels-idx1-ubyte")
           && fs::exists(dir / "t10k-images-idx3-ubyte")
           && fs::exists(dir / "t10k-labels-idx1-ubyte");
}

/// Resolve C:\HypercubeCNN\data. @p argv0 unused (kept for call-site stability).
inline std::filesystem::path FindMnistDataDir(const char* /*argv0*/)
{
    namespace fs = std::filesystem;
    const fs::path data = fs::path("C:/HypercubeCNN/data");

    if (DirHasMnistIdx(data))
        return fs::weakly_canonical(data);

    throw std::runtime_error(
        "Cannot find MNIST IDX files in C:\\HypercubeCNN\\data\n"
        "Need:\n"
        "  train-images-idx3-ubyte  train-labels-idx1-ubyte\n"
        "  t10k-images-idx3-ubyte   t10k-labels-idx1-ubyte\n"
        "See examples/mnist_train.md (Data loading)");
}

} // namespace hcnn_ex
