#pragma once

// Locate this repo's data/ directory on any machine (CLion, CLI, different cwd).
// Looks for MNIST IDX under candidate paths — never another project by name.
// Example-only helper (mirrors HypercubeWTF examples/common/find_data_dir.h).

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

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

/// Walk @p start and each parent, testing @p start/data and parent/data.
inline void PushAncestorsWithData(std::vector<std::filesystem::path>& out,
                                  std::filesystem::path start)
{
    namespace fs = std::filesystem;
    std::error_code ec;
    if (start.empty())
        return;
    fs::path p = fs::weakly_canonical(start, ec);
    if (ec)
        p = fs::absolute(start, ec);
    if (ec)
        p = start;

    for (int i = 0; i < 12 && !p.empty(); ++i)
    {
        out.push_back(p / "data");
        if (!p.has_parent_path() || p == p.root_path())
            break;
        const fs::path parent = p.parent_path();
        if (parent == p)
            break;
        p = parent;
    }
}

/// Resolve HypercubeCNN/data using argv[0] (exe) and cwd. Throws if not found.
inline std::filesystem::path FindMnistDataDir(const char* argv0)
{
    namespace fs = std::filesystem;
    std::vector<fs::path> candidates;

    // 1) Current working directory tree (CLion often sets cwd = project root)
    PushAncestorsWithData(candidates, fs::current_path());

    // 2) Executable location tree (build/ and install layouts)
    if (argv0 != nullptr && argv0[0] != '\0')
    {
        std::error_code ec;
        fs::path exe = fs::absolute(fs::path(argv0), ec);
        if (!ec)
            PushAncestorsWithData(candidates, exe.parent_path());
    }

    // 3) Source tree from this header (…/examples → repo root)
    {
        const fs::path from_header =
            fs::path(__FILE__).parent_path().parent_path() / "data";
        candidates.push_back(from_header);
    }

    for (const auto& c : candidates)
    {
        if (DirHasMnistIdx(c))
            return fs::weakly_canonical(c);
    }

    std::string msg =
        "Cannot find MNIST data/ (need the four *-ubyte IDX files).\n"
        "Tried candidates under cwd, executable path, and source tree.\n"
        "Place files in HypercubeCNN/data/ - see data/README.md\n"
        "Candidates:\n";
    for (const auto& c : candidates)
        msg += "  " + c.lexically_normal().make_preferred().string() + "\n";
    throw std::runtime_error(msg);
}

} // namespace hcnn_ex
