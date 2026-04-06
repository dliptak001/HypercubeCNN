#include "MoleculeNetDataset.h"
#include <cstring>
#include <fstream>
#include <stdexcept>

MoleculeNetDataset load_hcfp(const std::string& path, const std::string& name) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    // Read header
    char magic[4];
    uint32_t num_samples, num_bits, num_tasks;
    in.read(magic, 4);
    in.read(reinterpret_cast<char*>(&num_samples), 4);
    in.read(reinterpret_cast<char*>(&num_bits), 4);
    in.read(reinterpret_cast<char*>(&num_tasks), 4);

    if (std::memcmp(magic, "HCFP", 4) != 0) {
        throw std::runtime_error("Invalid .hcfp file: bad magic");
    }

    MoleculeNetDataset ds;
    ds.num_bits = static_cast<int>(num_bits);
    ds.num_tasks = static_cast<int>(num_tasks);
    ds.name = name;

    // Read samples
    std::vector<uint8_t> buf(num_bits + num_tasks + 1);

    for (uint32_t i = 0; i < num_samples; ++i) {
        in.read(reinterpret_cast<char*>(buf.data()), buf.size());
        if (!in) {
            throw std::runtime_error("Truncated .hcfp file at sample " +
                                     std::to_string(i));
        }

        MolSample s;
        s.fingerprint.resize(num_bits);
        for (uint32_t b = 0; b < num_bits; ++b) {
            s.fingerprint[b] = (buf[b] == 0) ? -1.0f : 1.0f;
        }

        uint8_t label = buf[num_bits];
        s.label = (label == 255) ? -1 : static_cast<int>(label);

        uint8_t split = buf[num_bits + num_tasks];

        if (split == 0) ds.train.push_back(std::move(s));
        else if (split == 1) ds.val.push_back(std::move(s));
        else ds.test.push_back(std::move(s));
    }

    return ds;
}
