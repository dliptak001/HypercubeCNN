#pragma once
// Speck32/64 block cipher — minimal header-only implementation.
// Block: 32 bits (two 16-bit words). Key: 64 bits (four 16-bit words). Rounds: 22.
// Reference: "The Simon and Speck Families of Lightweight Block Ciphers" (Beaulieu et al., 2013)

#include <cstdint>

namespace speck {

static constexpr int FULL_ROUNDS = 22;

inline uint16_t ror(uint16_t x, int r) { return (x >> r) | (x << (16 - r)); }
inline uint16_t rol(uint16_t x, int r) { return (x << r) | (x >> (16 - r)); }

// Expand 64-bit key into round keys.
inline void key_schedule(const uint16_t master[4], uint16_t rk[FULL_ROUNDS]) {
    uint16_t l[FULL_ROUNDS + 3]; // l buffer (indices 0..FULL_ROUNDS+1)
    l[0] = master[1];
    l[1] = master[2];
    l[2] = master[3];
    rk[0] = master[0];

    for (int i = 0; i < FULL_ROUNDS - 1; ++i) {
        l[i + 3] = (rk[i] + ror(l[i], 7)) ^ static_cast<uint16_t>(i);
        rk[i + 1] = rol(rk[i], 2) ^ l[i + 3];
    }
}

// Encrypt one 32-bit block (two 16-bit words) for `rounds` rounds.
inline void encrypt(uint16_t& x, uint16_t& y,
                    const uint16_t rk[], int rounds) {
    for (int i = 0; i < rounds; ++i) {
        x = (ror(x, 7) + y) ^ rk[i];
        y = rol(y, 2) ^ x;
    }
}

// Convenience: encrypt a 32-bit block from/to uint32_t.
inline uint32_t encrypt_block(uint32_t plaintext,
                              const uint16_t rk[], int rounds) {
    uint16_t x = static_cast<uint16_t>(plaintext >> 16);
    uint16_t y = static_cast<uint16_t>(plaintext & 0xFFFF);
    encrypt(x, y, rk, rounds);
    return (static_cast<uint32_t>(x) << 16) | y;
}

} // namespace speck
