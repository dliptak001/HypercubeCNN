# HCNN Application Space

## What HCNN Actually Is

A convolutional network on binary hypercubes. Input: a binary/bipolar vector of length 2^DIM mapped onto hypercube vertices. Convolution: each vertex aggregates its Hamming-distance-1 neighbors via XOR masks. Pooling: antipodal (bitwise complement) pairs.

**The geometry is set-based, not spatial.** There is no left/right/up/down. Nearby vertices differ by a single bit flip. The convolution learns which single-bit-flip neighbors are informative.

---

## Requirements for Target Application

1. **No spatial relationship between features.** The input is a set of binary states, not a grid or sequence. Features have no inherent ordering or adjacency.

2. **Real-world problem.** Not a toy benchmark — something people actually need solved.

3. **Publicly available datasets.** Reproducible results, fair comparison to baselines.

4. **Well-understood relationships.** The mapping from pattern to class should be articulable by domain experts. We need to be able to explain *why* the model works or doesn't, not just report a number.

5. **Balanced classes** (or at least not extreme imbalance) — no fighting the data before testing the model.

6. **Established baselines with published numbers** — know immediately if we're competitive.

7. **Reasonable dataset size** — big enough to train, small enough to iterate fast.

---

## What Fits Naturally

The ideal HCNN input is a **binary feature vector where each bit represents an independent yes/no property**, and classification depends on *combinations* of those properties rather than any single one.

### Candidate Domains

| Domain | Input | Spatial? | Public? | Understood? | Notes |
|--------|-------|----------|---------|-------------|-------|
| Android malware (permissions) | Binary permission flags | No | Yes | Yes | Each permission is independent yes/no |
| Ames mutagenicity (MACCS keys) | 166-bit structural keys | No | Yes | Yes | Each bit = independent substructure query |
| Chess endgame (kr-vs-kp) | 36 binary board properties | No | Yes | Yes | Logical predicates, no spatial ordering |
| Phishing website detection | 30 ternary site properties | No | Yes | Yes | Independent website checks |
| Congressional voting | 16 binary votes | No | Yes | Yes | Independent policy positions |
| Medical diagnosis (symptoms) | Binary symptom presence | No | Varies | Yes | Independent clinical observations |

---

## The First-Order Filter

**Will anybody care if we meet or beat existing benchmarks?**

Any domain where the input is a fixed binary feature vector has likely been mined to saturation by RF/XGBoost/SVM. Those methods are near-optimal for tabular binary data. A new architecture matching them is a "so what?" result.

HCNN needs a domain where its **hierarchical interaction learning** provides a genuine advantage over flat ensemble methods. Trees of depth d capture at most order-d interactions. For k-way pure interactions (XOR/parity-like), trees need O(2^k) leaves. HCNN's stacked conv+pool layers build k-order interactions from (k-1)-order interactions — a direct architectural match for high-order combinatorial structure.

**The question: which domains have important higher-order feature interactions that RF/XGBoost miss?**

| Domain | Interaction Order | Trees Fail? | People Care? | Evidence |
|--------|------------------|-------------|-------------|----------|
| **Epistasis (genetics)** | 2-5+ way gene interactions | Yes — match Lasso on pure epistasis | Yes — GWAS, precision medicine | Frontiers 2024, ADNI Alzheimer's |
| **Cipher distinguishing** | Full-depth (by design) | Catastrophically | Niche (cryptographers) | Gohr CRYPTO 2019 |
| **Software config spaces** | 3-5 way flag interactions | Yes — published plateau | Moderate (SE research) | Siegmund 2015, Ha & Zhang 2019 |
| **CTR prediction** | High-order, sparse | Need 100+ separate models | Yes — industry standard | Google DCN-V2, BARS benchmark |
| Android permissions | Low (RF gets 97%+) | No | No — problem already solved | N/A |
| Molecular fingerprints | Low-moderate | No — RF matches NNs | Moderate | MoleculeNet benchmarks |

---

## What Doesn't Fit

- **Images** — strong spatial structure (HCNN throws it away)
- **Sequences** (text, DNA) — positional ordering matters
- **Graphs** — topology is irregular, not a hypercube
- **Tabular with continuous features** — binarization loses information
- **Any domain where RF/XGBoost already saturates** — no room for HCNN to add value

---

## Evaluation Criteria for Candidate Domains

- [ ] Binary feature representation is natural (not forced)
- [ ] Competitive baselines exist and are published
- [ ] Dataset size is tractable (not 41K samples with 3% signal)
- [ ] Class balance is reasonable (or well-handled)
- [ ] HCNN's neighbor-aggregation has a plausible mechanism of action
- [ ] Results are interpretable to domain experts

---

## Candidates — Detailed

### 1. Epistasis Detection (SNP Genotype Data) *** PRIORITY ***

- **What**: Predict disease phenotype from binary SNP genotype data, where the signal is gene-gene interactions (epistasis), not individual gene effects
- **Why HCNN wins**: Pure epistasis has zero marginal effects — no single feature predicts anything. Trees get zero information gain on the first split and need O(2^k) leaves for k-way interactions. HCNN's stacked conv+pool builds k-order interactions hierarchically.
- **Evidence**: Frontiers in Medicine 2024 — Type 1 Diabetes: NNs AUC 0.82, gradient boosting 0.77. Alzheimer's ADNI: accuracy scales with interaction order (2-way 0.67 → 5-way 0.87). Trees can't capture this.
- **Data is natively binary**: SNPs are biallelic (0/1)
- **People care**: GWAS, precision medicine, drug target discovery

#### Datasets

**A. GAMETES (synthetic, controlled)** — recommended starting point
- **Source**: https://github.com/UrbsLab/GAMETES + pre-generated at https://github.com/EpistasisLab/scikit-rebate
- **Size**: Configurable (typically 1,600-10,000 samples)
- **Features**: 20-100 SNPs (binary 0/1), with 2-5 causal SNPs embedded
- **Interaction order**: Configurable (2-locus, 3-locus, etc.), pure/strict models
- **Classes**: 2 (case/control), balanced by design
- **Baselines**: RF fails on pure XOR epistasis; MDR and specialized methods partially recover
- **DIM fit**: 20 features → DIM=5 (32 vertices); 100 features → DIM=7 (128 vertices)
- **Advantage**: Known ground truth, controlled heritability, can benchmark by interaction order

**B. ADNI (Alzheimer's, real data)**
- **Source**: https://adni.loni.usc.edu/ (requires approved access)
- **Size**: ~2,000 subjects, ~826K SNPs (filtered to ~1,000-10,000 by LD pruning)
- **Classes**: 2 (Alzheimer's vs control) or 3 (AD/MCI/control)
- **Known epistasis**: APOE x TOMM40 and higher-order interactions published
- **Baselines**: DNN with SHAP: 0.86-0.87 accuracy (5-way), trees plateau at ~0.67-0.72

**C. UK Biobank (large-scale, real data)**
- **Source**: https://www.ukbiobank.ac.uk/ (requires approved access)
- **Size**: 488K individuals, ~826K SNPs
- **Multiple phenotypes**: Type 1 Diabetes (strong epistasis), Type 2, obesity, psoriasis
- **Binary PLINK format**: directly usable

### 2. Cipher Distinguishing (Gohr 2019) *** PRIORITY ***

- **What**: Given 64 binary bits, determine if they're output of Speck32/64 cipher or random
- **Source**: Gohr, CRYPTO 2019 — "Improving Attacks on Round-Reduced Speck32/64 Using Deep Learning"
- **Size**: Unlimited (generate ciphertext pairs on demand)
- **Features**: 64 binary bits
- **Classes**: 2 (cipher output vs random)
- **Balance**: Perfect 50/50 by construction
- **Baselines**: No classical distinguisher achieves this. Gohr's NN: ~62% accuracy on 5-round Speck. Trees: ~50% (random chance).
- **DIM fit**: 64 features → DIM=6 (64 vertices) — PERFECT 1:1 mapping
- **Why HCNN wins**: Ciphers are designed to maximize nonlinear bit interactions. The function is specifically constructed to defeat any low-order approximation. HCNN's XOR-based convolution mirrors the cipher's own XOR-based mixing.
- **Why it matters**: Published at CRYPTO (top venue). A novel architecture matching/beating Gohr's result would be immediately noticed by the cryptanalysis community.
- **Caveat**: Niche audience. Less real-world impact than genomics.

### 3. Android Malware Detection (Permissions) — DEPRIORITIZED

#### Dataset Options (best to worst)

**A. MH-100K** (recommended)
- **Source**: https://github.com/Malware-Hunter/MH-100K-dataset
- **Size**: 101,975 samples
- **Features**: 166 binary (Android permissions, each 0/1)
- **Classes**: 2 (malware vs benign)
- **Balance**: TBD — need to verify
- **Provenance**: VirusTotal-verified labels, SHA256 hashes included, 2023
- **DIM fit**: 166 features → DIM=8 (256 vertices) — same as Ames

**B. TUANDROMD** (good fallback)
- **Source**: https://archive.ics.uci.edu/dataset/855/tuandromd
- **Size**: 4,464 samples
- **Features**: 241 binary (214 permissions + 27 API calls)
- **Classes**: 2 (malware vs benign)
- **Balance**: TBD
- **Provenance**: Clean, no missing values, well-documented at UCI, 2020/2023
- **DIM fit**: 241 features → DIM=8 (256 vertices)

**C. Kaggle/Drebin-215** (NOT recommended)
- **Source**: https://www.kaggle.com/datasets/shashwatwork/android-malware-dataset-for-machine-learning
- **Size**: 15,036 samples (but only 7,261 unique — 51.7% duplicates)
- **Features**: 215 binary (mix: ~115 permissions, ~47 API calls, ~23 intents, ~30 system commands)
- **Classes**: 2 — S=malware (5,560, 37%) / B=benign (9,476, 63%)
- **PROBLEMS**: Massive duplicates, 3 label conflicts (same feature vector mapped to both classes), outdated Drebin malware from 2010-2012, no app identifiers, published accuracy numbers are inflated by duplicate leakage
- **DIM fit**: 215 features → DIM=8 (256 vertices)

#### Why this domain works
Each Android permission is independently granted or not. "Can this app send SMS?" is yes/no. No ordering among permissions. Classification depends on which *combination* of permissions an app requests — exactly what HCNN's neighbor-aggregation detects.

- **Baselines**: RF/SVM/XGBoost 96-99% on various datasets (but inflated on Drebin-215 due to duplicates)

### 4. Ames Mutagenicity (MACCS 166-bit Fingerprint) — DEPRIORITIZED

- **Source**: Hansen et al. 2009, available via rfFC R package and Figshare
- **Size**: 6,512 molecules
- **Features**: 166 binary (MACCS structural keys — each bit = presence of a specific chemical substructure)
- **Classes**: 2 (mutagen: 3,053 / non-mutagen: 3,009)
- **Balance**: Nearly perfect 50/50
- **Baselines**: RF ~85%, SVM ~85%, deep learning ~88%
- **DIM fit**: 166 features → DIM=8 (256 vertices), embed 166 into 256
- **Why it works**: Each MACCS key asks "does this molecule contain substructure X?" — pure binary, unordered. Mutagenicity depends on combinations of structural features. Regulatory toxicology — real-world importance. We already have MoleculeNet pipeline infrastructure.

### 3. Chess King-Rook vs King-Pawn (kr-vs-kp)

- **Source**: https://archive.ics.uci.edu/dataset/22/chess+king+rook+vs+king+pawn
- **Size**: 3,196 instances
- **Features**: 36 binary (board position properties)
- **Classes**: 2 (won: 52% / nowin: 48%)
- **Balance**: Nearly perfect
- **Baselines**: Decision tree ~99%, RF ~99%, neural nets ~85-99%
- **DIM fit**: 36 features → DIM=6 (64 vertices), embed 36 into 64
- **Why it works**: Each feature is a logical predicate about the position ("is the rook attacking the king?"). No spatial ordering among features. Chess endgame theorists can explain every feature. Non-trivial boundary (not 100% solvable by single features).

### 4. Phishing Website Detection

- **Source**: https://archive.ics.uci.edu/ml/datasets/phishing+websites
- **Size**: 11,055 instances
- **Features**: 30 ternary (-1/0/+1) → 60 binary after splitting
- **Classes**: 2 (phishing 44% / legitimate 56%)
- **Balance**: Good
- **Baselines**: RF ~98%, SVM ~97%, gradient boosting ~97%
- **DIM fit**: 30 features → DIM=5 (32 vertices) or 60 binary → DIM=6 (64)
- **Why it works**: Each feature tests an independent website property ("URL contains IP address?", "SSL valid?"). No spatial relationship between checks. Cybersecurity — real-world importance.

### 5. Congressional Voting Records (1984)

- **Source**: https://archive.ics.uci.edu/dataset/105/congressional+voting+records
- **Size**: 435 instances
- **Features**: 16 binary (yes/no votes)
- **Classes**: 2 (Democrat 61% / Republican 39%)
- **Balance**: Moderate
- **Baselines**: Logistic regression ~95%, RF ~96%
- **DIM fit**: 16 features → DIM=4 (16 vertices) — perfect 1:1 mapping
- **Why it works**: Each vote is independent. Political scientists can explain every feature. Quick sanity check — too small for serious benchmarking but instant validation.

### 6. SPECT Heart (Cardiac Diagnosis)

- **Source**: https://archive.ics.uci.edu/dataset/95/spect+heart
- **Size**: 267 patients
- **Features**: 22 binary (thresholded cardiac perfusion measures)
- **Classes**: 2 (normal 21% / abnormal 79%)
- **Balance**: Imbalanced — violates requirement 5
- **Baselines**: CLIP3 ~84%, various ML ~75-85%
- **DIM fit**: 22 features → DIM=5 (32 vertices)
- **Why it works**: Each feature is "is perfusion in heart segment X normal?" — binary, unordered. Medical diagnosis — real-world. But too small and imbalanced.

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-04-06 | Reset application search | MoleculeNet HIV proved that fingerprint+RF is extremely hard to beat; need a domain where hypercube geometry is a natural fit, not just an alternative encoding |
| 2026-04-06 | Primary targets selected | Android Permissions and Ames Mutagenicity — later deprioritized |
| 2026-04-06 | First-order filter applied | "Will anybody care?" — domains where RF already saturates fail this filter. Need domains with high-order interactions trees can't capture |
| 2026-04-06 | New primary targets | Epistasis detection (genomics) and cipher distinguishing (cryptanalysis). Both have published evidence that trees fail and NNs win due to interaction complexity |
