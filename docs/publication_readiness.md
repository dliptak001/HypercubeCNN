# Publication Readiness Assessment

Honest assessment of HypercubeCNN's potential interest to the ML community.
Original assessment 2026-04-04; MNIST numbers refreshed 2026-04-09.

## What's Strong

The architectural combination is genuinely novel — the prior art survey confirmed nobody has assembled hypercube substrate + distance-indexed kernels + antipodal pooling + XOR enumeration into a CNN. The vertex-transitive symmetry giving principled weight sharing is a clean theoretical story. The connection to group convolution on Z_2^n is well-understood but nobody has instantiated it this way.

The C++ implementation is also genuinely clean: a single-class SDK front door (`hcnn::HCNN`), allocation-free hot paths, RAII-safe threading, optional BN / Adam / LeakyReLU / FLATTEN readout, no external dependencies. That alone won't get a paper accepted, but it removes "the code is half-baked" as a reviewer escape hatch.

## What's Missing

### MNIST doesn't demonstrate anything

The ML community considers MNIST solved. A linear classifier gets ~92%. HypercubeCNN currently reaches **98.10%** on the full 60K/10K split (~200K parameters, 4 stages, SGD-momentum + cosine LR + L2 weight decay) — competitive with a well-tuned 2-layer MLP, still ~1% below standard 2D CNNs (~99.3%). That gap is roughly the cost of not encoding 2D locality, which is exactly the point of the experiment.

It's still not a *publication* result. Reviewers see "98% on MNIST" and ask "so what?" — the demonstration is "the architecture learns useful features without spatial inductive bias", which is necessary but not sufficient for a paper.

The fundamental problem: MNIST is 2D spatial data. Mapping it onto a hypercube via Direct Linear Assignment destroys the spatial locality that makes the problem easy. The architecture is handicapped by testing on data that's wrong for it.

### No domain where it wins

The pitch has to be "here's a problem class where HypercubeCNN beats existing approaches." That problem class hasn't been identified yet. Efficiency analysis shows it's 5-10x slower than CNNs at MNIST scale, and the accuracy is worse. The scaling argument ("it gets better at high DIM") is theoretical with no empirical backing.

### Missing baselines

A paper would need:
- GNN on the same hypercube graph (to prove distance-indexed kernels beat generic message passing)
- Standard CNN on the same data (to quantify the accuracy/speed tradeoff)
- MLP baseline (to show the hypercube structure matters at all)

## What Would Make It Publishable

### Find data that naturally lives on binary hypercubes

This is the unlock. Candidates:

- **Combinatorial feature interactions** — N binary features, model their interactions. Each sample IS a hypercube vertex. This is where the architecture is mathematically native.
- **Genomics** — binary allele states across loci. EHCube4P (from the prior art survey) already does this with generic GCN. Beat them with distance-indexed kernels.
- **Boolean function learning** — Abbe et al. (from the prior art survey) study this with MLPs. Show that HCNN's inductive bias (Hamming-distance isotropy) learns Boolean functions faster or with fewer parameters.
- **Drug interaction / molecular property prediction** — binary fingerprint vectors are standard in cheminformatics. They're literally hypercube vertices.

On any of these, if HCNN beats an MLP and a GNN, that's a workshop paper at minimum and potentially a full conference paper.

### Alternatively, prove something

A theoretical result showing tighter generalization bounds for distance-isotropic convolution on Z_2^n versus generic GNN message passing — that's publishable independent of experiments.

## Bottom Line

The architecture is a real contribution. The engineering is clean. But MNIST is the wrong benchmark, and "here's a novel architecture that gets 98% on MNIST" still won't get past reviewers — they will (correctly) point out that's MLP territory and ask why anyone should care. The work becomes interesting the moment it runs on data that belongs on a hypercube.
