# Boolean Functions: Lessons Learned

## The original hypothesis

Boolean functions f: {0,1}^n -> {0,1} were proposed as the ideal test case for HypercubeCNN. The reasoning: the input IS a hypercube vertex, there's no embedding distortion, and functions like parity have structure aligned with Hamming distance. If the architecture can't win here, it can't win anywhere.

**This hypothesis was wrong.** The encoding difficulty revealed a fundamental mismatch between the task and the architecture.

## What went wrong

### The task vs. the architecture

HypercubeCNN is a convolutional neural network. Like all CNNs, it processes a **field over a domain** — a value at every point in the space. A spatial CNN processes an image (a value at every pixel). HypercubeCNN processes an activation map (a value at every hypercube vertex).

Boolean function learning asks a different question: given a single **point** (vertex v), predict f(v). The input is a coordinate, not a field. This is the equivalent of asking a spatial CNN to predict the color of pixel (x, y) given just the coordinates — not a task CNNs are designed for. You'd use an MLP.

### Three failed encodings

Each encoding attempt tried to manufacture a field from a point, with escalating complexity:

1. **One-hot** (1 channel, 1.0 at vertex v, 0.0 elsewhere): Too sparse. Only 1 of 1024 vertices has signal. Conv layers had almost nothing to work with. Result: stuck at random chance (50%).

2. **Bipolar** (1 channel, +1 at vertex v, -1 elsewhere): Slightly better signal density, but conv+GAP is provably translation-invariant on the hypercube, and bipolar encodings of different vertices are related by XOR-translations. Mathematical result: `GAP(conv(input_a)) = GAP(conv(input_b))` for all vertices a, b. The network produced identical output for every input. Result: stuck at 50%.

3. **Bipolar + flatten readout**: Replaced GAP with flatten to preserve positional information. Learning occurred (50% -> 56% train) but severe overfitting (test dropped to 36%). The flatten readout had 4098 parameters for 716 training samples. The conv layers produced weak features from the sparse input.

4. **Bit-channel encoding** (DIM channels, channel k = +1 where bit k of u matches bit k of v): Dense, structured input that breaks translation symmetry. Not tested — we stopped here after recognizing the fundamental problem.

### Why the encoding struggle is the diagnosis

If a task requires increasingly elaborate input encoding to work with an architecture, the architecture is wrong for the task. Each encoding was a workaround, not the architecture operating on its native substrate. The bit-channel encoding in particular is essentially feeding the network a manufactured "image" that encodes positional information — the conv layers process this synthetic field, but the encoding does most of the work.

### The spatial CNN analogy

| Spatial CNN | HypercubeCNN |
|-------------|-------------|
| Image classification: input is a field (pixel values over a 2D grid). Natural fit. | Molecular fingerprint classification: input is a field (bit values over hypercube vertices). Natural fit. |
| "Predict color at coordinate (x,y)": input is a point. Wrong tool. Use an MLP. | "Predict f(v) for vertex v": input is a point. Wrong tool. Use an MLP. |

Boolean function learning is the second row. The input IS a hypercube vertex, but the task is point classification, not field classification.

## What native hypercube data actually looks like

The architecture's natural input is an **activation map** — a value at every vertex, or a binary vector where each bit has independent meaning. Examples:

- **Molecular fingerprints**: a 1024-bit ECFP fingerprint where each bit indicates the presence of a molecular substructure. The input IS a pattern over the hypercube. Hamming distance between fingerprints measures structural similarity. The conv kernel sees neighbors at distance 1 — structurally meaningful.

- **Feature interaction data**: N binary features where the prediction depends on feature combinations. The full feature vector IS the activation map.

The key distinction: in native hypercube data, the **entire vector** is the input, and the task is to classify the **pattern**. In Boolean function learning, a **single vertex** is the input, and the task is to classify the **position**.

## What this means for the research plan

Boolean functions are not the right validation target for HypercubeCNN. The architecture was designed for field classification on hypercubes, not point classification. The correct path forward:

1. **Molecular fingerprints** — the real native test case. Input is a binary vector (activation map). Classification task (e.g., solubility, toxicity). The architecture operates on its natural substrate without encoding gymnastics.

2. **MLP baseline for Boolean functions** — if we still want Boolean function results for comparison, an MLP (10 binary inputs -> hidden layers -> output) is the right architecture. It would demonstrate what the hypercube CNN is NOT designed for, contrasting with what it IS designed for (field classification on molecular fingerprints).

## The GAP invariance theorem

One useful theoretical result emerged from this investigation. For any encoding where input vertices are related by XOR-translations (including one-hot and bipolar):

> **Theorem**: Let the input encoding satisfy `encode(v) = T_a(encode(v XOR a))` where T_a is XOR-translation by a. Then for any sequence of hypercube convolution layers followed by global average pooling: `GAP(conv(encode(v))) = GAP(conv(encode(v')))` for all vertices v, v'.

> **Proof**: Hypercube convolution commutes with XOR-translation (the kernel uses XOR-based neighbors). GAP is invariant under vertex permutation. Therefore `GAP(conv(T_a(x))) = GAP(T_a(conv(x))) = GAP(conv(x))`.

This is the hypercube analogue of spatial CNNs' translation invariance: conv+GAP cannot distinguish the same pattern at different positions. For point-classification tasks, this is fatal. For pattern-classification tasks (the intended use), this is a feature — the classification doesn't depend on arbitrary relabeling of the hypercube vertices.
