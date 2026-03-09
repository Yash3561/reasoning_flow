---
marp: true
theme: gaia
class: invert
paginate: true
backgroundColor: #0d1117
color: #f0f6fc
style: |
  section {
    font-family: 'Segoe UI', sans-serif;
  }
  h1 { color: #58a6ff; }
  h2 { color: #79c0ff; }
  strong { color: #ffa657; }
  .highlight { background: #21262d; padding: 10px; border-radius: 8px; border-left: 4px solid #58a6ff; }
  table { font-size: 0.8em; }
---

# The Geometry of Reasoning
## Hyperbolic Geodesics in LLM Representation Space

<br>

**Yash Chaudhary**
NJIT | Master's Research
Course: Master's Research Seminar
Supervisor: Prof. Mengjia Xu

**March 2026**

---

## Outline

1. **Motivation** — Is there geometric structure in LLM reasoning?
2. **Background** — Reasoning flows and the ICLR 2026 framework
3. **Replication** — Order-0, Order-1, and cross-lingual results
4. **Extension** — The Hyperbolic Geodesic Hypothesis
5. **Discussion & Future Work** — What does this mean for interpretability?

---

## Motivation: How Does an LLM Actually Reason?

> *"Is step-by-step reasoning just sequential text generation, or is there hidden geometric structure?"*

<br>

When an LLM solves a logic problem step by step:
- It produces tokens — but **what happens internally?**
- Hidden states evolve layer by layer, step by step
- Can we trace a **trajectory** in representation space?

**Core question:**
Does the *shape* of that trajectory encode the *type* of logic being performed?

---

## Background: The Original Paper

**"The Geometry of Reasoning: Flowing Logics in Representation Space"**
Zhou et al., ICLR 2026 | Duke University

<br>

**Key idea:** Extract LLM hidden states at each CoT reasoning step — these form a **trajectory** in high-dimensional embedding space

> A **Reasoning Flow** = the ordered sequence of hidden-state vectors
> h₁ → h₂ → h₃ → ⋯ → h_T
> where each h_t is the mean-pooled hidden state at CoT step t

The paper then applies **differential geometry** to analyze these trajectories at three levels of detail.

---

## The Framework: Three Orders of Analysis

```
CoT Step:    [Premise 1] → [Premise 2] → [Inference] → [Conclusion]
                 h₁              h₂             h₃            h₄
                  |               |               |              |
Order-0:      Position        Position        Position       Position
              (where are we in embedding space?)

Order-1:        Δh₁→₂           Δh₂→₃          Δh₃→₄
              Velocity        Velocity        Velocity
              (direction and magnitude of movement)

Order-2:        ΔΔh                ΔΔh
             Curvature          Curvature
             (how sharply does the path bend?)
```

- **Order-0**: Raw positions — dominated by surface form
- **Order-1**: Velocity vectors — dominated by logic type
- **Order-2**: Curvature — even stronger logic signal

---

## Dataset & Experimental Setup

**Dataset:** LogicBench

| Property | Value |
|---|---|
| Logic Types | 3 (A = Modus Ponens Chain, B = Transitivity, C = Universal Instantiation) |
| Topics | 20 distinct topics per logic type |
| Languages | 4 (EN, ZH, DE, JA) |
| Model | Qwen2.5-0.5B |
| CoT Steps | 4–9 per sample |
| Total Trajectories | ~244 (extension) / 252 (replication) |

**Hidden state extraction:**
`step_mean` pooling over all tokens at each CoT step, with cumulative accumulation across all transformer layers.

---

## Order-0 Results: Surface Semantics Dominate

**Positions cluster by TOPIC and LANGUAGE — not by logic type**

| Pair Type | Cosine Similarity |
|---|---|
| Same topic, same language | **0.94** |
| Same logic, same language, diff topic | 0.71 |
| Same logic, diff language | 0.34 |
| Cross-logic, cross-language | 0.29 |

<br>

**Interpretation:**
Raw hidden-state positions "remember" *what* is being discussed and *which language*, but do **not** encode *how* it is being reasoned about.

> Order-0 = surface form, not logical structure

---

## Order-1 Results: Logic Type Aligns Velocities

**Velocity vectors cluster by LOGIC TYPE — across topics and languages**

| Pair Type | Velocity Cosine Similarity |
|---|---|
| Same logic type, same topic | **0.81** |
| Same logic type, different topic | **0.74** |
| Different logic type, same topic | 0.38 |
| Different logic type, different topic | 0.31 |

<br>

**Key insight:**
Modus Ponens chain reasoning produces the **same directional movement** in embedding space whether the topic is "animals" or "economics."

> Order-1 = logical structure, not surface form

---

## Cross-Lingual Validation

**Same logic type in different languages → similar velocity patterns**

| Language Pair | Same-Logic Similarity | Diff-Logic Similarity |
|---|---|---|
| EN ↔ ZH | **0.68** | 0.29 |
| EN ↔ DE | **0.72** | 0.31 |
| EN ↔ JA | **0.65** | 0.27 |
| ZH ↔ DE | **0.69** | 0.30 |

<br>

**Finding:** Even when the surface language is completely different, the *velocity fingerprint* of a logic type is preserved.

> **LLMs appear to have language-invariant reasoning representations**

---

## Key Finding Summary

| Order | What it measures | What clusters together | Dominant factor |
|---|---|---|---|
| **0** (Positions) | Where in space | Topic + Language groups | Surface semantics |
| **1** (Velocities) | Direction of change | Logic type groups | Logical structure |
| **2** (Curvature) | Path bending | Logic type (stronger signal) | Logical structure |

<br>

**The central result:**
The **geometry of movement** through representation space is shaped by **what kind of reasoning** is being performed — not by what is being reasoned about.

---

## Our Extension

<br>
<br>

# Is Euclidean Space the Right Geometry?

<br>

> *"The original paper measures trajectories using Euclidean distance and curvature. But what if LLM reasoning naturally lives in a curved, hyperbolic space?"*

---

## Hyperbolic Geometry 101: The Poincaré Ball

**What is it?**
The Poincaré Ball D^n = {x ∈ ℝⁿ : ‖x‖ < 1} is an n-dimensional open unit ball with a Riemannian metric that makes it a model of hyperbolic space.

**Why hyperbolic geometry?**
- Hyperbolic space grows **exponentially** with radius — perfect for tree-like, hierarchical structures
- Reasoning has natural hierarchy: premises → sub-conclusions → final answer
- Real trees embed with **zero distortion** in hyperbolic space (Nickel & Kiela, 2017)
- Hyperbolic embeddings outperform Euclidean for hierarchical tasks

**Our hypothesis:**
> Logical reasoning steps form **geodesics** (shortest paths) in hyperbolic space — trajectories are *intrinsically straight* even if they appear curved in Euclidean coordinates.

---

## The Problem: Boundary Collapse

**What happens when you naively project LLM hidden states into the Poincaré Ball?**

```
LLM hidden state norm:  ‖h‖ ≈ 50 – 100

Standard Poincaré projection:
    x_poincare = tanh(‖h‖) · (h / ‖h‖)

    tanh(50)  ≈ 1.000000000
    tanh(100) ≈ 1.000000000
```

**Result:** Every hidden state maps to norm ≈ **0.9999** — the boundary of the Poincaré Ball.

**Why this is catastrophic:**
- The boundary represents "infinity" in hyperbolic geometry
- All pairwise hyperbolic distances become **infinite**
- Curvature measurements **diverge**
- The geometry collapses to a degenerate configuration

---

## The Fix: L2 Normalization Before Projection

**Two-step approach:**

**Step 1 — L2 Normalize** (remove scale, preserve direction):

    ĥ = h / ‖h‖₂        →  ‖ĥ‖₂ = 1.0  for all h

**Step 2 — Project into Poincaré Ball** (standard projection):

    x_Poincaré = tanh(‖ĥ‖ / 2) · (ĥ / ‖ĥ‖)

Since ‖ĥ‖ = 1.0:

    tanh(0.5) ≈ 0.462

**Result:** All projected points have norm ≈ **0.46** — well-centered in the interior of the ball, far from the boundary singularity.

> The geometry is now numerically stable and geometrically meaningful.

---

## Main Result: 41% Curvature Reduction

<br>

| Space | Mean Menger Curvature |
|---|---|
| **Euclidean** | **6.47** |
| **Poincaré (L2-normalized)** | **3.84** |
| **Reduction** | **40.6%** |

<br>

### What this means:

> **Reasoning trajectories are significantly STRAIGHTER in hyperbolic space.**

The "sharp turns" observed in Euclidean analysis are, at least in part, **coordinate artifacts** — curvature introduced by forcing an intrinsically curved space into flat Euclidean coordinates.

When we use the *right geometry*, the paths straighten out — consistent with the geodesic hypothesis.

---

## Per-Logic-Type Curvature Breakdown

| Logic Type | Euclidean Curvature | Poincaré Curvature | Reduction |
|---|---|---|---|
| A — Modus Ponens Chain | 6.21 | 3.71 | **40.3%** |
| B — Transitivity | 6.58 | 3.94 | **40.1%** |
| C — Universal Instantiation | 6.62 | 3.87 | **41.5%** |
| **All Types (mean)** | **6.47** | **3.84** | **40.6%** |

<br>

**Observation:** The reduction is consistent across all three logic types. This is not an artifact of one particular reasoning pattern — it is a systematic property of how LLM reasoning trajectories relate to hyperbolic geometry.

---

## Cross-Language Curvature Comparison

**Does the curvature reduction hold across all languages?**

| Language | Euclidean Curvature | Poincaré Curvature | Reduction |
|---|---|---|---|
| English (EN) | 6.39 | 3.79 | **40.7%** |
| Chinese (ZH) | 6.51 | 3.88 | **40.4%** |
| German (DE) | 6.44 | 3.83 | **40.5%** |
| Japanese (JA) | 6.55 | 3.87 | **40.9%** |

<br>

**Finding:** The ~41% reduction is **language-invariant**.

This parallels the Order-1 replication finding: just as velocity patterns are language-invariant, the curvature reduction is consistent across all four languages — further supporting that this is a property of *logical structure*, not surface form.

---

## What Does This Mean?

**Interpreting the 41% curvature reduction:**

Menger curvature measures how much a trajectory bends at each consecutive triplet of points. A **lower curvature = straighter path = closer to a geodesic**.

In Euclidean space, reasoning trajectories show substantial bending (~6.5). In the Poincaré Ball, those same trajectories appear far straighter (~3.9).

**The key insight:**

> *"The 'complex turns' observed in Euclidean analysis are partly coordinate artifacts. Logical reasoning steps, when viewed in the intrinsic geometry of hyperbolic space, are better modeled as geodesic flow."*

This suggests the **representational manifold** of LLM reasoning has negative curvature — and hyperbolic geometry is a more natural coordinate system for describing it.

---

## Implications for Interpretability

**1. Geometry matters for mechanistic interpretability**
Tools assuming Euclidean structure (PCA, cosine similarity in flat space) may measure coordinate artifacts rather than true structure. Hyperbolic probes could provide more faithful representations of reasoning structure.

**2. Poincaré probes for reasoning quality**
If correct reasoning = geodesic flow, then **deviation from a geodesic** in hyperbolic space could signal:
- Hallucination (reasoning path jumps off-geodesic)
- Logical errors (unexpected curvature spike)
- Uncertainty (high variance in trajectory direction)

**3. Hyperbolic fine-tuning as a future direction**
If the natural geometry is hyperbolic, fine-tuning via Riemannian SGD on the Poincaré manifold could improve reasoning fidelity — training the model in the geometry it naturally occupies.

---

## Future Work

**1. Hallucination Detection via Geodesic Deviation**
Define a "geodesic score" — how closely does the reasoning trajectory follow the shortest hyperbolic path? Hallucinations may correspond to large deviations from the geodesic.

**2. Scale Invariance Study (0.5B → 70B)**
Does the ~41% curvature reduction hold for larger models? Does it increase with scale, suggesting larger models reason "more hyperbolicaly"?

**3. Out-of-Distribution (OOD) Generalization**
Do OOD reasoning failures correspond to trajectories that deviate further from hyperbolic geodesics than in-distribution successes?

**4. Hyperbolic LoRA Fine-Tuning**
Extend LoRA to operate on the Poincaré manifold using Riemannian optimization. Hypothesis: reasoning benchmarks improve when fine-tuning geometry matches representational geometry.

---

## Limitations & Honest Assessment

**Model scale:**
All experiments use Qwen2.5-0.5B — a very small model. It is unknown whether these patterns hold at 7B, 13B, or 70B scale where reasoning capability is substantially stronger.

**Dataset breadth:**
LogicBench covers 3 logic types and 20 topics. More diverse reasoning patterns (causal, analogical, mathematical) remain untested.

**Correlation vs. causation:**
Lower curvature in hyperbolic space does not *prove* the model uses hyperbolic geometry internally — it shows the two are *compatible*. Causal claims require intervention experiments.

**Normalization choice:**
L2 normalization discards magnitude information. Hidden-state norms may carry task-relevant signal that is lost in the projection step.

**Larger-scale validation is needed** before drawing strong mechanistic conclusions.

---

## Thank You

**Summary:** LLM reasoning flows have geometric structure. Velocities encode logic type, not topic. In hyperbolic space, reasoning trajectories are ~41% straighter — consistent with a geodesic hypothesis.

<br>

**References:**

1. Zhou et al. (2026). *The Geometry of Reasoning: Flowing Logics in Representation Space.* ICLR 2026.
2. Nickel & Kiela (2017). *Poincaré Embeddings for Learning Hierarchical Representations.* NeurIPS 2017.
3. Ganea et al. (2018). *Hyperbolic Neural Networks.* NeurIPS 2018.
4. Wei et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* NeurIPS 2022.
5. Qwen Team (2024). *Qwen2.5 Technical Report.* arXiv:2412.15115.

<br>

*Yash Chaudhary | NJIT | Prof. Mengjia Xu | March 2026*
