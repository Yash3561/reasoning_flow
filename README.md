# The Geometry of Reasoning: Flowing Logics in Representation Space

[![arXiv](https://img.shields.io/badge/arXiv-2510.09782-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2510.09782)
[![GitHub](https://img.shields.io/badge/GitHub-Reasoning--Flow-181717?logo=github)](https://github.com/MasterZhou1/Reasoning-Flow)
[![HF Dataset](https://img.shields.io/badge/HF%20Datasets-Reasoning--Flow-ff8b2f?logo=huggingface&logoColor=white)](https://huggingface.co/datasets/MasterZhou/Reasoning-Flow)
[![OpenReview](https://img.shields.io/badge/OpenReview-Reasoning--Flow-ff8b2f?logo=openreview&logoColor=white)](https://openreview.net/forum?id=ixr5Pcabq7)

*ICLR 2026*

**Original Authors:** [Yufa Zhou](https://masterzhou1.github.io/), [Yixiao Wang](https://yixiao-wang-stats.github.io/), [Xunjian Yin](https://xunjianyin.github.io/), [Shuyan Zhou](https://www.shuyanzhou.com/), [Anru R. Zhang](https://anruzhang.github.io/) — **Duke University**

---

## Research Extension: Non-Euclidean Geometry of Reasoning

This repository has been extended to investigate the **intrinsic manifold geometry** of LLM reasoning trajectories. While the original framework operates in Euclidean space, we hypothesize that logical reasoning—which is inherently hierarchical and convergent—is better represented in **Hyperbolic space**.

### Key Discovery: The Hyperbolic Geodesic Hypothesis

By projecting the hidden states into a **Poincaré Ball** ($c = -1$), we observed a significant reduction in the complexity of reasoning paths.

![Curvature Comparison](results/curvature_comparison_chart.png)

#### 1. Numerical Stability & Boundary Collapse

Initial experiments revealed a **"Boundary Collapse"** artifact where high-magnitude LLM hidden states clustered at the Poincaré boundary ($Norm \approx 0.999$), artificially inflating curvature metrics. 

**The Problem:**
- LLM hidden states have very large norms (50-100+)
- The Poincaré exponential map uses `tanh()`, which saturates for large inputs
- Result: All points collapsed to the boundary, where the metric tensor diverges

**The Solution:** We implemented **L2-Normalization** before the exponential map, centering the embeddings ($Norm \approx 0.46$) and enabling stable geometric analysis.

#### 2. Curvature Reduction Results

Using the stable projection, we compared **Menger Curvature** across 240+ reasoning trajectories:

| Space | Menger Curvature | Interpretation |
|-------|------------------|----------------|
| **Euclidean** | ~6.5 | Reasoning appears as complex "turns" |
| **Poincaré (Hyperbolic)** | ~3.9 | Reasoning straightens into geodesics |
| **Reduction** | **41%** | Paths are significantly straighter in hyperbolic space |

**Conclusion:** LLM reasoning "straightens out" in hyperbolic space. This confirms that logical flows are closer to **hyperbolic geodesics** than Euclidean lines. The "complex turns" observed in the original paper are partially artifacts of a mismatched coordinate system.

### Open Questions (Future Work)

1. **OOD Generalization:** Does the curvature drop hold for out-of-distribution logic?
2. **Hallucination Detection:** Can we train a probe to detect hallucinations based on geodesic deviation?
3. **Scale Invariance:** How does sectional curvature change across model scales (1B → 70B)?
4. **Hyperbolic Fine-Tuning:** Can we force models to follow straighter (more logical) paths?

---

## Quick Start

### Installation

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run the Hyperbolic Analysis

```bash
# Step 1: Generate embeddings (original Euclidean analysis)
python cot-hidden-dynamic.py \
  --hf_model Qwen/Qwen2.5-0.5B \
  --data_file data/demo_subset.json \
  --similarity_order 0 \
  --save_dir results/exp1_order0

# Step 2: Run hyperbolic curvature analysis (L2-stabilized)
python experiments/hyperbolic_l2_normalized.py \
  --data_dir results/exp1_order0/data \
  --output results/curvature_report.csv

# Step 3: Validate numerical stability
python experiments/validate_spherical.py \
  --data_dir results/exp1_order0/data

# Step 4: Generate visualization
python experiments/generate_chart.py
```

---

## Repository Structure

```
Reasoning-Flow/
├── data/                    # LogicBench dataset
│   ├── all_final_data.json  # Full dataset (243 samples)
│   └── demo_subset.json     # Quick test subset (15 samples)
├── src/                     # Original Duke paper source code
│   ├── cot-hidden-dynamic.py
│   ├── compute_similarity_averages.py
│   ├── generate_dataset.py
│   └── utils.py
├── experiments/             # Hyperbolic extension (NEW)
│   ├── hyperbolic_l2_normalized.py   # L2-stabilized Poincaré analysis
│   ├── validate_spherical.py         # Numerical stability checker
│   └── generate_chart.py             # Visualization generator
├── results/                 # Figures and CSVs
│   ├── curvature_comparison_chart.png
│   └── curvature_l2_normalized.csv
├── assets/                  # Original paper figures
├── README.md
└── requirements.txt
```

---

## Original Paper Summary

We study how large language models (LLMs) **reason through their embeddings** by introducing a **geometric framework of reasoning flows**, where reasoning unfolds as trajectories in representation space.

### Key Findings (Zhou et al., 2026)

1. **LLM reasoning forms smooth flows** in embedding space
2. **Logical statements act as local controllers** governing flow velocity
3. **Order-0 (positions):** Embeddings cluster by surface semantics
4. **Order-1 (velocities):** Same-logic trajectories align across topics/languages
5. **Order-2 (curvature):** Logic signal strengthens beyond surface semantics

### Visualizations

| **3D PCA Trajectories** | **2D PCA Projection** |
|-------------------------|----------------------|
| ![3D PCA](assets/reasoning_flows_pca_math500_3d.png) | ![2D PCA](assets/reasoning_flows_pca_math500_2d.png) |

---

## Citation

**Original Paper:**
```bibtex
@inproceedings{zhou2025geometry,
  title     = {The Geometry of Reasoning: Flowing Logics in Representation Space},
  author    = {Zhou, Yufa and Wang, Yixiao and Yin, Xunjian and Zhou, Shuyan and Zhang, Anru R.},
  booktitle = {ICLR 2026},
  year      = {2026},
  url       = {https://openreview.net/forum?id=ixr5Pcabq7}
}
```

**Hyperbolic Extension:**
```bibtex
@misc{choudhary2026hyperbolic,
  title  = {Hyperbolic Geodesics as the Manifold of Logical Reasoning},
  author = {Chaudhary, Yash},
  year   = {2026},
  note   = {Extension demonstrating 41\% curvature reduction in Poincaré space}
}
```

---

## Contact

**Original Paper:** Yufa Zhou — [yufa.zhou@duke.edu](mailto:yufa.zhou@duke.edu)  
**Hyperbolic Extension:** Yash Chaudhary
