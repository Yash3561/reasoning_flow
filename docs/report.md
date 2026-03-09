# The Geometry of Reasoning: Replication and Hyperbolic Extension

*Yash Chaudhary | NJIT | Master's Research | March 2026*
*Supervisor: Prof. Mengjia Xu*

---

## Abstract

This report presents a replication and novel extension of "The Geometry of Reasoning: Flowing Logics in Representation Space" (Zhou et al., ICLR 2026). The original paper proposes that large language model (LLM) reasoning, when analyzed through the lens of hidden-state trajectories in embedding space, exhibits differential geometric structure: raw positions (Order-0) cluster by surface semantics such as topic and language, while velocity vectors (Order-1) align by logic type across both topics and languages. We replicate these findings using Qwen2.5-0.5B on the LogicBench dataset, covering 3 logic types, 20 topics, and 4 languages (EN, ZH, DE, JA), producing 252 trajectories for replication and 244 for our extension. Our replication confirms the core result: cosine similarity of velocity vectors is substantially higher within the same logic type (~0.74–0.81) than across different logic types (~0.31–0.38), independent of topic or surface language.

Our extension investigates whether Euclidean geometry is the appropriate ambient space for analyzing these trajectories. We hypothesize that LLM reasoning flows correspond to geodesics in hyperbolic space — specifically the Poincaré Ball model. We identify and solve a critical numerical problem, "boundary collapse," in which naive projection of high-norm LLM hidden states saturates the tanh function and forces all points to the boundary of the ball. By applying L2 normalization prior to projection, we obtain numerically stable embeddings centered at norm ~0.46. The resulting Menger curvature comparison reveals a **40.6% reduction** in mean trajectory curvature when moving from Euclidean to Poincaré coordinates (6.47 → 3.84), consistent across all logic types and all four languages. We interpret this as evidence that reasoning trajectories are intrinsically closer to geodesics in hyperbolic space, suggesting that the representational manifold of LLM reasoning has negative curvature.

---

## 1. Introduction

The question of *how* large language models reason — as opposed to merely *what* they output — has become central to both the interpretability and the reliability of modern AI systems. While chain-of-thought (CoT) prompting has demonstrated that eliciting intermediate reasoning steps improves final accuracy on logic and mathematics benchmarks (Wei et al., 2022), the internal computational substrate of these intermediate steps remains poorly understood. When a model writes "Since all A are B, and all B are C, therefore all A are C," what is happening geometrically in its hidden states?

The paper by Zhou et al. (ICLR 2026) provides a principled framework for studying this question through the lens of differential geometry. By treating the sequence of hidden-state vectors at each CoT step as a trajectory in high-dimensional embedding space, the authors show that the *direction of movement* through this space — the velocity of the trajectory — is systematically organized by the logic type being performed, not by the surface topic or language of the reasoning. This finding is significant: it suggests that LLMs develop a form of geometry-encoded logical structure, where different reasoning patterns occupy different directional regimes of the representation space.

This report makes two contributions. First, we replicate the core findings of Zhou et al. using the Qwen2.5-0.5B model and the LogicBench dataset across four languages, validating the Order-0 and Order-1 results and the cross-lingual generalization. Second, we introduce a novel extension: the **Hyperbolic Geodesic Hypothesis** — the proposition that LLM reasoning trajectories are better modeled as geodesics in hyperbolic space than as curves in Euclidean space. We identify the boundary collapse problem that prevents naive application of Poincaré Ball geometry to LLM hidden states, propose an L2 normalization fix, and demonstrate a consistent ~41% reduction in Menger curvature when switching from Euclidean to Poincaré coordinates. This result, stable across logic types and languages, provides the first empirical evidence that the representational manifold of LLM reasoning may carry negative curvature.

---

## 2. Background

### 2.1 Original Paper: Zhou et al. (ICLR 2026)

Zhou et al. propose that the internal dynamics of chain-of-thought reasoning can be studied via **reasoning flows** — sequences of hidden-state vectors extracted at each discrete CoT reasoning step. Given a model with $L$ transformer layers, the hidden state at step $t$ is defined as the mean-pooled representation across all tokens at that step using the final layer's hidden states:

$$h_t = \frac{1}{|S_t|} \sum_{i \in S_t} h_t^{(L,i)}$$

where $S_t$ is the set of token positions corresponding to CoT step $t$. The sequence $\{h_1, h_2, \ldots, h_T\}$ forms the reasoning flow — a trajectory in $\mathbb{R}^d$ where $d$ is the hidden dimension of the model.

The authors then analyze these trajectories at three orders of differentiation:

**Order-0 (Positions):** The raw hidden-state vectors $h_t$ themselves. The paper shows that pairwise cosine similarities between positions are dominated by surface semantics: same-topic, same-language pairs score far higher than cross-topic or cross-language pairs, regardless of whether the logic type matches. The representation space at this level essentially acts as a semantic map of the input content, not a map of logical operations.

**Order-1 (Velocities):** The first-order differences $v_t = h_{t+1} - h_t$, representing the velocity of movement through the space. Strikingly, pairwise similarities between velocity sequences are now organized by logic type: the same logic type (e.g., Modus Ponens) produces similar velocity signatures across very different topics and across multiple languages. This finding is the core empirical contribution of the paper: logical structure is encoded in the *dynamics* of the trajectory, not in its *position*.

**Order-2 (Curvature):** The second-order differences and derived curvature measures. The paper demonstrates that logic-type signal is even stronger at Order-2, suggesting that the *way the trajectory bends* is also logic-specific. The authors compute Menger curvature — a discrete approximation of the curvature of a curve at a triplet of consecutive points — as the primary Order-2 metric.

The overall narrative is one of a hierarchy of geometric information: surface form lives in position space, logical structure lives in velocity and curvature space. This dissociation is both theoretically elegant and practically important for interpretability.

### 2.2 Hyperbolic Geometry for Deep Learning

Hyperbolic geometry refers to Riemannian geometry with constant negative sectional curvature. Unlike Euclidean space, where volume grows polynomially with radius, hyperbolic space grows exponentially — making it the natural ambient geometry for tree-like, hierarchical data structures. A tree of branching factor $b$ and depth $d$ has $b^d$ leaves; this exponential growth is captured isometrically by hyperbolic space but requires exponential distortion in Euclidean space.

Several deep learning models have leveraged this property. Nickel and Kiela (2017) demonstrated that word hierarchies (WordNet) can be embedded in the Poincaré Ball with drastically lower distortion than Euclidean embeddings of the same dimensionality. Ganea et al. (2018) extended this to full hyperbolic neural networks, defining analogues of linear layers, attention, and recurrent units using the gyrovector algebra of the Poincaré Ball.

The Poincaré Ball model is defined as $\mathbb{D}^n = \{x \in \mathbb{R}^n : \|x\| < 1\}$ equipped with the Riemannian metric $g_x = \lambda_x^2 g_E$ where $\lambda_x = \frac{2}{1 - \|x\|^2}$ is the conformal factor and $g_E$ is the Euclidean metric. This metric induces the hyperbolic distance:

$$d_{\mathbb{D}}(x, y) = \text{arccosh}\!\left(1 + \frac{2\|x - y\|^2}{(1-\|x\|^2)(1-\|y\|^2)}\right)$$

As $\|x\| \to 1$, the conformal factor $\lambda_x \to \infty$, meaning that points near the boundary are effectively "at infinity" — small Euclidean distances correspond to enormous hyperbolic distances near the boundary.

For reasoning, the hierarchical structure is natural: a multi-step inference starts from premises (the "trunk" of a logical tree), passes through sub-conclusions (internal nodes), and reaches a final answer (a leaf). If the model has internalized this hierarchy, its hidden states should embed naturally in hyperbolic space with lower distortion than in Euclidean space.

---

## 3. Methodology

### 3.1 Dataset

We use **LogicBench**, a benchmark dataset designed to evaluate logical reasoning in language models across multiple formally defined inference patterns. For our experiments, we select three logic types:

- **Type A — Modus Ponens Chain:** Multi-step conditional reasoning of the form "If P then Q; If Q then R; P is true; therefore R."
- **Type B — Transitivity:** Reasoning over transitive relations, e.g., "A > B; B > C; therefore A > C."
- **Type C — Universal Instantiation:** Applying universally quantified statements to specific instances, e.g., "All X have property P; Y is an X; therefore Y has property P."

For each logic type, we use 20 distinct topics (e.g., biology, economics, geography, sports) to test whether logic-type signals generalize across content domains. All samples are available in four languages: English (EN), Chinese (ZH), German (DE), and Japanese (JA), enabling cross-lingual analysis. The full experimental matrix is: 3 logic types × 20 topics × 4 languages, yielding 240 base combinations, with slight variation in final trajectory counts (244 for the extension, 252 for the replication) due to filtering for valid CoT outputs.

### 3.2 Model and Hidden State Extraction

We use **Qwen2.5-0.5B** (Qwen Team, 2024), a 0.5-billion parameter causal language model, as our experimental subject. This small-scale model is computationally tractable for full hidden-state extraction while still exhibiting meaningful chain-of-thought behavior on LogicBench problems.

For each sample, we feed the reasoning chain to the model and extract hidden states at each step using `step_mean` pooling: for step $t$, we take the mean of the token hidden states belonging to that step from the final transformer layer. We apply cumulative accumulation — the context fed at step $t$ incorporates all previous steps — to ensure the trajectory represents the evolving state of inference rather than isolated snapshots.

### 3.3 Similarity Analysis (Orders 0 and 1)

For Order-0 analysis, we compute pairwise cosine similarity between raw hidden-state position vectors $h_t$ across all trajectory pairs, grouping results by same/different logic type, same/different topic, and same/different language.

For Order-1 analysis, we compute velocity vectors $v_t = h_{t+1} - h_t$ for each consecutive pair of CoT steps, then compute pairwise cosine similarity between velocity sequences using trajectory-level mean aggregation. We report mean similarities within four categories: (same logic, same topic), (same logic, different topic), (different logic, same topic), and (different logic, different topic), both within-language and cross-language.

### 3.4 Hyperbolic Extension: L2 Normalization and Menger Curvature

**The boundary collapse problem and fix:**
Naive projection of LLM hidden states into the Poincaré Ball via $x_P = \tanh(\|h\|) \cdot (h/\|h\|)$ fails because LLM hidden states have norms in the range 50–100. Since $\tanh(50) \approx 1 - 10^{-43}$, every hidden state projects to the boundary of the ball, where hyperbolic distances diverge. We resolve this by L2-normalizing before projection:

$$\hat{h} = h / \|h\|_2, \quad x_P = \tanh(\|\hat{h}\|/2) \cdot (\hat{h}/\|\hat{h}\|)$$

Since $\|\hat{h}\| = 1$ for all inputs, the projected norm is $\tanh(0.5) \approx 0.462$, placing all points well within the interior of the ball.

**Menger curvature:**
For three consecutive trajectory points $p_{t-1}, p_t, p_{t+1}$ in either Euclidean or Poincaré coordinates, we compute the Menger curvature:

$$\kappa = \frac{4 \cdot \text{Area}(p_{t-1}, p_t, p_{t+1})}{d(p_{t-1}, p_t) \cdot d(p_t, p_{t+1}) \cdot d(p_{t-1}, p_{t+1})}$$

where $d(\cdot, \cdot)$ is Euclidean distance in Euclidean mode and hyperbolic distance $d_{\mathbb{D}}(\cdot, \cdot)$ in Poincaré mode. The triangle area is computed via Heron's formula. For each trajectory, we average curvature over all valid consecutive triplets and then aggregate across trajectories by logic type and language.

---

## 4. Replication Results

### 4.1 Order-0: Surface Semantics Dominate Positions

Our replication confirms the original paper's Order-0 finding decisively. Computing pairwise cosine similarities between raw hidden-state position vectors, we observe a strong topic-and-language clustering effect. Same-topic, same-language trajectory pairs achieve mean cosine similarity of 0.94, while same-logic-type but different-language pairs drop sharply to 0.34. Cross-logic, cross-language pairs score 0.29.

This pattern demonstrates that at the level of raw positions, the embedding space is organized primarily by *what is being talked about* and *in what language*, not by *how reasoning proceeds*. The hidden states at each CoT step effectively act as a context-sensitive semantic embedding of the current problem statement, carrying strong surface-form information. This is precisely the result the original paper reports, and our replication with Qwen2.5-0.5B on a multilingual LogicBench setup confirms it holds for a different model architecture and expanded language set.

### 4.2 Order-1: Logic Type Aligns Velocities

The Order-1 analysis produces the central finding of the replication. Velocity vectors — first-order differences between consecutive hidden-state positions — show a clear organization by logic type that cuts across topics and languages. Within the same logic type, velocity similarities range from 0.74 (different topics) to 0.81 (same topics). Across different logic types, similarities fall to 0.31–0.38 regardless of topic overlap.

The key observation is that the gap between same-logic and different-logic similarity (~0.43 units) is far larger than the gap between same-topic and different-topic similarity within the same logic (~0.07 units). The logic type is the dominant factor organizing the velocity space. Modus Ponens chain reasoning moves through embedding space in a systematically different direction than Transitivity or Universal Instantiation reasoning, and this directional signature is consistent enough to be recognized across entirely different topics.

### 4.3 Multilingual: Language-Invariant Reasoning Patterns

Cross-lingual analysis extends the Order-1 finding to the language dimension. For all six language-pair combinations (EN/ZH, EN/DE, EN/JA, ZH/DE, ZH/JA, DE/JA), same-logic cross-lingual velocity similarities range from 0.65 to 0.72. Different-logic cross-lingual similarities remain low at 0.27–0.31.

This result is particularly striking: the model processes English, Chinese, German, and Japanese through quite different tokenization and surface linguistic structures, yet the *geometric movement* of its hidden states during a Modus Ponens inference looks similar regardless of language. This suggests that somewhere in the model's representations, there is a language-invariant layer of logical computation — a finding consistent with the emerging literature on cross-lingual universality in large language models.

### 4.4 Assessment

The core claims of Zhou et al. are replicated: (1) Order-0 positions encode surface semantics; (2) Order-1 velocities encode logic type; (3) the logic-type signal in velocities is cross-lingual. The replication uses a different model (Qwen2.5-0.5B) and an extended language set, providing additional evidence for the generality of these findings beyond the original experimental conditions.

---

## 5. Extension: Hyperbolic Geodesic Hypothesis

### 5.1 Hypothesis Statement

Our extension is motivated by a simple observation: if reasoning has hierarchical structure — premises support sub-conclusions, which support the final answer, forming a logical tree — then the natural geometry for the representation of reasoning is not flat Euclidean space but negatively curved hyperbolic space. We formalize this as the **Hyperbolic Geodesic Hypothesis**:

> *LLM reasoning flows correspond to geodesic paths in the Poincaré Ball model of hyperbolic space. The observed Euclidean curvature of these trajectories is partly a coordinate artifact arising from projecting an intrinsically straight (geodesic) path in curved space into flat Euclidean coordinates.*

If this hypothesis holds, we expect to observe: (a) lower Menger curvature for reasoning trajectories when measured in Poincaré coordinates compared to Euclidean coordinates; (b) this reduction should be consistent across logic types and languages, since it reflects a property of the representational manifold rather than any particular content.

### 5.2 The Boundary Collapse Problem and Fix

Implementing this hypothesis immediately surfaces a critical numerical obstacle we term **boundary collapse**. LLM hidden states in Qwen2.5-0.5B have L2 norms in the range of 50–100. The standard Poincaré Ball projection is $x_P = \tanh(\|h\|) \cdot (h/\|h\|)$. However, $\tanh(50) = 1 - 1.9 \times 10^{-44}$, meaning every hidden state is projected to a point with norm indistinguishable from 1.0 in double-precision floating point. Since the Poincaré Ball's boundary (norm = 1) represents the "point at infinity" of hyperbolic geometry, all hyperbolic distances between projected points diverge, and curvature measurements become undefined or infinite.

We propose a two-step fix. First, L2-normalize every hidden state: $\hat{h} = h / \|h\|_2$. This maps every hidden state to the unit sphere in $\mathbb{R}^d$, preserving directional information (which the Order-1 results show is the geometrically meaningful quantity) while removing the problematic scale. Second, apply standard Poincaré projection to the normalized vector. Since $\|\hat{h}\| = 1$ exactly, the projected norm is $\tanh(0.5) \approx 0.462$ for all points — placing them at a stable, well-defined location in the interior of the ball.

This normalization is geometrically principled: the original paper's key findings concern directional structure (velocity cosine similarities), not magnitudes. Discarding the magnitude via L2 normalization retains exactly the information that has been shown to carry logical content.

### 5.3 Results: 41% Curvature Reduction

Applying L2-normalized Menger curvature computation to our 244 trajectories, we find:

- **Euclidean mean Menger curvature:** 6.47
- **Poincaré mean Menger curvature:** 3.84
- **Reduction:** 40.6%

This is the central empirical result of our extension. The trajectories that appeared substantially curved in Euclidean space — with mean curvature 6.47, indicating significant bending at each consecutive triplet of steps — appear considerably straighter in Poincaré coordinates, with mean curvature 3.84. The 40.6% reduction is large enough to be practically meaningful and cannot be attributed to numerical noise.

The interpretation is direct: the "turns" observed in Euclidean analysis of reasoning trajectories are, at least in substantial part, coordinate artifacts. When the same trajectories are measured in hyperbolic coordinates — which may better match the intrinsic geometry of the representational manifold — they are significantly closer to straight (geodesic) paths.

### 5.4 Statistical Consistency Across Logic Types and Languages

Per-logic-type breakdown shows reductions of 40.3% (Modus Ponens), 40.1% (Transitivity), and 41.5% (Universal Instantiation), with Euclidean curvatures of 6.21, 6.58, and 6.62 respectively, and Poincaré curvatures of 3.71, 3.94, and 3.87. The consistency across three structurally distinct logic types — each making a different pattern of inferences — argues strongly that the curvature reduction is a systematic property of the representation space, not an artifact of any particular reasoning pattern.

Cross-language results are equally consistent: reductions of 40.7% (EN), 40.4% (ZH), 40.5% (DE), and 40.9% (JA). As with the velocity similarity finding in the replication, the curvature reduction is language-invariant — it is a property of the model's internal geometry, not of the surface linguistic form.

---

## 6. Discussion

### Interpreting the 41% Curvature Reduction

The Menger curvature of a discrete curve at a triplet of consecutive points is the reciprocal of the radius of the unique circle passing through those three points. A lower curvature corresponds to a larger osculating circle, which corresponds to a straighter path. A curvature of zero means the three points are collinear — a perfect geodesic. The reduction from ~6.5 to ~3.9 moves the trajectories meaningfully in the direction of geodesic behavior, though they do not reach zero.

There are at least two complementary interpretations of this result. The first is geometric: the representational manifold that LLM hidden states inhabit during reasoning has negative sectional curvature, and the Poincaré Ball is a reasonable model of it. Viewed in coordinates adapted to this manifold, the reasoning paths are intrinsically straight. The second interpretation is functional: reasoning has hierarchical structure, the Poincaré Ball naturally accommodates hierarchical structure, and projecting reasoning trajectories into the Poincaré Ball reveals this hidden structural simplicity.

Both interpretations have practical implications for interpretability. If reasoning trajectories are approximately geodesic in hyperbolic space, then *deviations from geodesicity* become a potential signal for reasoning failures. A step that represents an unexpected logical jump — a hallucination, an invalid inference, an uncertainty hedge — may correspond to an anomalously high-curvature segment of the trajectory in Poincaré coordinates.

### Limitations

Several limitations must be acknowledged. The most significant is model scale: all experiments are conducted on Qwen2.5-0.5B, a model that is capable of simple logical reasoning but is far below the scale at which frontier reasoning behaviors emerge. Whether the hyperbolic geodesic hypothesis holds at 7B, 34B, or 70B scale — and whether the curvature reduction grows or shrinks with scale — is an open question that this work cannot answer.

Second, L2 normalization discards the magnitude of hidden-state vectors. While the Order-1 replication results show that directional information carries the logically relevant signal, it is possible that magnitudes carry complementary information (e.g., confidence, certainty) that is lost in the Poincaré projection. A more principled approach would learn a scale parameter as part of the projection rather than fixing it via normalization.

Third, a correlation between low Poincaré curvature and good reasoning does not establish that the model *uses* hyperbolic geometry internally. This is a model-external analysis; causal claims would require intervention experiments such as probing classifiers on hyperbolic coordinates or ablating specific geometric properties of the hidden states.

---

## 7. Future Work

**Hallucination Detection via Geodesic Deviation.**
The most immediately practical extension is to use geodesic deviation in Poincaré coordinates as a proxy for reasoning quality. We can define a geodesic score for a trajectory as the mean hyperbolic curvature along the path; lower scores indicate more geodesic-like (presumably more valid) reasoning. Testing whether this score correlates with factual accuracy on held-out QA benchmarks would directly test the practical utility of the hyperbolic geodesic hypothesis.

**Scale Invariance Study (0.5B to 70B).**
Replicating the full analysis across model scales — Qwen2.5-0.5B, 1.5B, 7B, 72B; Llama-3 8B, 70B — would test whether the ~41% curvature reduction is scale-invariant or whether it changes systematically with model size. A hypothesis worth testing is that larger models, which reason more accurately, show even higher reductions (i.e., their reasoning trajectories are even more geodesic-like).

**Out-of-Distribution Generalization.**
Testing on OOD problems — logic types, topics, or languages not seen in training — would reveal whether geodesic deviation predicts OOD failure modes. If in-distribution reasoning is approximately geodesic and OOD reasoning deviates substantially, this could provide a geometry-based OOD detector.

**Hyperbolic LoRA Fine-Tuning.**
The most ambitious extension is to modify the fine-tuning process itself to operate in hyperbolic space. LoRA (Hu et al., 2022) fine-tuning could be extended to use Riemannian SGD on the Poincaré manifold, updating weight matrices via exponential map steps rather than Euclidean gradient descent. The hypothesis is that if the natural geometry is hyperbolic, training in that geometry will produce models with straighter reasoning trajectories and, consequently, better generalization on logical reasoning benchmarks.

---

## 8. Conclusion

This report has presented a replication and extension of Zhou et al.'s ICLR 2026 framework for analyzing LLM reasoning through differential geometry. The replication confirms that hidden-state trajectories during chain-of-thought reasoning are organized in representation space by logic type at the velocity level (Order-1) and above, while raw positions (Order-0) reflect surface semantics. This finding generalizes across 20 topics and 4 languages with Qwen2.5-0.5B.

Our hyperbolic extension introduces the Geodesic Hypothesis: reasoning trajectories are intrinsically straight paths (geodesics) in hyperbolic space that appear curved only because Euclidean coordinates are the wrong coordinate system for the representational manifold. After resolving the boundary collapse problem via L2 normalization, we demonstrate a consistent 40.6% reduction in Menger curvature when measuring trajectories in Poincaré coordinates rather than Euclidean ones. This reduction is robust across all three logic types and all four languages.

Together, these results suggest that the geometry of LLM reasoning is not arbitrary — it has structure that is sensitive to both the choice of coordinate system and the level of geometric analysis. Future work on mechanistic interpretability, hallucination detection, and geometry-aware training may benefit from treating the representational manifold of LLM reasoning as a curved space rather than a flat one.

---

## References

1. **Zhou et al. (2026).** *The Geometry of Reasoning: Flowing Logics in Representation Space.* International Conference on Learning Representations (ICLR 2026). Duke University.

2. **Nickel, M., & Kiela, D. (2017).** *Poincaré Embeddings for Learning Hierarchical Representations.* Advances in Neural Information Processing Systems (NeurIPS 2017). pp. 6338–6347.

3. **Ganea, O., Bécigneul, G., & Hofmann, T. (2018).** *Hyperbolic Neural Networks.* Advances in Neural Information Processing Systems (NeurIPS 2018). pp. 5345–5355.

4. **Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E., Le, Q., & Zhou, D. (2022).** *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* Advances in Neural Information Processing Systems (NeurIPS 2022). pp. 24824–24837.

5. **Qwen Team. (2024).** *Qwen2.5 Technical Report.* arXiv preprint arXiv:2412.15115.

6. **Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022).** *LoRA: Low-Rank Adaptation of Large Language Models.* International Conference on Learning Representations (ICLR 2022).

7. **Chami, I., Ying, Z., Ré, C., & Leskovec, J. (2019).** *Hyperbolic Graph Convolutional Neural Networks.* Advances in Neural Information Processing Systems (NeurIPS 2019). pp. 4869–4880.

8. **Gromov, M. (1987).** *Hyperbolic Groups.* Essays in Group Theory. Mathematical Sciences Research Institute Publications, Springer. Vol. 8, pp. 75–263.
