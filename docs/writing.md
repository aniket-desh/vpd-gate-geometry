# Citations and LessWrong Writing Style Guide

## For the VPD gate-geometry LessWrong post

**Working post title:** *The Gate Geometry of Parameter Decompositions*

**Post type:** technical research note / mini-project writeup, not a polished paper and not a generic explainer.

**Audience:** Goodfire / Lee Sharkey stream readers, MATS reviewers, LessWrong/Alignment Forum readers interested in mechanistic interpretability, and technically literate people who have skimmed VPD/SPD but have not internalized the causal-importance field as an object of study.

---

## 1. Citation strategy

The post should cite enough to show real context, but not become a literature review. Use citations in four layers:

1. **Core object:** VPD and Goodfire's parameter-decomposition agenda.
2. **Predecessor method:** SPD/APD lineage.
3. **Polynomial / weight-based bridge:** bilinear MLPs, BAE, chi-nets.
4. **Conceptual neighbors:** SAEs/transcoders/activation-based decomposition only when motivating contrast.

Do not cite every interpretability paper you know. The signal is not bibliography size; it is having read the exact agenda deeply enough to ask a narrow question.

---

## 2. Minimal citation list

### 2.1 Essential citations

#### Interpreting Language Model Parameters / VPD

**Use for:** main method, causal importances, adversarial ablations, target model details, subcomponent counts, limitations about clustering and future work.

```bibtex
@misc{bushnaq2026interpretinglmparameters,
  title = {Interpreting Language Model Parameters},
  author = {Lucius Bushnaq and Dan Braun and Oliver Clive-Griffin and Bart Bussmann and Nathan Hu and Michael Ivanitskiy and Linda Linsefors and Lee Sharkey},
  year = {2026},
  url = {https://www.goodfire.ai/research/interpreting-lm-parameters}
}
```

Links:

- Goodfire paper/blog: https://www.goodfire.ai/research/interpreting-lm-parameters
- LessWrong linkpost: https://www.lesswrong.com/posts/eAQZaiC3PcBhS4HjM/linkpost-interpreting-language-model-parameters
- Explainer: https://www.goodfire.ai/research/vpd-explainer
- Code: https://github.com/goodfire-ai/param-decomp

Key facts to cite from it:

- VPD decomposes a four-layer, 67M parameter decoder-only transformer trained on an uncopyrighted Pile subset.
- The decomposed model has ~28M non-embedding parameters and 24 decomposed matrices.
- It uses 38,912 rank-one subcomponents, with ~9,972 alive.
- The method learns causal importances \(g^\ell_{b,t,c}\in[0,1]\).
- VPD uses adversarially selected ablation masks in addition to stochastic sampling.
- The authors explicitly state that their clustering currently uses same-position correlations and is blind to multi-sequence-position circuits.

#### Goodfire `param-decomp` repo

**Use for:** reproducibility, exact configs, canonical W&B run, postprocessing pipeline, smaller SimpleStories experiment.

```bibtex
@software{goodfire2026paramdecomp,
  title = {Parameter Decomposition},
  author = {{Goodfire AI}},
  year = {2026},
  url = {https://github.com/goodfire-ai/param-decomp}
}
```

Links:

- Repo: https://github.com/goodfire-ai/param-decomp
- Full Pile config: https://github.com/goodfire-ai/param-decomp/blob/main/param_decomp/experiments/lm/pile_llama_simple_mlp-4L.yaml
- SimpleStories config: https://github.com/goodfire-ai/param-decomp/blob/main/param_decomp/experiments/lm/ss_llama_simple_mlp-2L.yaml
- Harvest docs: https://github.com/goodfire-ai/param-decomp/blob/main/param_decomp/harvest/CLAUDE.md

#### Stochastic Parameter Decomposition

**Use for:** predecessor method and conceptual lineage.

```bibtex
@misc{bushnaq2025stochasticparameterdecomposition,
  title = {Stochastic Parameter Decomposition},
  author = {Lucius Bushnaq and Dan Braun and Lee Sharkey},
  year = {2025},
  eprint = {2506.20790},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url = {https://arxiv.org/abs/2506.20790}
}
```

Links:

- arXiv: https://arxiv.org/abs/2506.20790
- LessWrong linkpost: https://www.lesswrong.com/posts/yjrpmCmqurDmbMztW/paper-stochastic-parameter-decomposition

#### Attribution-based Parameter Decomposition

**Use for:** original parameter-space decomposition framing; APD's MDL/minimality motivation; contrast with SPD/VPD scaling.

Find/cite from VPD or Lee's MATS page if needed. Do not overdo APD details unless you discuss the lineage.

Relevant LessWrong/AXRP discussion:

- AXRP episode on APD: https://www.lesswrong.com/posts/gnyna4Rb2S7KdzxvJ/41-lee-sharkey-on-attribution-based-parameter-decomposition

#### Bilinear MLPs enable weight-based mechanistic interpretability

**Use for:** weight-based / tensorized interpretability bridge; third-order tensor framing; low-rank spectral analysis from weights.

```bibtex
@misc{pearce2025bilinearmlps,
  title = {Bilinear MLPs enable weight-based mechanistic interpretability},
  author = {Michael T. Pearce and Thomas Dooms and Alice Rigg and Jose M. Oramas and Lee Sharkey},
  year = {2025},
  eprint = {2410.08417},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url = {https://arxiv.org/abs/2410.08417}
}
```

Links:

- arXiv: https://arxiv.org/abs/2410.08417
- ICLR page: https://iclr.cc/virtual/2025/poster/28827

#### Finding Manifolds With Bilinear Autoencoders

**Use for:** future-work bridge; quadratic/polynomial features; importance ordering; clustering; activation sparsity.

```bibtex
@misc{dooms2025findingmanifoldsbilinearautoencoders,
  title = {Finding Manifolds With Bilinear Autoencoders},
  author = {Thomas Dooms and Ward Gauderis},
  year = {2025},
  eprint = {2510.16820},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url = {https://arxiv.org/abs/2510.16820}
}
```

Links:

- arXiv: https://arxiv.org/abs/2510.16820
- Project page: https://tdooms.github.io/research/bae/
- Code: https://github.com/tdooms/bae

#### Compositionality Unlocks Deep Interpretable Models / chi-nets

**Use for:** global compositional weight-based interpretability, tensor-network/multilinear angle.

```bibtex
@misc{dooms2025compositionality,
  title = {Compositionality Unlocks Deep Interpretable Models},
  author = {Thomas Dooms and Ward Gauderis and Geraint A. Wiggins and Jose Oramas},
  year = {2025},
  eprint = {2504.02667},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url = {https://arxiv.org/abs/2504.02667}
}
```

Links:

- arXiv: https://arxiv.org/abs/2504.02667

#### Lee Sharkey MATS stream page

**Use for:** application strategy and why this post is stream-aligned.

```bibtex
@misc{mats2026sharkey,
  title = {Lee Sharkey at MATS: Summer 2026},
  author = {{MATS Program}},
  year = {2026},
  url = {https://www.matsprogram.org/stream/sharkey}
}
```

Key points to cite/keep in mind:

- stream focuses on mechanistic interpretability methods for reverse-engineering neural networks;
- Goodfire developed SPD/APD as alternatives to sparse dictionary learning;
- projects should be conceptually informed by SPD if not directly building on it;
- candidate criteria include scientific thinking, quantitative skills, engineering skill, interpretability prerequisites, safety interests, and conscientiousness.

---

## 3. Optional citations

Use only if relevant.

### Lee Sharkey technical note on bilinear layers

```bibtex
@misc{sharkey2023bilinear,
  title = {A technical note on bilinear layers for interpretability},
  author = {Lee Sharkey},
  year = {2023},
  eprint = {2305.03452},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url = {https://arxiv.org/abs/2305.03452}
}
```

Use if you want to say bilinear layers were proposed as easier-to-analyze nonlinear layers expressible by third-order tensors.

### Earlier bilinear decomposition paper

```bibtex
@misc{pearce2024weightbasedbilinear,
  title = {Weight-based Decomposition: A Case for Bilinear MLPs},
  author = {Michael T. Pearce and Thomas Dooms and Alice Rigg},
  year = {2024},
  eprint = {2406.03947},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url = {https://arxiv.org/abs/2406.03947}
}
```

Use if you want the earlier framing before the ICLR version.

### Goodfire neural geometry post

Use if you write a future-work paragraph about manifolds and SAE shattering. Do not cite if the current post never discusses manifolds beyond a sentence.

- https://www.goodfire.ai/research/the-world-inside-neural-networks

### Transcoders / CLTs / activation-based baselines

Only cite if you explicitly compare VPD to activation-based decomposition. Otherwise, keep the post narrow.

---

## 4. LessWrong / Goodfire account notes

### 4.1 Goodfire / VPD LessWrong presence

The VPD LessWrong linkpost is short, direct, and high-context. It starts by identifying the work as the latest in the parameter-decomposition agenda, names the new method, says what it improves over, and emphasizes why attention decomposition and adversarial ablations matter. It also includes a compact abstract and links to the paper, explainer, and Twitter thread.

Takeaways for your post:

- start with the object and contribution quickly;
- do not bury the result under generic motivation;
- include links to code and figures;
- use footnotes for caveats and jokes, not long asides;
- distinguish speculation from result.

VPD linkpost:

- https://www.lesswrong.com/posts/eAQZaiC3PcBhS4HjM/linkpost-interpreting-language-model-parameters

### 4.2 Lee Sharkey's LessWrong account

Lee's profile identifies him as Goodfire/London, formerly Apollo, with interests in mechanistic interpretability and inner alignment. His top posts include Apollo Research, "Mech interp is not pre-paradigmatic", SAE/superposition work, polytope lens, APD, and VPD.

Profile:

- https://www.lesswrong.com/users/lee_sharkey

Style takeaways:

- research-program framing matters;
- mechanistic interpretability is presented as a young empirical science, not just pretty demos;
- claims are usually tied to methods and failure modes;
- minimality / MDL / basis-finding are recurring conceptual themes;
- serious posts often say what remains uncertain.

### 4.3 Lucius Bushnaq's LessWrong account

Lucius's profile and comments are useful because he engages technical objections directly. In the SPD comment thread, he clarifies that parameter decomposition is not merely a decomposition of linear maps and explains why nonlinearities and data-dependent causal usage matter.

Profile:

- https://www.lesswrong.com/users/lblack

Style takeaways:

- respond to likely objections in the post before comments force them;
- define exactly what the method can and cannot infer;
- avoid vague "superposition" language unless you specify activation-space vs parameter-space objects;
- be precise about when coactivation is evidence vs when it is only a proxy.

### 4.4 Oliver Clive-Griffin / other coauthors

Oliver's profile confirms the Goodfire VPD linkpost presence, but there is less stylistic signal available from the profile itself. Use the VPD post and Goodfire blog as the main style reference.

Profile:

- https://www.lesswrong.com/users/oliver-clive-griffin

---

## 5. Recommended post structure

### Title

Use a title that clearly states the object.

Best options:

1. **The Gate Geometry of Parameter Decompositions**
2. **A First Look at VPD's Causal-Importance Geometry**
3. **Do VPD Gates Contain Mechanism Clusters?**
4. **Lagged Co-Importance in Parameter Decompositions**
5. **Parameter Atoms, Gate Fields, and Cross-Token Circuits**

Best overall: **The Gate Geometry of Parameter Decompositions**.

It sounds conceptual but not overclaimy.

### Subtitle

> VPD decomposes model weights into rank-one subcomponents. I looked at the geometry of when those subcomponents are causally important.

### TL;DR

Use 3--5 bullets.

Example:

```markdown
TL;DR:

- VPD produces a causal-importance field \(g^\ell_{b,t,c}\), which says when rank-one parameter subcomponents are needed on each token.
- I treat this field as a geometric object and compute co-importance kernels between subcomponents.
- A token-identity baseline separates lexical gate structure from residual/contextual structure.
- Lagged co-importance probes cross-position circuits that same-token clustering can miss.
- This is not a replacement for VPD attribution graphs; it is a cheap diagnostic for how parameter atoms organize into candidate mechanisms.
```

### Section outline

```markdown
# The Gate Geometry of Parameter Decompositions

## TL;DR

## Why look at gates?

## VPD in the minimum necessary detail

## The causal-importance field

## Experiment 1: same-position co-importance

## Experiment 2: subtracting token identity

## Experiment 3: lagged co-importance

## A small case study

## What I think this means

## Limitations

## Future work: bilinear gates and variable-resolution VPD

## Appendix: methods and reproducibility
```

---

## 6. Writing style rules

### 6.1 Voice

Aim for:

- technically precise;
- direct;
- humble;
- empirically grounded;
- excited but not promotional;
- "here is a concrete object and a few probes" rather than "here is a grand theory."

Avoid:

- over-poetic opening;
- Em-dashes
- generic AI safety throat-clearing;
- exaggerated claims like "we solve clustering";
- too much personal application strategy;
- phrases that sound LLM-generated: "delve", "unlock", "revolutionize", "this has profound implications".

### 6.2 Good first paragraph

```markdown
VPD gives us two things. The first is the obvious one: a decomposition of model weights into rank-one parameter subcomponents. The second is easier to treat as just an implementation detail: a causal-importance field \(g^\ell_{b,t,c}\), saying which subcomponents are needed on which token positions. This post is about the second object.
```

This is strong because it immediately identifies a neglected object.

### 6.3 Bad first paragraph

```markdown
Mechanistic interpretability is one of the most important problems in AI safety. As models become more capable, we need to understand their inner workings. Recently, Goodfire published an exciting paper called Interpreting Language Model Parameters. In this post I will explore...
```

This is too generic. Everyone has read this opening.

### 6.4 Use caveats as credibility

Good caveat:

```markdown
Co-importance is not causal attribution. Two subcomponents can be important on the same tokens because they belong to the same mechanism, because they both respond to the same lexical artifact, or because two unrelated mechanisms are correlated in the dataset. The point of the token-residualized and lagged analyses below is to start separating these cases.
```

Bad caveat:

```markdown
Of course, much more work remains to be done.
```

The good caveat names the actual failure modes.

### 6.5 Keep math local and useful

LessWrong readers tolerate math, but every equation should earn its place. Define only the objects you use in plots.

Good:

```markdown
I flatten batch and position into a single index \(n=(b,t)\), giving a matrix
\[
G^\ell_{n,c}=g^\ell_{b,t,c}.
\]
The simplest co-importance kernel is then
\[
K^\ell_{c,c'}=\frac{\langle G^\ell_c,G^\ell_{c'}\rangle}{\|G^\ell_c\|\|G^\ell_{c'}\|}.
\]
```

Bad:

```markdown
Let \(\mathcal{M}\) be a smooth manifold of mechanisms embedded in a high-dimensional Hilbert space...
```

Unless you actually use that structure, don't introduce it.

---

## 7. How to frame the contribution

### 7.1 One-sentence contribution

> I analyze VPD's causal-importance tensor as a first-class geometric object and show how same-position, token-residualized, and lagged co-importance reveal different candidate groupings of parameter subcomponents.

### 7.2 More application-oriented framing

> If VPD gives rank-one parameter atoms, we still need to know how atoms assemble into mechanisms. Gate geometry is a cheap diagnostic for that assembly problem.

### 7.3 More research-program framing

> This is a small step toward replacing post-hoc component clustering with a geometry-aware account of parameter-subcomponent usage across data and sequence positions.

### 7.4 Avoid this framing

> I extend VPD with a new clustering method.

Unless you actually evaluate it against VPD's clustering and reconstruction metrics, this is too strong.

---

## 8. Where to place the BAE/Dooms connection

Do not make the post primarily about BAE unless you actually train one. Instead:

- mention BAE in the motivation/future-work section;
- say spectral gate geometry is the linear baseline;
- say BAE-on-gates is a natural nonlinear continuation.

Suggested paragraph:

```markdown
This also clarifies what I would want from a bilinear-autoencoder version of this project. A BAE over the gate vectors would not just be "a fancier clustering method"; it would look for quadratic factors \(g_cg_{c'}\), i.e. pairwise usage patterns of parameter atoms. But before adding that complexity, I wanted to know what the linear geometry already says.
```

This is a mature framing.

---

## 9. Figure style tips

### 9.1 General

- Use 4--5 figures max in the main post.
- Put extra plots in appendix or GitHub.
- Every figure should answer one question.
- Captions should state the takeaway, not merely describe the axes.

### 9.2 Figure sequence

Recommended:

1. **Schematic:** model weights -> rank-one subcomponents -> gate field -> kernels.
2. **Spectrum:** gate co-importance has / lacks low-rank structure.
3. **Token residualization:** how much raw geometry is lexical.
4. **Lagged co-importance:** cross-position relationships.
5. **Case study:** one cluster/pair with tokens/context examples.

### 9.3 Color

Use muted colors. Avoid rainbow heatmaps unless necessary. For heatmaps, use a perceptually uniform sequential or diverging palette.

Suggested semantic palette:

- raw gates: muted purple;
- token baseline: muted orange;
- residual gates: teal;
- lagged relation: green;
- baselines/noise: gray.

### 9.4 Captions

Good caption:

> **Figure 3. Token identity explains a large fraction of some components' gate variance, but not all of it.** Each point is a layer-2 MLP-down subcomponent. The x-axis is mean causal importance; the y-axis is \(R^2\) under a token-only baseline. High-activity components are not uniformly lexical, which motivates residualized clustering.

Bad caption:

> Token residualization results.

---

## 10. Suggested wording for limitations

```markdown
## Limitations

The main limitation is that co-importance is not causality. VPD's attribution graphs are closer to causal explanations of particular forward passes; the kernels here are distributional summaries of when subcomponents are needed. A high edge in a gate kernel can mean that two subcomponents participate in one mechanism, but it can also mean that two mechanisms are correlated in the dataset.

The second limitation is that token residualization is a crude baseline. It removes first-order lexical identity, but not topic, syntax, document source, or longer n-gram effects. This means that surviving residual structure is more contextual than raw structure, but not automatically mechanistic.

The third limitation is scale. I analyze [N] sequences / [modules] rather than every possible module pair over the full training distribution. The goal is to test whether this diagnostic is useful, not to produce a finished clustering of the whole model.
```

Fill in `[N]` and `[modules]` honestly.

---

## 11. Suggested wording for future work

```markdown
## Future work

The obvious next step is to compare gate-geometry clusters to VPD's attribution graphs. If a group of subcomponents coactivates across the dataset, do those same subcomponents form connected attribution subgraphs on individual prompts?

A second step is to replace linear co-importance with a bilinear model over gates. Bilinear autoencoders are attractive here because the natural object is pairwise usage \(g_cg_{c'}\): not just whether a parameter atom is active, but whether two atoms are jointly needed. This would turn the present analysis into a baseline for polynomial mechanism-bundle extraction.

A third step is to integrate clustering into VPD training. The current VPD paper notes that post-hoc clustering can make adversarial sampling stricter than necessary. If gate geometry identifies stable bundles, future methods could sample masks over bundles rather than only over rank-one atoms.
```

---

## 12. Comment-response preparation

### Objection 1: "Isn't this just clustering activations?"

Answer:

> It is clustering an activation-like object, but the columns are parameter subcomponents and the values are trained causal importances under adversarial reconstruction constraints. That makes the object different from residual-stream activations or SAE latents. I agree the statistical tools are similar; the interpretation of the columns is not.

### Objection 2: "Coactivation is not mechanism membership."

Answer:

> Agreed. The post explicitly treats coactivation as a diagnostic, not a causal proof. That is why I include token residualization and lagged analyses, and why I distinguish this from VPD attribution graphs.

### Objection 3: "Token residualization removes real mechanisms that are token-specific."

Answer:

> Yes. The goal is not to declare token-specific structure unmechanistic. It is to separate lexical-support structure from residual/contextual structure so we can inspect both. A component that vanishes under token residualization may still be a real token-specific mechanism.

### Objection 4: "Why not just use VPD's own clustering?"

Answer:

> VPD's clustering is the starting point. The paper itself notes that same-position clustering is blind to multi-position circuits and that clustering was not heavily tuned. This post explores what additional geometry is available in the same causal-importance field.

### Objection 5: "Why not train a BAE?"

Answer:

> I think that is the natural next step. But before training a nonlinear model over gates, I wanted the linear/spectral baseline. Otherwise it is hard to know whether BAE is finding new structure or rediscovering simple coactivation geometry.

---

## 13. Tags and venue settings

Suggested LessWrong tags:

- Interpretability (ML & AI)
- Mechanistic Interpretability
- Goodfire
- Sparse Autoencoders (SAEs), only if you discuss contrast
- Transformers
- AI
- MATS Program, if the post is relevant and not too self-promotional

Consider crossposting to Alignment Forum if the post has enough technical substance and safety relevance.

---

## 14. What the post should not include

Do not include:

- your MATS application motivation in the main text;
- long autobiography;
- claims about impressing Goodfire;
- speculative PhD/career framing;
- too much Claude/LLM process;
- full code dumps;
- generic intros to AI safety;
- a massive citation dump.

Put code in GitHub, not in the post, except for short equations/pseudocode.

---

## 15. Minimal final post draft skeleton

```markdown
# The Gate Geometry of Parameter Decompositions

VPD gives us two things. The first is the obvious one: a decomposition of model weights into rank-one parameter subcomponents. The second is easier to treat as just an implementation detail: a causal-importance field \(g^\ell_{b,t,c}\), saying which subcomponents are needed on which token positions. This post is about the second object.

## TL;DR

- ...

## Why look at gates?

Goodfire's VPD decomposes a 4-layer, 67M parameter language model into rank-one subcomponents...

## The object

For each layer/module, flatten batch and position into \(n\):
\[
G^\ell_{n,c}=g^\ell_{b,t,c}.
\]

## Experiment 1: same-position co-importance

Define:
\[
K^\ell_{c,c'}=\frac{\langle G_c,G_{c'}\rangle}{\|G_c\|\|G_{c'}\|}.
\]

[Figure 1]

Takeaway: ...

## Experiment 2: subtracting token identity

Fit:
\[
g_{b,t,c}\approx \mu_c+a_{z_{b,t},c}.
\]

[Figure 2]

Takeaway: ...

## Experiment 3: lagged co-importance

Define:
\[
K^{A\to B}_{c,c'}(\tau)=\operatorname{corr}(g^A_{b,t,c},g^B_{b,t+\tau,c'}).
\]

[Figure 3]

Takeaway: ...

## Case study

[Cluster/pair examples]

## Interpretation

VPD gives rank-one atoms, but mechanisms need not be rank-one...

## Limitations

...

## Future work

...

## Reproducibility

- Goodfire repo commit: ...
- W&B run: ...
- sequences analyzed: ...
- modules analyzed: ...
```

---

## 16. Final quality bar

Before posting, ask:

1. Can a reader state your contribution in one sentence?
2. Is every figure tied to a specific question?
3. Did you show one real empirical result, not just propose an idea?
4. Did you avoid claiming causality from correlation?
5. Did you explicitly connect to a limitation/open direction in VPD?
6. Did you cite the exact Goodfire code/config you used?
7. Would Lee/Goodfire see this as inside their agenda rather than adjacent fanfiction?
8. Would a skeptical reader think, "this was small, but real"?

If yes, publish.

