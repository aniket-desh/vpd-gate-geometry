# The Gate Geometry of Parameter Decompositions

## A theory breakdown for a 4--5 day LessWrong mini-project

**Working title:** *The Gate Geometry of Parameter Decompositions: do VPD causal-importance fields contain mechanism clusters?*

**Target audience:** Lee Sharkey / Goodfire / MATS readers who already know the motivation for parameter decompositions, but have not yet thought of the VPD causal-importance tensor itself as a geometric object worth analyzing.

**Core claim:** VPD does not only produce rank-one parameter subcomponents. It also produces a learned, per-token *causal-importance field*

\[
g^\ell_{b,t,c}\in [0,1],
\]

where \(\ell\) indexes a decomposed weight matrix or layer/module group, \(b\) indexes batch item, \(t\) indexes sequence position, and \(c\) indexes a rank-one parameter subcomponent. This field is an activation-like trace over parameter-space primitives. The mini-project asks whether this field already contains interpretable geometry: low-rank structure, token-identity artifacts, same-position mechanism clusters, and cross-position circuit couplings.

The project is deliberately framed as a **linear / spectral baseline** for a later bilinear-autoencoder-on-gates project. It should be small enough to execute quickly, but serious enough that a Goodfire/MATS reader sees an actual research instinct: identify an object the original paper created, analyze it in its native geometry, and use failures of the baseline to motivate the next method.

---

## 1. Why this project exists

### 1.1 The strategic reason

Your target stream is Lee Sharkey / Goodfire / parameter and weight-based interpretability. Lee's current MATS stream states that projects should be at least conceptually informed by SPD, and Goodfire's recent VPD paper is now the canonical object in this agenda. That means a good application-signal post should not merely summarize SAEs, activation patching, or generic circuits; it should show that you can think *inside* the parameter-decomposition research program.

The prior brainstorm correctly identified the strongest conceptual bridge: SPD/VPD gives rank-one parameter primitives and Dooms-style bilinear/polynomial methods give compositional structure extractors over nonlinear or manifold-valued objects. In particular, the idea "BAE on SPD/VPD gates" is attractive because the VPD gate field \(g^\ell_{b,t,c}\) has exactly the form of a high-dimensional activation trace over parameter atoms. Treating it as an analyzable signal can reveal mechanism bundles, token artifacts, and cross-position coupling patterns before one modifies the decomposition objective itself.

But a full bilinear autoencoder project is risky in a 2--3 committed day window. The right mini-project is therefore:

> First build the spectral/tensor baseline that any future bilinear gate method would need to beat.

This is a recognizable research move. It is also an honest one.

### 1.2 The scientific reason

VPD currently has a clustering step that groups rank-one subcomponents into fuller parameter components. The original VPD paper explicitly says that this clustering is based on causal-importance correlations at the same sequence position and is blind to multi-position circuits. For example, query and key subcomponents in an induction-like circuit may not be causally important at the same token position. That means VPD has already exposed an open technical question:

> What is the right geometry for grouping parameter subcomponents into mechanisms?

This project studies the first-order version of that question.

---

## 2. Background: parameter decomposition in one page

### 2.1 Activation-based vs parameter-based decompositions

Activation-based methods, such as SAEs, transcoders, crosscoders, and cross-layer transcoders, decompose *activation trajectories* or learned activation-to-activation maps. They often provide useful explanations of intermediate representations, but they introduce learned auxiliary models whose functional form differs from the original network.

Parameter decomposition asks for a decomposition of the actual model parameters:

\[
\theta \approx \sum_i \theta_i,
\]

where each \(\theta_i\) is intended to correspond to a mechanism, or at least to a simple parameter-space primitive from which mechanisms can be assembled.

The philosophical attraction is that this moves the interpretability object closer to the thing actually doing computation: the weights and nonlinearities of the original model.

### 2.2 VPD's decomposition object

For a target weight matrix \(W_\ell\), VPD parameterizes subcomponents as rank-one matrices:

\[
W_\ell \approx \sum_{c=1}^{C_\ell} U^\ell_c (V^\ell_c)^\top.
\]

Each subcomponent has a "write" vector \(U^\ell_c\) and a "read" vector \(V^\ell_c\). In the simplest mental model, when the layer sees an activation \(x\), the subcomponent contributes

\[
U^\ell_c (V^\ell_c)^\top x
= U^\ell_c \langle V^\ell_c, x\rangle.
\]

This is why rank-one parameter atoms are appealing: each subcomponent reads one direction and writes one direction. This is mathematically clean, though not automatically mechanistically complete.

### 2.3 Parameter faithfulness

VPD introduces a residual or delta component

\[
\Delta^\ell := W_\ell - \sum_c U^\ell_c (V^\ell_c)^\top,
\]

so the decomposed system can remain parameter-faithful even when the rank-one dictionary does not perfectly reconstruct the original matrix without a residual.

The delta component is trained to be small / causally unimportant. For the mini-project, the important point is that the rank-one atoms are not merely a post-hoc SVD of \(W_\ell\); they are trained jointly with causal-importance predictors and reconstruction losses.

### 2.4 Minimality and causal importance

VPD trains a causal-importance function \(\Gamma\) that outputs

\[
g^\ell_{b,t,c}\in [0,1].
\]

Interpretation:

- \(g^\ell_{b,t,c}\approx 0\): subcomponent \((\ell,c)\) should be ablatable on token \((b,t)\) without changing the model's output much.
- \(g^\ell_{b,t,c}\approx 1\): subcomponent \((\ell,c)\) is causally important for that token's output.

VPD minimizes an importance-minimality loss of the schematic form

\[
\mathcal{L}_{\mathrm{importance}}=
\frac{1}{BT}\sum_{b,t,\ell,c}|g^\ell_{b,t,c}|^p,
\]

so it prefers short explanations: few important subcomponents per token.

### 2.5 Mechanistic faithfulness and adversarial masks

Given causal importances, VPD samples or optimizes ablation masks

\[
m^\ell_{b,t,c}\in[g^\ell_{b,t,c},1].
\]

If \(g=1\), the mask cannot ablate the subcomponent. If \(g=0\), the mask may ablate it completely or partially. A masked weight matrix is then used in a replacement forward pass, and the decomposed model is penalized if its output distribution diverges from the target model.

The distinctive VPD move is adversarial masking: instead of only sampling masks stochastically, VPD uses projected gradient descent to search for masks that maximally break reconstruction. This makes the faithfulness criterion stricter.

For this project, the important conceptual fact is:

> The causal-importance field \(g\) is trained to be sparse but adversarially sufficient. Therefore, co-activation patterns in \(g\) are plausible evidence about which parameter subcomponents jointly participate in computation.

They are not guaranteed to be ground-truth mechanisms, but they are not arbitrary activations either.

---

## 3. The new object: the VPD gate field

### 3.1 Flattening the gate tensor

Fix a layer/module group \(\ell\). Define

\[
G^\ell \in \mathbb{R}^{N\times C_\ell},
\qquad
G^\ell_{n,c}=g^\ell_{b(n),t(n),c},
\]

where \(n=(b,t)\) flattens batch and token position.

\(G^\ell\) is the central data matrix of the project.

Rows are token instances.
Columns are rank-one subcomponents.
Entries are causal importances.

This is formally analogous to a feature activation matrix in SAE analysis, but its interpretation is different: columns are not activation features; they are learned parameter atoms whose ablatability has been trained to preserve the target model's behavior.

### 3.2 The gate field as a sparse code over parameter atoms

For a token instance \(n\), define the active set

\[
S_n^\ell = \{c: g^\ell_{n,c}>\tau\}.
\]

VPD wants \(|S_n^\ell|\) to be small while preserving model behavior. Thus each row of \(G^\ell\) is a sparse description of which rank-one parameter atoms are needed for that token.

That makes \(G^\ell\) a *description matrix*:

- row sparsity measures description length per token;
- column frequencies measure how broadly useful a subcomponent is;
- column co-activation measures candidate mechanism bundling;
- cross-position correlations measure candidate temporal / attention circuits.

### 3.3 Why co-activation is not enough, but still the right first baseline

If two subcomponents are often important on the same token positions, they may be part of the same mechanism. But this implication is imperfect.

Same-position co-activation can conflate:

1. real mechanism membership;
2. token identity artifacts;
3. topic/domain artifacts;
4. syntax or formatting artifacts;
5. generic always-on computations;
6. downstream effects of another earlier computation;
7. correlated but distinct mechanisms.

The mini-project should explicitly say this. The post's contribution is not "cosine similarity solves component clustering." The contribution is:

> The VPD gate field is a new empirical object. Same-position spectral geometry, token-residualized geometry, and lagged geometry are the first sanity checks for whether this object contains mechanism-level structure.

---

## 4. First analysis: same-position co-importance geometry

### 4.1 The raw co-importance kernel

For each layer/module \(\ell\), define centered or uncentered component vectors

\[
G_c^\ell = (g^\ell_{1,c},\ldots,g^\ell_{N,c})\in\mathbb{R}^{N}.
\]

The simplest co-activation kernel is cosine similarity:

\[
K^\ell_{c,c'} =
\frac{\langle G_c^\ell,G_{c'}^\ell\rangle}
{\|G_c^\ell\|_2\|G_{c'}^\ell\|_2 + \epsilon}.
\]

Alternative kernels:

- Pearson correlation:

\[
K^\ell_{c,c'} =
\operatorname{corr}(G_c^\ell,G_{c'}^\ell).
\]

- Jaccard similarity after thresholding:

\[
K^\ell_{c,c'} =
\frac{|S_c\cap S_{c'}|}{|S_c\cup S_{c'}|}.
\]

- Pointwise mutual information-like score:

\[
\operatorname{PMI}(c,c')=
\log\frac{\Pr(c,c')}{\Pr(c)\Pr(c')}.
\]

The raw cosine kernel is the easiest to implement and visualize. The Jaccard / PMI variants are useful if the gate field is extremely sparse.

### 4.2 Spectral structure

Compute the eigenspectrum of \(K^\ell\):

\[
K^\ell = Q\Lambda Q^\top.
\]

Questions:

1. Is there a steep spectral decay?
2. Are there outlier eigenvalues?
3. Do top eigenvectors localize on interpretable groups of subcomponents?
4. Are spectra different for attention vs MLP matrices?
5. Are spectra different across layers?

A steep spectrum suggests the gate field has a low-dimensional mechanism-bundle structure. A flat spectrum suggests highly distributed, fragmented, or noisy usage.

### 4.3 Clustering from the kernel

Given \(K^\ell\), cluster components using one or more simple methods:

- spectral clustering on top \(r\) eigenvectors;
- agglomerative clustering with distance \(1-K\);
- Leiden/Louvain on a thresholded graph;
- connected components after thresholding high-similarity edges.

The mini-project should prefer robustness over sophistication:

> If a cluster only appears for one very specific threshold, do not oversell it.

Good outputs:

- a clustered heatmap of \(K\);
- a scree plot of eigenvalues;
- top token examples per cluster;
- top subcomponents per cluster by mean causal importance;
- cluster-level activity traces across tokens.

### 4.4 Cluster-level gates

For a cluster \(A\subseteq\{1,\ldots,C_\ell\}\), define a cluster gate:

\[
h_A(n)=\sum_{c\in A} g^\ell_{n,c}
\quad\text{or}\quad
h_A(n)=\max_{c\in A}g^\ell_{n,c}.
\]

Then analyze:

- examples where \(h_A\) is high;
- token distribution where \(h_A\) is high;
- nearby context windows;
- whether cluster members share Goodfire autointerp labels if harvested labels exist;
- whether cluster members are in the same or different modules.

The post should include at least one qualitative cluster case study, even if short.

---

## 5. Second analysis: token identity residualization

### 5.1 Motivation

A dangerous failure mode is that gate geometry may mostly reflect lexical identity.

Example: some subcomponents may fire whenever the token is `:` or `)` because those subcomponents are useful for emoticon prediction. That can be mechanistically real, but it can also cause clustering to overstate mechanism structure: it might group everything that happens to be useful on a token type, not everything that belongs to a coherent computation.

Token residualization asks:

> How much of the gate field is explained by token identity alone?

This is directly inspired by tokenized SAE logic: remove trivial token-level reconstruction variance before interpreting the remaining features.

### 5.2 The simplest model

Let \(z_{b,t}\) be the token ID at sequence position \((b,t)\). Fit a token-only baseline for each component:

\[
g^\ell_{b,t,c}\approx \mu^\ell_c + a^\ell_{z_{b,t},c}.
\]

For frequent tokens, estimate

\[
\hat a^\ell_{z,c}=\mathbb{E}[g^\ell_{b,t,c}\mid z_{b,t}=z]-\mu^\ell_c.
\]

Then define residual gates:

\[
\tilde g^\ell_{b,t,c} = g^\ell_{b,t,c} - \hat\mu^\ell_c - \hat a^\ell_{z_{b,t},c}.
\]

Use shrinkage for rare tokens:

\[
\hat a_{z,c}=\frac{n_z}{n_z+\lambda}
\left(\bar g_{z,c}-\bar g_c\right).
\]

This avoids overfitting rare tokens.

### 5.3 Optional richer baselines

If time permits:

- previous-token identity baseline:

\[
g_{b,t,c}\approx \mu_c + a_{z_{b,t},c}+b_{z_{b,t-1},c};
\]

- local n-gram baseline;
- position baseline;
- document/source baseline, if metadata is available;
- topic baseline using simple embedding clusters.

But the post should not depend on these.

### 5.4 Explained variance

Report:

\[
R^2_c = 1-\frac{\sum_n(g_{n,c}-\hat g_{n,c})^2}{\sum_n(g_{n,c}-\bar g_c)^2}.
\]

Then aggregate across components:

- mean \(R^2\);
- weighted mean \(R^2\) weighted by mean causal importance;
- histogram of \(R^2_c\);
- top token-explained components;
- least token-explained components.

Interpretation:

- High token-explained variance means raw clusters may be lexical.
- Low token-explained variance means gate geometry reflects more contextual / computational structure.
- A mixed result is likely and interesting.

### 5.5 Residualized kernel

Compute

\[
\tilde K^\ell_{c,c'} =
\frac{\langle \tilde G_c^\ell,\tilde G_{c'}^\ell\rangle}
{\|\tilde G_c^\ell\|_2\|\tilde G_{c'}^\ell\|_2+\epsilon}.
\]

Compare raw and residualized clusters:

- Adjusted Rand Index between clusterings;
- overlap of top edges;
- change in eigenspectrum;
- examples of clusters that survive;
- examples of clusters that dissolve.

A strong LessWrong result could simply be:

> Many visually compelling raw gate clusters disappear after token residualization. The surviving clusters are therefore much better candidates for mechanisms.

That is useful even if it is deflationary.

---

## 6. Third analysis: lagged co-importance and cross-position circuits

### 6.1 Motivation

VPD's own discussion says same-position clustering can miss cross-position circuits. Query and key subcomponents in attention may operate at different positions: the query is important at the destination token; the key/value may be important at a source token.

Therefore, define lagged co-importance:

\[
K^\ell_{c,c'}(\tau)=
\mathbb{E}_{b,t}\left[g^\ell_{b,t,c}\,g^{\ell'}_{b,t+\tau,c'}\right]
\]

or normalized:

\[
K^\ell_{c,c'}(\tau)=
\frac{\mathbb{E}_{b,t}[(g^\ell_{b,t,c}-\bar g_c)(g^{\ell'}_{b,t+\tau,c'}-\bar g_{c'})]}
{\sigma_c\sigma_{c'}+\epsilon}.
\]

Here \(\tau\) can be positive or negative. Same-position clustering is the special case \(\tau=0\).

### 6.2 Within-module vs cross-module lagged kernels

The most useful pairs are likely:

- attention query \(Q\) at destination vs key \(K\) at source;
- attention output \(O\) at destination vs value \(V\) at source;
- MLP output at one layer vs attention query/key in a later layer;
- layer \(\ell\) subcomponents at token \(t\) vs layer \(\ell+1\) subcomponents at token \(t\) or \(t+\tau\).

Define module-pair kernels:

\[
K^{A\to B}_{c,c'}(\tau)
=\operatorname{corr}\left(g^A_{b,t,c},g^B_{b,t+\tau,c'}\right).
\]

This turns the gate field into a temporal interaction graph over parameter subcomponents.

### 6.3 Lag profiles

For a pair \((c,c')\), define the lag profile

\[
\tau\mapsto K_{c,c'}(\tau).
\]

A pair with peak at \(\tau=0\) is same-token coupled.
A pair with peak at \(\tau=-1\) or \(+1\) may be previous-token behavior.
A pair with broader peaks may reflect delimiters, syntactic boundaries, or attention over longer contexts.

Good visual:

- small multiples of lag profiles for top component pairs;
- heatmap with x-axis \(\tau\), y-axis top pair index;
- module-pair summary matrix where each cell reports max over \(\tau\neq 0\).

### 6.4 Cross-position mechanism graph

Build a graph:

- nodes: subcomponents \((\ell,\mathrm{module},c)\);
- edges: high lagged correlation at some \(\tau\);
- edge labels: \(\tau^*\), correlation strength, sign.

This graph is not an attribution graph. It is a co-importance graph. The post should clearly distinguish them.

Possible phrasing:

> VPD attribution graphs ask which subcomponents causally route information to which others on a prompt. Gate-lag graphs ask which subcomponents tend to be *needed together* across token offsets in the data distribution.

The two should eventually be compared, but this project only builds the latter.

---

## 7. Fourth analysis: importance ordering / variable-resolution descriptions

### 7.1 Motivation

The VPD paper explicitly points toward variable-resolution descriptions: instead of a binary important/unimportant decision, one might want a Pareto frontier between description length and reconstruction quality.

This project can include a lightweight version without retraining:

> Rank subcomponents by a global importance score derived from the gate field and ask whether prefixes preserve most gate mass / cluster mass / attribution mass.

### 7.2 Candidate ranking scores

For component \(c\):

1. Mean causal importance:

\[
s_c=\mathbb{E}_{b,t}g_{b,t,c}.
\]

2. Gate energy:

\[
s_c=\mathbb{E}_{b,t}g_{b,t,c}^2.
\]

3. Sparse support frequency:

\[
s_c=\Pr(g_{b,t,c}>\tau).
\]

4. Residualized energy:

\[
s_c=\mathbb{E}_{b,t}\tilde g_{b,t,c}^2.
\]

5. Spectral centrality in co-importance graph:

\[
s = \operatorname{eigencentrality}(K_+).
\]

6. Cross-lag centrality:

\[
s_c=\sum_{c',\tau\neq 0}|K_{c,c'}(\tau)|.
\]

The post should not claim any of these is the correct importance ordering. It should present them as probes.

### 7.3 Prefix curves

For a ranking \(\pi\), define prefix \(P_k=\{\pi(1),\ldots,\pi(k)\}\). Plot:

- fraction of total gate mass captured:

\[
M(k)=\frac{\sum_{n,c\in P_k}g_{n,c}}{\sum_{n,c}g_{n,c}};
\]

- fraction of residualized gate energy captured;
- fraction of high-similarity graph edges preserved;
- optional reconstruction KL if you implement masked forward passes.

This connects the project to BAE's analytic importance ordering without requiring BAE training.

---

## 8. Relation to bilinear MLPs and bilinear autoencoders

### 8.1 Bilinear MLPs

Bilinear MLP work shows that some neural layers can be rewritten as third-order tensors and then studied by spectral methods. The conceptual parallel is:

- bilinear MLP: weight computation has a tensor form;
- VPD gate geometry: subcomponent usage has a tensor form.

The object here is not the same as the bilinear MLP tensor. But both are attempts to make weight-based computation analyzable with linear algebra.

### 8.2 Bilinear autoencoders

BAEs reconstruct the quadratic input space, exposing polynomial factors that can be linearly analyzed. The natural future project is:

\[
G_n \mapsto \text{quadratic features } G_{n,c}G_{n,c'}.
\]

This could reveal pairwise mechanism bundles more directly than a linear coactivation kernel.

However, BAE-on-gates should come after the spectral baseline, because:

1. If linear gate geometry is already clean, BAE may be unnecessary for the first post.
2. If linear gate geometry fails in specific ways, that failure motivates BAE precisely.
3. A spectral baseline is easier to audit and harder to dismiss as black-box overfitting.

### 8.3 Chi-nets / global compositional decompositions

Chi-net-style work aims for globally compositional, weight-based interpretability across layers. VPD currently gives subcomponents per matrix plus post-hoc clustering. Gate-lag analysis is a smaller version of global composition: it asks how usage patterns compose across modules and positions.

This is the strongest connection to your tensor-network background, but it must stay scoped.

Do not try to implement a global chi-net decomposition of all VPD atoms in the first post. Instead, write:

> This analysis is a data-driven first pass at the compositional structure that a future integrated tensor decomposition of VPD subcomponents would need to explain.

---

## 9. Hypotheses

### H1: Raw gate co-importance is highly structured

Prediction: \(K\) has block structure and outlier eigenvalues, especially in attention modules and MLP down projections.

Interpretation if true: VPD gate fields may already encode mechanism bundles; post-hoc clustering can be improved with better geometry.

Interpretation if false: VPD subcomponents may be individually interpretable but not cleanly co-clustered by simple same-position usage.

### H2: Token identity explains a nontrivial but incomplete fraction of gate variance

Prediction: Some components are mostly token-identity-driven, but many meaningful clusters survive token residualization.

Interpretation if true: residualized gate geometry gives a sharper mechanism-clustering signal.

Interpretation if token identity explains almost everything: VPD may be finding many lexical support components; future causal-importance functions may need explicit token baselines.

Interpretation if token identity explains almost nothing: VPD gates are more contextual/computational than expected.

### H3: Lagged co-importance reveals structure missed by same-position clustering

Prediction: Some Q/K, V/O, and cross-layer pairs peak at \(\tau\neq 0\).

Interpretation if true: VPD's stated concern about same-position clustering is empirically important, and lagged kernels are a simple fix/probe.

Interpretation if false: same-position clustering may be less blind in practice than expected, at least for the analyzed model/data subset.

### H4: Different modules have different gate geometries

Prediction: MLP down projections may have more lexical or topic-specific clusters; attention Q/K/V/O may show stronger lagged structure.

Interpretation: parameter subcomponent geometry depends strongly on module type, so one-size-fits-all clustering is probably suboptimal.

---

## 10. What counts as a successful result?

A successful post needs one or two of the following:

1. A clean spectral plot showing nontrivial gate geometry.
2. A token-residualization plot showing which clusters survive lexical baselines.
3. A lagged co-importance plot showing cross-token relationships missed by \(\tau=0\).
4. One qualitative case study of a cluster or lagged pair using activation examples or autointerp labels.
5. A clear negative result that changes how one should interpret VPD clusters.

It does not need:

- a new VPD training objective;
- full BAE training;
- frontier-scale models;
- perfect causal validation;
- a claim to have solved component clustering.

The post should sound like:

> I found a useful diagnostic and ran it on the natural object VPD gives us.

Not:

> I invented the next generation of parameter decomposition.

---

## 11. Failure modes and how to write them honestly

### Failure mode 1: The gates are too sparse

If \(G\) is too sparse, cosine similarities may be unstable. Use Jaccard/PMI and restrict to alive components or components with enough support.

Write:

> For very low-frequency components, co-importance estimates are noisy. I therefore restrict most plots to components above a support threshold and treat rare-component clusters as exploratory.

### Failure mode 2: Token identity dominates

This is not a project failure. It becomes a result.

Write:

> A surprisingly large fraction of raw gate geometry is lexical. This suggests that VPD analyses should distinguish token-support components from contextual mechanism components before interpreting clusters.

### Failure mode 3: Lagged correlations are weak

Also not a failure. It says either cross-position structure is not well captured by marginal co-importance, or the analyzed subset/model does not show it strongly.

Write:

> The absence of strong lagged correlations in this simple statistic should not be read as absence of cross-position circuits. VPD attribution graphs remain a more causal object. The result only says that cross-position usage correlations are not large under this probe.

### Failure mode 4: The Goodfire codebase is hard to run

Use canonical W&B runs or the smaller SimpleStories experiment. The post can be framed as exploratory analysis of existing VPD artifacts.

### Failure mode 5: You cannot obtain raw \(g\) quickly

Fallback: use harvested component correlations and token stats from the postprocessing pipeline. Then the post becomes:

> A first look at the geometry already present in Goodfire's harvested component statistics.

This is less ideal but still usable.

---

## 12. What not to claim

Do not claim:

- coactivation equals mechanism membership;
- spectral clusters are ground truth circuits;
- token-residualized clusters are fully non-lexical;
- lagged correlation is causal attribution;
- VPD is solved;
- BAE is necessary before showing the baseline;
- this project improves VPD training unless you actually retrain/evaluate reconstruction.

Good claim:

> Gate geometry is a lightweight diagnostic layer over VPD that exposes which parts of the subcomponent organization are same-token, lexical, residual/contextual, or cross-position.

Better claim:

> The causal-importance field should be treated as a first-class object in parameter-decomposition research, not only as an internal routing variable.

---

## 13. Suggested final thesis for the post

A strong final paragraph could say:

> VPD gives us rank-one parameter atoms, but mechanisms are not generally rank-one. The missing object is the organization of those atoms across data, positions, and modules. The causal-importance field is the first place to look for that organization. Simple spectral probes already separate lexical from contextual structure and reveal whether same-position clustering is missing cross-token coupling. This is not a replacement for adversarial faithfulness or attribution graphs, but it is a cheap diagnostic for deciding where the next generation of parameter decomposition methods should spend their complexity.

---

## 14. Source notes

Primary sources to read before writing:

1. Goodfire, *Interpreting Language Model Parameters* — VPD paper/blog and implementation details.  
   https://www.goodfire.ai/research/interpreting-lm-parameters
2. Goodfire `param-decomp` repo — exact VPD code, canonical run, postprocessing pipeline, nano implementation.  
   https://github.com/goodfire-ai/param-decomp
3. Bushnaq, Braun, Sharkey, *Stochastic Parameter Decomposition*.  
   https://arxiv.org/abs/2506.20790
4. Pearce, Dooms, Rigg, Oramas, Sharkey, *Bilinear MLPs enable weight-based mechanistic interpretability*.  
   https://arxiv.org/abs/2410.08417
5. Dooms and Gauderis, *Finding Manifolds With Bilinear Autoencoders*.  
   https://arxiv.org/abs/2510.16820
6. Thomas Dooms BAE project page and code.  
   https://tdooms.github.io/research/bae/  
   https://github.com/tdooms/bae
7. Lee Sharkey MATS stream page.  
   https://www.matsprogram.org/stream/sharkey

