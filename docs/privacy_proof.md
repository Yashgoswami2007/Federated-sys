# Differential Privacy Mathematical Proof & Accounting

This document provides the formal mathematical foundations, noise calibration derivations, and composition accounting used by FusionNet's DP-SGD implementation (`federation/privacy.py`).

---

## 1. Formal (ε, δ)-Differential Privacy Definition

**Definition (Dwork & Roth, 2014):**

A randomized mechanism $\mathcal{M}: \mathcal{D} \to \mathcal{R}$ satisfies $(\varepsilon, \delta)$-differential privacy if for all pairs of neighboring datasets $D, D' \in \mathcal{D}$ (differing in exactly one record) and all measurable subsets $S \subseteq \mathcal{R}$:

$$\Pr[\mathcal{M}(D) \in S] \leq e^{\varepsilon} \cdot \Pr[\mathcal{M}(D') \in S] + \delta$$

**Interpretation:**
- $\varepsilon$ (epsilon) — **privacy loss budget**. Lower = more private. FusionNet targets $\varepsilon \leq 1.0$.
- $\delta$ (delta) — **failure probability**. Probability the $e^\varepsilon$ bound breaks. FusionNet targets $\delta \leq 10^{-5}$.
- "Neighboring datasets" — one client's training data vs. the same data with one record removed.

---

## 2. Gaussian Mechanism & Noise Calibration

### 2.1 Sensitivity

The **L2 sensitivity** of a function $f: \mathcal{D} \to \mathbb{R}^d$ is:

$$\Delta_2 f = \max_{D, D'} \|f(D) - f(D')\|_2$$

For gradient descent with per-sample gradient clipping at norm $C$, the sensitivity of the clipped average gradient is:

$$\Delta_2 = C / n$$

where $n$ is the batch size (per-sample clipping ensures no single record contributes more than $C$ to the gradient).

### 2.2 Gaussian Mechanism Theorem

The Gaussian mechanism adds noise $\eta \sim \mathcal{N}(0, \sigma^2 I)$ to the output of $f$:

$$\mathcal{M}(D) = f(D) + \eta, \quad \eta \sim \mathcal{N}(0, \sigma^2 I)$$

This satisfies $(\varepsilon, \delta)$-DP when:

$$\sigma \geq \frac{\Delta_2 \cdot \sqrt{2 \ln(1.25/\delta)}}{\varepsilon}$$

### 2.3 FusionNet Noise Multiplier (per round)

With $\Delta_2 = C = 1.0$ (gradient clipping norm), $\varepsilon_0 = 1.0$, $\delta = 10^{-5}$:

$$\sigma = \frac{1.0 \cdot \sqrt{2 \ln(1.25 / 10^{-5})}}{1.0} = \sqrt{2 \ln(125000)} \approx \sqrt{2 \times 11.736} \approx \mathbf{4.84}$$

> **FusionNet uses $\sigma \approx 4.84$ per training step** when the Opacus engine is active. The fallback `CustomPrivacyEngine` uses the identical formula.

---

## 3. DP-SGD Algorithm

FusionNet's local training implements the **DP-SGD algorithm** (Abadi et al., 2016):

```
For each mini-batch B of size L:
  1. For each sample xᵢ ∈ B:
       Compute per-sample gradient: gᵢ = ∇L(θ; xᵢ)
       Clip to norm C:              g̃ᵢ = gᵢ / max(1, ||gᵢ||₂ / C)

  2. Average and add calibrated noise:
       g̃ = (1/L) · [Σᵢ g̃ᵢ + N(0, σ²C²I)]

  3. Update:  θ ← θ - η · g̃
```

**Opacus** (PyTorch's DP library) automates Steps 1 and 2 via gradient hooks that:
1. Compute per-sample gradients in the backward pass
2. Clip each per-sample gradient to norm $C$
3. Sum the clipped gradients and add $\mathcal{N}(0, \sigma^2 C^2 I)$ noise
4. Divide by the batch size

FusionNet's `CustomPrivacyEngine` fallback implements the same algorithm manually when Opacus hooks fail on 4-bit quantized modules.

---

## 4. Privacy Composition Across Rounds

Each FL round involves one local training epoch with DP-SGD. Privacy budget **accumulates** across rounds.

### 4.1 Basic Composition (Conservative Upper Bound)

After $R$ rounds each satisfying $(\varepsilon_0, \delta_0)$-DP:

$$\varepsilon_{\text{total}} = R \cdot \varepsilon_0, \quad \delta_{\text{total}} = R \cdot \delta_0$$

**At $\varepsilon_0 = 1.0$, $\delta_0 = 10^{-5}$:**

| Rounds (R) | ε_total (basic) | δ_total (basic) |
|---|---|---|
| 1  | 1.0  | 1×10⁻⁵ |
| 5  | 5.0  | 5×10⁻⁵ |
| 10 | 10.0 | 10⁻⁴   |
| 20 | 20.0 | 2×10⁻⁴ |

### 4.2 Rényi Differential Privacy (RDP) Composition — Tighter Bound

Opacus uses the **RDP accountant** (Mironov, 2017) which gives much tighter composition than basic:

The Gaussian mechanism with $\sigma$ satisfies $(\alpha, \alpha / 2\sigma^2)$-RDP for any order $\alpha > 1$.

After $R$ rounds, the RDP composition theorem gives:

$$\varepsilon_{\text{RDP}}(\alpha) = R \cdot \frac{\alpha}{2\sigma^2}$$

Converting to $(\varepsilon, \delta)$-DP:

$$\varepsilon = \varepsilon_{\text{RDP}}(\alpha) + \frac{\ln(1/\delta)}{\alpha - 1}$$

Minimizing over $\alpha$ gives the tightest $(\varepsilon, \delta)$ bound.

**At $\sigma \approx 4.84$, $\delta = 10^{-5}$ (approximate RDP results from Opacus):**

| Rounds (R) | ε_total (RDP) | ε_total (basic) | RDP savings |
|---|---|---|---|
| 1  | ~1.00 | 1.0  | 1.00× |
| 5  | ~2.10 | 5.0  | 2.38× tighter |
| 10 | ~3.15 | 10.0 | 3.17× tighter |
| 20 | ~4.80 | 20.0 | 4.17× tighter |

> **FusionNet's `PrivacyAccountant` uses basic composition** (conservative) for simplicity. In production, switch to Opacus's built-in RDP accountant via `privacy_engine.get_epsilon(delta)` for the tighter bounds shown above.

---

## 5. Privacy Budget Tracking Implementation

`fusionnet-client/federation/privacy.py` — `PrivacyAccountant` class:

```python
class PrivacyAccountant:
    def record_round(self, round_num: int, epsilon_spent: float) -> bool:
        self.cumulative_epsilon += epsilon_spent
        # Hard stop if budget exceeded
        if self.cumulative_epsilon >= self.budget_cap:
            raise PrivacyBudgetExhausted(...)
        return True  # budget OK
```

The accountant **persists** `cumulative_epsilon` to `checkpoints/privacy_budget.json` after every round, so budget is not reset on client restarts.

---

## 6. Actual ε Values from Training Runs

> ⚠️ This section is updated after each training run. Values below are from the local FL simulation.

| Run | Rounds | Clients | ε per round | ε total (Opacus RDP) | Dataset |
|---|---|---|---|---|---|
| Local sim (CPU) | 3 | 3 | ~0.18 | 0.575 | Banking77 (simulated) |
| *Real CUDA run* | — | — | — | — | *TBD — run on RTX 5060* |
| *ROCm run* | — | — | — | — | *TBD — run on AMD GPU* |

---

## 7. Security Analysis: AFLoRA A-Matrix Leakage

**Open question:** Since the global `A` matrix (shape `[out_features, rank]`) is trained locally and aggregated globally, does the aggregated `A` leak information about any individual client's data?

**Current analysis:**
- The aggregated `A` is a weighted sum of local `A` matrices: $A_{\text{global}} = \Sigma_i w_i A_i^{(local)}$
- Each $A_i^{(local)}$ was trained with DP-SGD noise, so it satisfies $(\varepsilon, \delta)$-DP individually
- By the post-processing theorem (DP is closed under post-processing), $A_{\text{global}}$ also satisfies $(\varepsilon, \delta)$-DP for the same $\varepsilon$ as the training run

**Post-processing theorem:**
If $\mathcal{M}$ is $(\varepsilon, \delta)$-DP, then for any (possibly randomized) function $g$, the composition $g \circ \mathcal{M}$ is also $(\varepsilon, \delta)$-DP.

Since FedAvg is a deterministic post-processing step on DP-trained outputs, the global model inherits the DP guarantee.

**Limitation:** This does not protect against **model inversion attacks** on the *inference output* of the global model — a separate threat model requiring output perturbation or prediction-result filtering, listed in the roadmap.

---

## References

1. Dwork, C., & Roth, A. (2014). *The Algorithmic Foundations of Differential Privacy*. Foundations and Trends in Theoretical Computer Science.
2. Abadi, M., Chu, A., Goodfellow, I., et al. (2016). *Deep Learning with Differential Privacy*. CCS 2016. https://arxiv.org/abs/1607.00133
3. Mironov, I. (2017). *Rényi Differential Privacy*. CSF 2017. https://arxiv.org/abs/1702.07476
4. Yousefpour, A., et al. (2021). *Opacus: User-Friendly Differential Privacy Library in PyTorch*. https://arxiv.org/abs/2109.12298
5. McMahan, H.B., et al. (2018). *Learning Differentially Private Recurrent Language Models*. ICLR 2018. https://arxiv.org/abs/1710.06963
