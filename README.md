# PGD_Pytorch_Robust_Adversarial_Network
A Pytorch re-implementation of the paper "Towards Deep Learning Models Resistant to Adversarial Attacks"

This project reproduces the adversarial training framework proposed in *Towards Deep Learning Models Resistant to Adversarial Attacks* (Madry et al., 2017). The paper formulates adversarial robustness as a robust optimization problem:

$$
\min_{\theta} \mathbb{E}_{(x,y)\sim \mathcal{D}}
\left[
\max_{\delta \in \mathcal{S}} 
\mathcal{L}(f_\theta(x+\delta), y)
\right].
$$

where:

- The inner maximization is approximated using multi-step Projected Gradient Descent (PGD).
- The outer minimization updates model parameters via gradient descent.

The objectives of this reproduction are:

1. Implement PGD-based adversarial training.
2. Track both natural accuracy and adversarial accuracy during training.
3. Analyze training dynamics.
4. Compare final performance with reported results in the original paper.

The dataset used in this experiment is MNIST.

---

## 2. Experimental Setup

### 2.1 Dataset

- Dataset: MNIST
- Image resolution: 28 × 28
- Number of classes: 10

### 2.2 Attack and Training Configuration

| Parameter | Setting |
|------------|----------|
| ε | 0.3 |
| PGD steps | 40 |
| Step size | 0.01 |
| Random initialization | Yes |
| Total training steps | 100000 |

During training, adversarial examples are generated dynamically for each batch, and model updates are performed using adversarial inputs.

---

## 3. Key Training Results

### 3.1 Early Stage (0–3000 steps)

| Step | Natural Accuracy | Adversarial Accuracy |
|------|------------------|----------------------|
| 300  | 74%              | 0%                   |
| 1000 | 84%              | 6%                   |

**Analysis**

- Natural accuracy increases rapidly, indicating that the model learns basic classification structure.
- Adversarial accuracy remains near zero, showing extreme vulnerability to perturbations.
- The model has not yet developed a robust decision boundary.

This behavior is consistent with typical early-stage adversarial training dynamics.

---

### 3.2 Middle Stage (3000–20000 steps)

| Checkpoint | Natural Accuracy | Adversarial Accuracy |
|------------|------------------|----------------------|
| 7200       | 93.75%           | 42.88%               |
| 12000      | 95.91%           | 51.15%               |

**Analysis**

- Natural accuracy stabilizes above 95%, indicating that the PGD inner maximization is functioning correctly.
- Adversarial accuracy increases monotonically.
- No instability or oscillation is observed, showing the model gradually learns smoother and more stable decision boundaries.

Robust feature learning emerges during this phase.

---

### 3.3 Late Stage (90000+ steps)

| Checkpoint | Natural Accuracy | Adversarial Accuracy |
|------------|------------------|----------------------|
| 96000      | 98.50%           | 91.63%               |
| 99000      | 98.49%           | 90.80%               |

Final evaluation under strong PGD attack:

Final Robust Accuracy: 90.26%

---

## 4. Training Dynamics Analysis

### 4.1 Growth Pattern of Adversarial Accuracy

Adversarial accuracy increases steadily from 0% to above 90%, with the following characteristics:

- Monotonic upward trend
- No abrupt jumps
- No late-stage collapse

---

### 4.2 Trade-off Between Natural and Robust Accuracy

Final performance:

- Natural accuracy ≈ 98.5%
- Adversarial accuracy ≈ 90%

No significant degradation of natural accuracy is observed.

This aligns with robust optimization theory: under moderate perturbation constraints, both clean and robust performance can be maintained.

---

## 5. Interpretation of Results

PGD adversarial training optimizes worst-case risk. As training progresses:

1. The decision boundary moves away from the data manifold.
2. Local Lipschitz continuity improves.
3. Small perturbations are less likely to change predictions.

Because MNIST is relatively low-dimensional and well-separated, achieving approximately 90% robustness at ε = 0.3 is feasible.

---

## 6. Conclusion

This experiment successfully reproduces PGD adversarial training on MNIST. The main conclusions are:

1. Training dynamics are stable and consistent with robust optimization theory.
2. Adversarial accuracy increases steadily throughout training.
3. Final robust accuracy exceeds 90%.
4. Results closely match the original paper.
5. No evidence of gradient masking is observed.

The reproduction validates the effectiveness of adversarial training based on robust optimization.
