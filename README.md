# Explaining and Harnessing Adversarial Examples (FGSM) — ICLR 2015

This is a NumPy & PyTorch reproduction of the paper Towards Deep Learning Models Resistant to Adversarial Attacks (Mądry et al., 2017). This project focuses on the MNIST dataset, implementing Projected Gradient Descent (PGD) as the ultimate first-order adversary and conducting robust adversarial training.

---

## Quick start

### PGD Adversarial Training
Train the network, storing checkpoints along the way.
```bash
python train.py
```

### Robustness Evaluation
Evaluation script for testing robustness across multiple checkpoints.
```bash
python eval.py
```

### Apply Attack
Apply the attack to the MNIST eval set and stores the resulting adversarial eval set in a `.npy` file.
```bash
python pgd_attack.py
```

### Run Attack & Generate Predictions
Evaluate the model on the examples in the `.npy` file specified in config, while ensuring that the adversarial examples are indeed a valid attack. The script also saves the network predictions in `pred.npy`.
```bash
python run_attack.py
```

---

## Environment / Dependencies

* Python 3.8+
* PyTorch & torchvision
* NumPy
* Matplotlib, tqdm

---

## Datasets used in these experiments

MNIST: Handwritten digits (28x28 grayscale images).

---
