# 🧬 TXGNN-StaR
### Stability-Aware GNN Framework with Reliability-Quantified Toxicophore Discovery

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-EE4C2C.svg)](https://pytorch.org)
[![Live Demo](https://img.shields.io/badge/🤗-Live%20Demo-yellow)](https://huggingface.co/spaces/yashikaSharma/txgnn-star)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> UG Research Fellowship — Thapar Institute of Engineering & Technology, Patiala
> **Authors:** Savree Dohar · Yashika Sharma | **Mentor:** Dr. Vijay Kumari

---

## 🚀 Live Demo
👉 **[Try it on Hugging Face Spaces](https://huggingface.co/spaces/yashikaSharma/txgnn-star)**

---

## 🎯 Overview
TXGNN-StaR predicts molecular toxicity using a **5-layer Graph Attention Network (GAT)** trained on Tox21 dataset (12 endpoints, 7,823 molecules). It goes beyond simple prediction by quantifying **how much you can trust each prediction**.

---

## ✨ Key Features
- **12 Tox21 Endpoint Predictions** — NR-AR, NR-AhR, SR-p53 and more
- **ERI Score** — Explanation Reliability Index combining stability + sparsity + chemical validity
- **RGRT Tiers** — Trust / Caveat / Mandatory Review decision layer
- **PBES** — Perturbation-Based Explanation Stability metric

---

## 📊 Key Results

| Metric | Value |
|---|---|
| GAT Test ROC-AUC | **0.777** |
| Mean ACS (n=100) | **0.801 ± 0.137** |
| High Stability Molecules | **86%** |
| Mean ERI | **0.647 ± 0.159** |

---

## 🛠️ Tech Stack
`PyTorch` `PyTorch Geometric` `RDKit` `Gradio` `Tox21` `GAT`

---

## 👩‍💻 Authors
**Savree Dohar** · **Yashika Sharma**
Mentor: Dr. Vijay Kumari, CSE — Thapar Institute of Engineering & Technology
