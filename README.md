# Hybrid AI-Based MIMO-OTFS Receiver

## Overview

This project investigates advanced AI-assisted signal detection techniques for MIMO-OTFS wireless communication systems under Rayleigh fading channels. The work combines classical communication theory with deep learning to improve Bit Error Rate (BER), spectral efficiency, and robustness in high-mobility wireless environments.

The project evaluates and compares multiple detection approaches including conventional linear detectors and modern AI-enhanced hybrid receivers.

---

# Objectives

- Simulate MIMO-OTFS communication systems in MATLAB
- Compare conventional and AI-based detection methods
- Improve BER performance over classical MMSE detection
- Investigate hybrid deep learning architectures for wireless receivers
- Analyze throughput and detection reliability under noisy channels

---

# Implemented Detection Models

## Conventional Detectors
- Zero Forcing (ZF)
- Minimum Mean Square Error (MMSE)

---

## AI-Based Models

### 1. Residual Learning Receiver
Neural network learns the residual error between transmitted symbols and MMSE estimates.

### 2. Hybrid Residual AI Receiver
Combines:
- MMSE estimation
- Residual correction
- Confidence gating
- Hard constellation projection

### 3. Deep Unfolding Inspired Detector
A DetNet-inspired unfolding architecture that maps iterative signal recovery into trainable neural network layers.

### 4. Classification-Based Detector
Symbol classification neural network for constellation prediction.

---

# System Configuration

| Parameter | Value |
|---|---|
| MIMO Size | 2×2 / 4×4 |
| Channel | Rayleigh Fading |
| Modulation | QPSK / 16-QAM |
| SNR Range | 0–20 dB |
| Simulation Platform | MATLAB |
| AI Framework | MATLAB Deep Learning Toolbox |

---

# Project Structure

```text
Hybrid-MIMO-OTFS-AI/
│
├── data/
│   ├── raw/
│   ├── cleaned/
│
├── scripts/
│   ├── generate_dataset.m
│   ├── baseline_mmse.m
│   ├── residual_ai.m
│   ├── hybrid_ai.m
│   ├── unfolding_detector.m
│
├── results/
│   ├── ber_plots/
│   ├── logs/
│
├── docs/
│
├── README.md
├── requirements.txt
