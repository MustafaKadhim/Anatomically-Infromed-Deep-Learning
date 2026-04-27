<p align="center">
  <img src="assets/logo.png" width="180" alt="Anatomically informed deep learning logo">
</p>

<h1 align="center">
  Anatomically Informed Deep Learning for Fast, Low-Dose Synthetic CBCT
</h1>

<p align="center">
  <strong>Generating volumetric synthetic CBCT from ultra-sparse 2D projections and planning CT for prostate radiotherapy</strong>
</p>

<p align="center">
  <a href="https://www.nature.com/articles/s41598-025-23781-7">
    <img src="https://img.shields.io/badge/Paper-Scientific%20Reports-blue" alt="Scientific Reports paper">
  </a>
  <img src="https://img.shields.io/badge/Framework-PyTorch%20%7C%20MONAI-orange" alt="PyTorch and MONAI">
  <img src="https://img.shields.io/badge/Task-2D--to--3D%20sCBCT-green" alt="2D to 3D sCBCT">
  <img src="https://img.shields.io/badge/Application-Prostate%20Radiotherapy-lightgrey" alt="Prostate radiotherapy">
</p>

<p align="center">
  <a href="https://www.nature.com/articles/s41598-025-23781-7"><strong>Read the paper</strong></a>
  ·
  <a href="#overview"><strong>Overview</strong></a>
  ·
  <a href="#installation"><strong>Installation</strong></a>
  ·
  <a href="#citation"><strong>Citation</strong></a>
</p>

---

<p align="center">
  <a href="https://www.nature.com/articles/s41598-025-23781-7">
    <img src="assets/framework_overview.png" width="850" alt="Overview of the proposed framework">
  </a>
</p>

---

## Overview

This repository provides the open-source implementation of the **Fusion-Skip-Res** deep learning framework presented in:

> **Anatomically informed deep learning framework for generating fast, low-dose synthetic CBCT for prostate radiotherapy**  
> Mustafa Kadhim et al., *Scientific Reports*, 2025.

The framework generates volumetric **synthetic cone-beam CT (sCBCT)** images from:

- two orthogonal 2D digitally reconstructed radiographs (DRRs), and  
- a reference 3D planning CT (pCT).

The goal is to explore whether fast, low-dose 2D imaging can be used to recover 3D anatomical information for image-guided prostate radiotherapy.

---

## Key features

- **Dual-input architecture**  
  Combines information from sparse 2D DRR projections and 3D planning CT images.

- **Fusion-Skip-Res model**  
  A dual-branch 2D/3D encoder-decoder framework with skip and residual connections.

- **Anatomically informed loss function**  
  Uses clinically relevant structures such as the PTV, bladder, and rectum to guide reconstruction quality.

- **Fast volumetric inference**  
  Generates synthetic CBCT volumes in approximately 8 ms per case, excluding data loading and GPU warm-up.

- **Radiotherapy-focused evaluation**  
  Includes masked image-quality metrics to reduce the influence of background voxels.

---

## Repository contents

```text
.
├── assets/                  # Logo and README figures
├── models/                  # Model architectures
├── preprocessing/           # Data preprocessing scripts
├── training/                # Training scripts
├── evaluation/              # Evaluation and metric scripts
├── losses/                  # Anatomically informed loss functions
├── requirements.txt         # Python dependencies
└── README.md
