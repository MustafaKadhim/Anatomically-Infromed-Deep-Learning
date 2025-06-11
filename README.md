# Anatomically informed deep learning framework for generating fast, low-dose synthetic CBCT for prostate radiotherapy ✨

This is an open-source PyTorch and MONAI implementation of a dual-branch Fusion-Skip-Res model that synthesizes 3D cone-beam CT (sCBCT) volumes from ultra-sparse 2D DRR projections and planning CTs. 🔬⚡

- 🛠️ **Architecture:** Dual-branch 2D and 3D encoder/decoder frameork with skip & residual connections  
- 📐 **Anatomical Fidelity:** Custom Anatomically Informed Loss (ALF) focused on PTV, bladder, and rectum  
- ⏱️ **Speed:** Generates high-fidelity volumetric images in < 8 ms per case (excluding Data loading and GPU warmup times)
- ✅ **Use Case:** Feasability of real-time, low-dose IGRT verification in prostate radiotherapy  
- 📦 **What’s Inside:** Data preprocessing pipelines, model definitions, training scripts, and masked evaluation metrics   

> Enhance reproducibility and streamline IGRT workflows! 🤝🔍
