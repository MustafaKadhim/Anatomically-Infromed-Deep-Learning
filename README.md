# Anatomically Informed Deep Learning for Rapid Volumetric CBCT Synthesis

This is an open-source PyTorch and MONAI implementation of a dual-branch Fusion-Skip-Res model that synthesizes 3D cone-beam CT (sCBCT) volumes from ultra-sparse 2D DRR projections and planning CTs. 
By integrating skip & residual connections and a custom anatomically informed loss function (ALF) focused on PTV, bladder, and rectum, our model reconstructs high-fidelity volumetric images for anatomy verification in prostate cancer radiotherapy.  
The repo includes: data preprocessing pipelines, model architectures, and training scripts for reproducible research.
