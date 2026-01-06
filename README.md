# Deep Learning for Computer Vision (DLCV)
**National Taiwan University — Fall 2025**

- **Course code:** CommE 5052
- **Instructor:** Prof. Yu-Chiang Frank Wang

---

## Repository Structure


- **`HW1/` — Self-Supervised Learning & Semantic Segmentation**  
  Implemented DINO-based self-supervised pretraining for image classification and CNN-based semantic segmentation models, including U-Net and improved architectures.

- **`HW2/` — Diffusion Models**  
  Built conditional diffusion models, implemented DDIM sampling, and applied ControlNet for condition-guided image generation.

- **`HW3/` — Vision-Language Models**  
  Performed zero-shot inference with LLaVA, implemented Visual Contrastive Decoding, and applied PEFT (LoRA) on VLMs for image captioning.

- **`HW4/` — Spatial & 3D Understanding**  
  Explored geometry-aware reasoning, including wide-baseline pose estimation and sparse-view 3D Gaussian Splatting.

- **`Final-Project/` (Final Project Challenge 1)**  
  Team project on **Warehouse Spatial Intelligence** (Track 3 of the ICCV 2025 AI City Challenge). 
  
  We developed an enhanced Spatial QA Agent based on the Warehouse SpatialQA framework. Our approach introduces three key improvements: (1) a distance model augmented with a geometric shortcut and 14 depth features, trained using a two-stage loss (MSE → Log-MSE) for high-precision distance estimation; (2) an improved inclusion model that enhances robustness to edge cases by fusing explicit geometric features (e.g., IoU, depth differences) into the visual backbone; and (3) LLM-driven parsing that replaces rule-based rephrasing for more accurate mask identification. Guided by a custom automatic error analysis pipeline, these refinements led to a top-tier score of 95.16 on testing set.

---
