# Vision Transformer-Based Image Search and Recognition System

**Multi-Task Lightweight Image Understanding: Classification · Attribute Prediction · Text-to-Image Retrieval**

**Tech Stack:** Python · PyTorch · Vision Transformers (DeiT-Tiny, MobileViT-S/XXS) · TIMM · t-SNE · TensorBoard :contentReference[oaicite:0]{index=0}  

---

## 1. Overview

This project implements a lightweight, multi-task image understanding system that can:

1. Classify an image into one of several everyday object classes  
2. Predict visual attributes (color, material, condition, and size)  
3. Perform text → image retrieval using natural-language queries  

Two compact Vision Transformer (ViT) variants — **DeiT-Tiny** and **MobileViT-XXS/S** — are fine-tuned on both a curated in-house dataset and a larger pooled dataset to study the effect of dataset scale and label quality on performance and generalization. :contentReference[oaicite:1]{index=1}  

The repository contains training code, evaluation scripts, visualization utilities (confusion matrices, PCA/t-SNE), and a notebook-based prototype for interactive experimentation. :contentReference[oaicite:2]{index=2}  

---

## 2. Datasets

Two datasets are used in the project. :contentReference[oaicite:3]{index=3}  

### 2.1 Dataset 1 – Own (Clean, Balanced)

- 10 everyday object classes (e.g., clothing_cap, tableware_water_bottle, travel_backpack)  
- Approximately 60 images per class; around 600 images in total  
- High-quality, manually verified annotations for:
  - Color (15 values)
  - Material (9 values)
  - Condition (4 values)
- Identity-disjoint train/validation/test splits to avoid subject leakage  
- Images resized to 96×96 px during curation and later to 224×224 px for ViT models  

Dataset 1 is intentionally small but clean and balanced, which enables stable optimization and clear feature clustering in representation space. :contentReference[oaicite:4]{index=4}  

### 2.2 Dataset 2 – Pooled (Large, Noisy)

- 197 object classes  
- Attribute vocabularies:
  - 33 colors
  - 39 materials
  - 14 conditions
- Constructed by pooling multiple sources, resulting in:
  - Missing or invalid attribute labels (e.g., `"unknown"`)
  - Redundant classes (e.g., `water_bottle_1 … water_bottle_10`)
  - Incomplete or generic captions (e.g., “photo of a bottle”)  
- Contains black/placeholder images and caption noise

While Dataset 2 is ~20× larger than Dataset 1, label noise, redundancy, and poor caption quality significantly reduce its effective usability and generalization performance. :contentReference[oaicite:5]{index=5}  

---

## 3. Model Architectures

Two lightweight transformer architectures are used. :contentReference[oaicite:6]{index=6}  

| Model            | Type                         | Parameters | Pretraining | Fine-tuning Strategy                  |
|------------------|------------------------------|-----------:|------------|--------------------------------------|
| **DeiT-Tiny**    | Pure Vision Transformer      | ≈ 5M       | ImageNet   | 2 phases: head training → full FT    |
| **MobileViT-XXS**| Hybrid CNN + Transformer     | ≈ 1.2M     | ImageNet   | 2 phases: head training → full FT    |

Each model is extended with additional heads for **multi-attribute prediction** (color, material, condition, size) and a joint embedding space that supports **text-to-image retrieval**.

### 3.1 Training Configuration

Common hyperparameters across experiments: :contentReference[oaicite:7]{index=7}  

- Optimizer: AdamW  
- Learning rate: 3e-4  
- Training schedule:
  - Phase 1: 8 epochs (classification and attribute heads on frozen backbone)
  - Phase 2: 20 epochs (gradual unfreezing of 6–8 transformer layers)
- Loss: Cross-Entropy for class prediction + auxiliary attribute heads  
- Input size: 224×224  
- Device: CPU or CUDA  

Regularization and optimization strategies explored include: dropout=0.3, label smoothing=0.15, and cosine-annealed learning rate schedules. :contentReference[oaicite:8]{index=8}  

---

## 4. Methods

The training and analysis pipeline includes: :contentReference[oaicite:9]{index=9}  

1. **Data preprocessing**
   - Image resizing and normalization
   - Handling of missing attributes and captions
   - Train/validation/test splitting with identity disjointness for Dataset 1

2. **Two-phase fine-tuning**
   - Phase 1: Train only the newly initialized classification and attribute heads
   - Phase 2: Unfreeze and fine-tune backbone layers progressively to adapt ImageNet features to the new domain

3. **Evaluation**
   - Multi-class accuracy and macro F1-score
   - Attribute-wise performance (color, material, condition, size)
   - Qualitative assessment via confusion matrices, PCA/t-SNE visualizations, and retrieval examples

4. **Error analysis**
   - Inspecting confusion between visually similar classes (e.g., shampoo bottle vs soap bar)
   - Examining noisy clusters and overlapping classes in Dataset 2

---

## 5. Results

This section intentionally uses **only the consolidated results from the provided result summary image** for quantitative reporting.

### 5.1 Multi-Task Performance on Everyday Object Recognition

- Developed a **multi-task deep learning system** achieving:
  - **93.36% classification accuracy**
  - **93.46% F1-score**
- Evaluated on **10 everyday object categories** with multi-attribute prediction showing:
  - **Color:** 78.7%
  - **Material:** 72.0%
  - **Condition:** 69.2%
  - **Size:** 66.4%
- Integrated a **text-to-image retrieval** module that supports natural-language queries such as  
  “blue plastic jug” to retrieve semantically relevant images.

### 5.2 Dataset Curation and Scaling Effects

- Curated **600+ annotated images** (96×96 px, 12+ instances per class) with identity-disjoint train/validation splits.  
- Fine-tuned two Vision Transformer variants:
  - **DeiT-Tiny:** 5.7M parameters
  - **MobileViT-S:** 2M parameters  
- Used ImageNet initialization and staged fine-tuning (frozen backbone → progressive unfreezing of 6–8 layers).  
- Scaling from the curated dataset to a larger dataset consisting of **1,406 images** demonstrated:
  - **+6.36% improvement in classification accuracy**
  - **+30–37% gains in attribute-prediction performance**  

### 5.3 Experimental Design

- Conducted **four controlled training experiments** comparing:
  - Model architectures (DeiT-Tiny vs MobileViT variants)
  - Dataset scales (small balanced vs larger/noisy)
- Tuned:
  - Dropout = 0.3
  - Label smoothing = 0.15
  - AdamW optimizer with cosine annealing  
- Produced:
  - Confusion matrices
  - Per-class F1-scores
  - t-SNE visualizations  
- Validated solutions via an **interactive Jupyter Notebook prototype** to test classification, attributes, and retrieval in real time.

---

## 6. Visualizations and Analysis

### 6.1 Confusion Matrices

- For **Dataset 1**, confusion matrices reveal that most mistakes occur between semantically similar categories such as *shampoo bottle* and *soap bar*. DeiT-Tiny shows a strong diagonal with very few off-diagonal errors, indicating high precision, whereas MobileViT exhibits more confusion for similar items. :contentReference[oaicite:10]{index=10}  
- For **Dataset 2**, confusion matrices for both models show heavy overlap among semantically related classes (e.g., toothpaste vs toothbrush, wallet vs bag), highlighting a noisy label taxonomy and redundant categories. :contentReference[oaicite:11]{index=11}  

### 6.2 PCA and t-SNE Feature Projections

- On **Dataset 1**, PCA plots illustrate tight and well-separated clusters, especially for DeiT-Tiny. MobileViT also shows separable clusters, though with reduced margins between classes. :contentReference[oaicite:12]{index=12}  
- On **Dataset 2**, PCA and t-SNE plots exhibit dense overlaps and diffuse boundaries, reflecting semantic noise, label redundancy, and attribute inconsistencies; this correlates with the lower quantitative scores. :contentReference[oaicite:13]{index=13}  

### 6.3 Retrieval Examples

Text-to-image retrieval examples (captured via screenshots and video in the original report) show successful retrieval for attribute-rich queries and reveal degraded quality when attributes are missing or inconsistent, confirming the strong dependency of retrieval on clean attribute labels. :contentReference[oaicite:14]{index=14}  

---

## 7. Key Insights

The project yields several key observations about data quality, model capacity, and fine-tuning strategy. :contentReference[oaicite:15]{index=15}  

1. **Data quality outweighs data quantity**  
   - The smaller, clean Dataset 1 delivers higher and more stable performance than the much larger but noisy Dataset 2.  
   - Poor or inconsistent annotations substantially limit achievable performance.

2. **Full model fine-tuning is crucial**  
   - Training only the classification head on frozen features yields F1 scores below ~30%, while full fine-tuning boosts performance to around 70% or more.

3. **Class imbalance and noisy attributes matter**  
   - Rare classes and missing or `"unknown"` attributes reduce effective sample size and hurt both classification and retrieval.  
   - Class-weighted losses and better curation significantly improve macro F1 and minority-class predictions.

4. **Retrieval quality is highly sensitive to attribute correctness**  
   - Clean and consistent color/material/condition annotations directly improve text-image alignment and retrieval quality.

5. **Model choice and capacity**  
   - DeiT-Tiny benefits from richer token mixing and positional embeddings, leading to better generalization on clean data.  
   - MobileViT trades some accuracy for parameter efficiency, which may still be attractive for resource-constrained deployments.

6. **Future work**  
   - Increase epochs for large noisy datasets, potentially with stronger regularization  
   - Apply label cleaning, Mixup/CutMix, and alternative optimizers  
   - Explore larger ViT variants for improved robustness once label quality is improved  

---




