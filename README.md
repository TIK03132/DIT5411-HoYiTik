# THEi DIT5411 Machine Learning Assignment (Character Recognition) 
**Handwritten Chinese Character Recognition using Deep Learning**  
*Course: DIT5411 – Deep Learning for Computer Vision*  
*Date: November 17, 2025*

---

## Project Overview

This project implements a **large-scale handwritten Chinese character recognition system** capable of classifying **13,065 unique characters** using deep neural networks.

- **Dataset**: Custom collection of handwritten Chinese characters (one folder per class)
- **Input**: 64×64 grayscale images
- **Goal**: Train models to predict the correct character from a live drawing

---

## Dataset & Preprocessing

### Source
- Raw images located in: `data/characters/<class_id>/`
- Each class folder contains **at least 40 seed images** (per assignment requirement)

### Augmentation Pipeline
To reach **≥200 training samples per class**, we applied **on-the-fly PIL-based augmentation**:

```python
random_augment_pil():
    - Random rotation: ±20°
    - Random affine: shear (±8°), translate (±12%)
    - Random zoom: 85% – 115% + center crop/pad
    - Random contrast: 0.8 – 1.2
    - Random brightness: 0.85 – 1.15
    - Final: convert to 'L' (grayscale), resize to 64×64
