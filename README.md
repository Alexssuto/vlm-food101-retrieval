# Vision–Language Model Evaluation on Food-101

## Overview
This project evaluates the zero-shot performance of a pre-trained vision–language foundation model (CLIP) on the Food-101 dataset. The goal is to understand how well a large-scale image–text model generalizes to a specific food domain and how prompt wording and visual context affect classification performance.

The project focuses on evaluation, experimentation, and qualitative analysis rather than model training.

---

## Model
- CLIP (ViT-B/32)
- Framework: PyTorch
- Inference: GPU-accelerated (CUDA)
- Task: Zero-shot image classification via text prompts

---

## Dataset
- Food-101
- 101 food categories
- Evaluation split: validation
- Samples evaluated per run: 2,000

---

## Methodology
1. Encode class labels as natural language prompts
2. Encode images and text into a shared embedding space
3. Rank class labels by cosine similarity
4. Measure Top-1 and Top-5 accuracy
5. Log all experiments to CSV for reproducibility

---

## Prompt Sensitivity Experiments

Three prompt templates were evaluated:

- `a photo of {label}`
- `a centered photo of {label}`
- `a close-up photo of {label}`

### Results (2,000 images)

| Prompt style | Top-1 | Top-5 |
|-------------|------|------|
| base | 0.755 | 0.967 |
| centered | 0.787 | 0.981 |
| close_up | **0.800** | 0.980 |

These results show that prompt wording has a measurable impact on zero-shot performance.

---

## Failure Analysis

A qualitative failure analysis was conducted to identify systematic error patterns in misclassified examples.

Observed failure modes include:
- Visual similarity between food categories (e.g., beignets vs donuts)
- Cultural and linguistic overlap in food naming
- Contextual cues such as dipping sauces or multiple foods in frame
- Unusual plating or presentation styles

Detailed failure examples (images + notes): **[analysis/failures.md](analysis/failures.md)**


## How to Run

```bash
pip install -r requirements.txt
python -m scripts.run_eval