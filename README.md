#  Neural News Recommendation System (MIND Dataset)

##  Overview

This project implements an end-to-end **news recommendation system** using the **MIND (Microsoft News Dataset)**. The goal is to predict which news articles a user is likely to click based on their reading history and candidate articles.

The system follows a full machine learning pipeline:

- Exploratory Data Analysis (EDA)
- Text preprocessing and feature engineering
- Neural model implementation (NRMS)
- Training and hyperparameter tuning
- Evaluation using ranking metrics

The model is based on the **NRMS (Neural News Recommendation with Multi-Head Self-Attention)** architecture.

---

## Problem Formulation

Unlike traditional classification tasks, this problem is framed as a **ranking problem**:

> Given a user’s history and a set of candidate articles, rank the articles such that clicked items are scored higher than non-clicked ones.

Each training instance consists of:

- A user’s recent click history  
- A candidate set (1 positive + K negative samples)

---

##  Project Structure
News_Recommendation/

├── data/ # MIND dataset + GloVe embeddings (not tracked)

├── notebooks/

│ ├── 01_eda.ipynb

│ ├── 02_preprocessing.ipynb

│ └── 03_training.ipynb

├── src/

│ ├── preprocessing.py

│ ├── embeddings.py

│ ├── data_loader.py

│ ├── model.py

│ └── download_data.py

├── models/ # saved models

├── results/ # plots

├── requirements.txt

└── README.md



---

## Exploratory Data Analysis (EDA)

Key observations:

- **Category imbalance:** Certain news categories dominate the dataset  
- **User behavior sparsity:** Most users have few interactions  
- **Short text inputs:** Titles are short → efficient representation matters  
- **Low click-through rate:** Highly imbalanced dataset  

These characteristics make recommendation challenging and motivate attention-based models.

---

##  Data Preprocessing

### Pipeline Steps

**Tokenization**
- Lowercasing and removing special characters  
- Word-level tokenization  

**Vocabulary Construction**
- Built from training titles  
- Limited to frequent words  
- Unknown words mapped to default index  

**Word Embeddings**
- Pre-trained **GloVe (300d)**  
- Random initialization for missing words  

**Sequence Encoding**
- Fixed-length padded/truncated sequences  

**Sample Construction**
- User history: last 50 clicked articles  
- Candidate set: 1 positive + 4 negatives  
- Formulated as a ranking task  

---

##  Model Architecture (NRMS)

### News Encoder
- Embedding layer (GloVe)
- Multi-head self-attention
- Additive attention pooling

### User Encoder
- Encodes clicked news history
- Multi-head self-attention
- Additive attention pooling

### Prediction
- Dot product between user vector and candidate vectors  
- Outputs ranking scores  

---

##  Training Details

Due to CPU limitations, training was conducted on a subset.

| Parameter        | Value |
|-----------------|------|
| Samples         | 5,000 |
| Batch Size      | 16 |
| Learning Rate   | 1e-4 |
| Epochs          | 3 |
| Negative Samples| 4 |

### Training Loss

Epoch 1: 1.4091  
Epoch 2: 1.3795  
Epoch 3: 1.3529

The steady decrease indicates successful learning of user preferences.

---

## Evaluation

Evaluation was performed on the same subset (pipeline validation).

| Metric   | Value |
|----------|------|
| AUC      | 0.7385 |
| MRR      | 0.6608 |
| nDCG@5   | 0.7457 |
| nDCG@10  | 0.7457 |

### Critical Analysis

- Model successfully ranks clicked items above non-clicked  
- Results are **optimistic** due to training-set evaluation  
- True performance requires validation set evaluation  

---

##  Hyperparameter Experiments

| Experiment              | Batch Size | Learning Rate | Final Loss |
|------------------------|------------|---------------|------------|
| Baseline               | 16         | 1e-4          | 1.4836     |
| Smaller batch size     | 8          | 1e-4          | 1.4608     |
| Higher learning rate   | 16         | 5e-4          | **1.3909** |

### Observations

- Smaller batch → slightly better optimization  
- Higher learning rate → faster convergence  
- Learning rate had stronger impact than batch size  

---

## Limitations

- Trained on subset (CPU constraint)  
- Evaluated on training data  
- No temporal validation split  
- Not optimized for large-scale deployment  

---

##  Future Improvements

- Train on full dataset with GPU  
- Evaluate on validation set  
- Incorporate:
  - News categories  
  - Entity embeddings  
- Replace GloVe with BERT  
- Improve cold-start handling  

---

## How to Run

### 1. Clone repository

```bash
git clone https://github.com/MominaAmjad24/News_Recommendation.git
cd News_Recommendation
```

### 2. Set up environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
### 3. Add Data

Manually place:
- MIND dataset inside data/
- GloVe embeddings inside data/glove/

### 4. Run notebooks (in order)

notebooks/01_eda.ipynb

notebooks/02_preprocessing.ipynb

notebooks/03_training.ipynb

### 5. Run Training Experiment

In 03_training.ipynb:

small_samples = samples[:5000]

Adjust for speed:

small_samples = samples[:3000]

### 6. Train Model

Run training cell:
EPOCHS = 3

## References

Wu et al. (2020) - MIND Dataset

Wu et al. (2019) - NRMS Model

Pennington et al. (2014) - GloVe


## Summary

This project demonstrates a complete implementation of a neural news recommendation system using attention-based architectures. Despite computational constraints, the model successfully captures user preferences and provides meaningful ranking predictions, validating the effectiveness of the NRMS framework.

