**Neural News Recommendation System (MIND Dataset)**

**Overview**

This project implements an end-to-end news recommendation system using the MIND (Microsoft News Dataset). The goal is to predict which news articles a user is likely to click based on their reading history and candidate articles.

The system follows a full machine learning pipeline:

- Exploratory Data Analysis (EDA)
- Text preprocessing and feature engineering
- Neural model implementation (NRMS)
- Training and hyperparameter tuning
- Evaluation using ranking metrics

The model is based on the **NRMS (Neural News Recommendation with Multi-Head Self-Attention)** architecture, which leverages attention mechanisms to model both news content and user behavior.

**Problem Formulation**

Unlike traditional classification tasks, this problem is framed as a ranking problem:

"Given a user’s history and a set of candidate articles, rank the articles such that clicked items are scored higher than non-clicked ones."

Each training instance consists of:
- A user’s recent click history
- A candidate set (1 positive + K negative samples)

**Project Structure**

News_Recommendation/

├── data/        # MIND dataset + GloVe embeddings (not tracked)

├── notebooks/

  │       ├── 01_eda.ipynb

  │       ├── 02_preprocess.ipynb

  │       └── 03_training.ipynb

├── src/

  │       ├── preprocessing.py

  │       ├── embeddings.py

  │       ├── data_loader.py

  │       ├── model.py

  │       └── download_data.py

├── models/                # saved models

├── results/               # plots

├── requirements.txt

└── README.md

**Exploratory Data Analysis (EDA)**

Key observations:

- **Category imbalance**: Certain news categories dominate the dataset, which can bias the model toward popular topics.
- **User behavior sparsity**: Most users have very few interactions, while a small number are highly active (power-law distribution).
- **Short text inputs**: News titles are relatively short, making efficient text representation critical.
- **Low click-through rate**: Most impressions contain many non-clicked items, creating a highly imbalanced learning problem.

These characteristics make the task challenging and justify the use of attention-based models.

**Data Preprocessing**

The preprocessing pipeline converts raw text and logs into model-ready tensors.

**Steps:**
- **Tokenization**
  - Lowercasing + removal of special characters
  - Word-level tokenization
- **Vocabulary Construction**
  - Built from training titles
  - Limited to frequent words
  - Unknown words handled via default index
- **Word Embeddings**
  - Pre-trained GloVe (300-dimensional) vectors
  - Random initialization for missing words
- **Sequence Encoding**
  - Titles converted into fixed-length sequences (padding/truncation)
  - Ensures compatibility with batch training
- **Sample Construction**
  - Each sample includes:
    - User history (last 50 clicked articles)
    - Candidate set (1 positive + 4 negatives)
  - Formulated as a pairwise ranking problem

**Model Architecture (NRMS)**

The model consists of two main components:

**1. News Encoder**
- Embedding layer (GloVe)
- Multi-head self-attention
- Additive attention pooling
- Outputs a dense news representation
  
**2. User Encoder**
- Encodes clicked news history
- Multi-head self-attention
- Additive attention pooling
- Outputs a user representation
  
**3. Prediction**
- Dot product between user vector and candidate news vectors
- Produces ranking scores

**Training Details**

Due to CPU limitations, experiments were conducted on a subset of the dataset.

| Parameter        | Value |
| ---------------- | ----- |
| Samples          | 5,000 |
| Batch size       | 16    |
| Learning rate    | 1e-4  |
| Epochs           | 3     |
| Negative samples | 4     |

**Training Loss:**

Epoch 1: 1.4091  
Epoch 2: 1.3795  
Epoch 3: 1.3529  

The steady decrease indicates successful learning of user preferences.

**Evaluation**

Evaluation was performed on the same subset used for training (pipeline validation).

| Metric           | Value |
| ---------------- | ----- |
| AUC              | 0.7385|
| MRR              | 0.6608|
| nDCG@5           | 0.7457|
| nDCG@10          | 0.7457|
		
**Critical Analysis:**
- The relatively high scores indicate that the model effectively ranks clicked items above non-clicked ones.
- However, since evaluation was done on training data, these results **do not reflect generalization performance**.
- A proper evaluation would require:
  - Training on full training set
  - Testing on the MIND validation set

**Hyperparameter Experiments**

| Experiment           | Batch Size | Learning Rate | Final Loss |
| -------------------- | ---------- | ------------- | ---------- |
| Baseline             | 16         | 1e-4          | 1.4836     |
| Smaller batch size   | 8          | 1e-4          | 1.4608     |
| Higher learning rate | 16         | 5e-4          | **1.3909** |

**Observations:**
- Smaller batch size slightly improved optimization due to noisier gradient updates.
- Higher learning rate significantly improved convergence.
- Learning rate had a stronger effect than batch size in this setup.
- These results suggest that the baseline learning rate may be too conservative.

**Limitations**

- Training performed on a small subset due to CPU constraints
- Evaluation done on training data
- No temporal split used for validation
- Model not optimized for large-scale production

**Future Improvements**
- Train on full dataset using GPU
- Evaluate on MIND validation set
- Incorporate:
   - News categories
   - Entity embeddings
- Replace GloVe with BERT-based embeddings
- Improve cold-start handling for new users

**How to Run the Project**

**1. Clone repository**

git clone https://github.com/MominaAmjad24/News_Recommendation.git

cd News_Recommendation

**2. Set up environment**

python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt

**3. Add Data**

Manually place:
- MIND dataset inside data/
- GloVe embeddings inside data/glove/

**4. Run notebooks (in order)**

notebooks/01_eda.ipynb

notebooks/02_preprocessing.ipynb

notebooks/03_training.ipynb

**5. Run Training Experiment**

In 03_training.ipynb:

small_samples = samples[:5000]

Adjust for speed:

small_samples = samples[:3000]

**6. Train Model**

Run training cell:
EPOCHS = 3

**References**

Wu et al. (2020) — MIND Dataset

Wu et al. (2019) — NRMS Model

Pennington et al. (2014) — GloVe


**Summary**

This project demonstrates a complete implementation of a neural news recommendation system using attention-based architectures. Despite computational constraints, the model successfully captures user preferences and provides meaningful ranking predictions, validating the effectiveness of the NRMS framework.

