# Fake News Detection using Text Classification and Ensemble Learning

This project implements a complete machine learning pipeline for classifying news as "real" or "fake" based on text content (tweets). The pipeline covers data preprocessing, feature engineering using TF-IDF, training of multiple base classification models, and evaluation of ensemble techniques like Voting and Stacking Classifiers.

## Project Structure 📁

| Notebook/Directory | Description |
| :--- | :--- |
| `preprocessing.ipynb` | **Data Preparation:** Handles cleaning, feature engineering (TF-IDF), splitting the dataset, and creating training splits for cross-validation and ensemble training. |
| `ml_models.ipynb` | **Base Models Training:** Trains and evaluates four different base classifiers using **Stratified K-Fold Cross-Validation** and **Grid Search** for hyperparameter tuning. |
| `ensemble.ipynb` | **Ensemble Training:** Combines the best base models using **Voting Classifier** and **Stacking Classifier** to potentially improve performance. |
| `./tfidf/` | Directory containing the final TF-IDF feature matrices for training and testing. |
| `./split/` | Directory containing the preprocessed text data splits (raw and for cross-validation). |
| `./models/` | Directory for saving the trained base and ensemble models as `.pkl` files. |
| `label_encoder.pkl` | The fitted `LabelEncoder` object for converting string labels to numeric values. |

---

## 1. Data Preprocessing and Feature Engineering (Refer to `preprocessing.ipynb`)

The data undergoes extensive text preprocessing to clean the raw tweets before feature extraction.

### Preprocessing Steps

The `preprocess_text_improved` function executes the following steps:
1.  **Lowercasing**.
2.  Removal of **URLs** and **@mentions**.
3.  **Emojization:** Converts emojis into descriptive text tokens (e.g., "🙂" to " :slightly\_smiling\_face: ").
4.  **Hashtag Expansion:** Splits concatenated words within hashtags into separate tokens (e.g., `#IndiaFightsCorona` is segmented).
5.  **Number Replacement:** Replaces all digits with a `<NUM>` token placeholder.
6.  **Character Normalization:** Reduces elongated characters (e.g., "looove" $\rightarrow$ "loove").
7.  **Punctuation Removal**.
8.  **Tokenization**.
9.  **Stopword Removal** (optional, but applied in the notebook).
10. **Lemmatization** using NLTK's **Part-of-Speech (POS) tagging** for accurate root word extraction.

### Feature Extraction

* **TF-IDF (Term Frequency-Inverse Document Frequency):** The cleaned text is converted into a numerical feature vector using `TfidfVectorizer`.
* The vectorizer is configured for **unigrams and bigrams** (`ngram_range=(1, 2)`), a maximum of **2000 features** (`max_features=2000`), and uses **L2 normalization** (`norm='l2'`).
* The dataset is split into **90% training** and **10% testing** (`test_size=0.1, random_state=42`).
* The training data is further split into **four parts** (`train_1.csv` to `train_4.csv`) for use in ensemble and cross-validation training.

---

## 2. Base Model Training and Evaluation (Refer to `ml_models.ipynb`)

Four distinct classification models are trained on the prepared TF-IDF features using the `train_logistic_regression`, `train_knn`, `train_svm`, and `train_decision_tree` functions. All models utilize **StratifiedKFold (n\_splits=5)** and **GridSearchCV** for robust hyperparameter optimization.

### Summary of Best Performing Base Models on Test Data

| Model | Best Parameters | Accuracy | Precision (Weighted) | Key Metrics from Test Data |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** (`model2`) | `{'C': 1, 'penalty': 'l2', 'solver': 'lbfgs'}` | **0.8708** | **0.8708** | Confusion Matrix: `[[445 72], [65 478]]` |
| **Linear Support Vector Machine (SVC)** (`model3`) | `{'C': 0.1}` | **0.8792** | **0.8794** | Confusion Matrix: `[[447 70], [58 485]]` |
| **K-Nearest Neighbors (KNN)** (`model1`) | `{'metric': 'euclidean', 'n_neighbors': 11, 'weights': 'distance'}` | **0.8255** | **0.8423** | Confusion Matrix: `[[482 35], [150 393]]` |
| **Decision Tree** (`model4`) | `{'max_depth': 20}` | **0.8113** | **0.8133** | Confusion Matrix: `[[436 81], [119 424]]`` |

---

## 3. Ensemble Model Training (Refer to `ensemble.ipynb`)

Ensemble models are trained using the best-performing base models as input, aiming to leverage their combined strengths.

### Stacking Classifier

The **Stacking Classifier** uses the predictions (probabilities) of the four base models as features for a final **Logistic Regression** meta-classifier.

* **Best Parameters:** `{'final_estimator__C': 0.1, 'final_estimator__solver': 'liblinear'}`
* **Test Accuracy:** **0.8708**
* **Test Precision (Weighted):** **0.8726**

### Voting Classifier

The **Voting Classifier** aggregates the predictions from the four base models. The best performance was achieved using **soft voting** with equal weights.

* **Best Parameters:** `{'voting': 'soft', 'weights': [1, 1, 1, 1]}`
* **Test Accuracy:** **0.8481**
* **Test Precision (Weighted):** **0.8673**

### Conclusion

The **Linear Support Vector Machine (SVC)** model achieved the highest individual test accuracy of **0.8792**, while the **Stacking Classifier** also performed strongly at **0.8708** accuracy. The performance indicates that LinearSVC was the most effective single model for this specific TF-IDF feature set.
