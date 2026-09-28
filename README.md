<div align="center">

# 📨 Customer Complaint Classifier

**Classify real consumer complaints into product categories using classic NLP: Text Cleaning → TF-IDF → Logistic Regression**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-NLP-2E7D32)
![Accuracy](https://img.shields.io/badge/Accuracy-~80%25-brightgreen)
![Status](https://img.shields.io/badge/Status-Complete-blue)

</div>

---

## 📌 Overview

This project builds a **multi-class text classifier** that reads a customer's complaint and predicts which financial **product category** it belongs to (one of **9 categories**, such as credit reporting, debt collection, mortgage, student loan, and more). This kind of model is used in real support systems to **route complaints automatically** to the right team.

The project applies core NLP concepts: **text preprocessing, Bag-of-Words / TF-IDF vectorization, and a linear classifier**, on real-world data.

## 🏆 Results

| Metric | Score |
|---|---|
| **Model** | Logistic Regression |
| **Features** | TF-IDF (max 20,000 features) |
| **Test Accuracy** | **~80%** |
| **Train / Test split** | 80% / 20% (stratified) |

> Baseline for a random guess is `1 / number_of_classes`, so ~80% means the model learned strong, real signal from the complaint text.

<!-- Optional: add a confusion matrix image here -->
<!-- ![Confusion Matrix](images/confusion_matrix.png) -->

## 🗂️ Dataset

- **Source:** [CFPB Consumer Complaint Database](https://www.kaggle.com/datasets/selener/consumer-complaint-database) (Kaggle), real complaints published by the U.S. Consumer Financial Protection Bureau.
- **Columns used:** `Product` (label) and `Consumer complaint narrative` (text).
- **Sample used for training:** ~20,000 complaints, sampled to be balanced across product categories.
- The raw file is very large (millions of rows), so it is **not included** in this repo. Download it from Kaggle and place it as `rows.csv`.

### 🏷️ Categories the model was trained on

The model classifies each complaint into one of **9 product categories**:

| # | Category |
|---|---|
| 1 | Checking or savings account |
| 2 | Credit card or prepaid card |
| 3 | Credit reporting, credit repair services, or other personal consumer reports |
| 4 | Debt collection |
| 5 | Money transfer, virtual currency, or money service |
| 6 | Mortgage |
| 7 | Payday loan, title loan, or personal loan |
| 8 | Student loan |
| 9 | Vehicle loan or lease |

> The 20,000-complaint sample was drawn evenly across these categories (up to ~2,222 complaints each) so no single class dominates training.

## ⚙️ Pipeline

```
Raw complaint text
      │
      ▼
 Cleaning  ── lowercase → remove URLs → HTML tags → mentions → CFPB "XXXX" redactions
      │        → punctuation → numbers → extra whitespace → stopword removal (NLTK)
      ▼
 Label encoding (LabelEncoder)
      │
      ▼
 Train / test split (80/20, stratified)
      │
      ▼
 TF-IDF vectorizer  (fit on train only, no data leakage)
      │
      ▼
 Logistic Regression
      │
      ▼
 Evaluation: accuracy, precision, recall, F1
```

### Key design choices

- **No data leakage:** the TF-IDF vocabulary is learned from the training set only; the test set is only transformed.
- **CFPB-specific cleaning:** complaints contain `XXXX` placeholders for redacted names and dates, which are removed as noise.
- **Duplicates and empty texts removed** before training.
- **Stratified split** keeps class proportions identical in train and test.

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/hammad268/customer-complaint-classifier-nlp.git
cd customer-complaint-classifier-nlp
```

### 2. Install dependencies

```bash
pip install pandas scikit-learn nltk joblib
```

### 3. Add the dataset

Download `rows.csv` from the [Kaggle dataset](https://www.kaggle.com/datasets/selener/consumer-complaint-database) and put it in the project folder.

### 4. Run

```bash
python complaint_classifier.py
```

This will build the 20k sample (first run only), clean the text, train the model, print the classification report, and save the trained model files.

## 💻 Final Code

<details>
<summary><b>Click to expand the full script</b></summary>

```python
import re
import joblib
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

RAW_FILE = "rows.csv"
SAMPLE_FILE = "complaints_20k.csv"
SAMPLE_SIZE = 20_000
RANDOM_STATE = 42

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)


# 1. Load + sample
def build_sample():
    df = pd.read_csv(
        RAW_FILE,
        usecols=["Product", "Consumer complaint narrative"],
        engine="python",
        on_bad_lines="skip",
    )
    df = df[df["Consumer complaint narrative"].notnull()]
    per_class = SAMPLE_SIZE // df["Product"].nunique()
    sample = df.groupby("Product", group_keys=False).apply(
        lambda x: x.sample(min(len(x), per_class), random_state=RANDOM_STATE)
    )
    sample.to_csv(SAMPLE_FILE, index=False)
    return sample


try:
    df = pd.read_csv(SAMPLE_FILE)
except FileNotFoundError:
    df = build_sample()

df.columns = ["product", "text"]
df = df.dropna(subset=["text"]).drop_duplicates(subset=["text"])

# 2. Text cleaning
stop_words = set(stopwords.words("english"))


def remove_stopwords(text):
    return " ".join(w for w in word_tokenize(text) if w not in stop_words)


df["clean"] = df["text"].str.lower()
df["clean"] = df["clean"].str.replace(r"https?://\S+|www\.\S+", "", regex=True)
df["clean"] = df["clean"].str.replace(r"<.*?>", "", regex=True)
df["clean"] = df["clean"].str.replace(r"@\w+", "", regex=True)
df["clean"] = df["clean"].str.replace(r"x{2,}", "", regex=True)
df["clean"] = df["clean"].str.replace(r"[^\w\s]", "", regex=True)
df["clean"] = df["clean"].str.replace(r"\d+", "", regex=True)
df["clean"] = df["clean"].str.replace(r"\s+", " ", regex=True).str.strip()
df["clean"] = df["clean"].apply(remove_stopwords)
df = df[df["clean"].str.strip() != ""]

print(df["product"].value_counts())

# 3. Encode labels + split
le = LabelEncoder()
df["label"] = le.fit_transform(df["product"])

X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["label"],
    test_size=0.2, random_state=RANDOM_STATE, stratify=df["label"],
)

# 4. TF-IDF + Logistic Regression
tfidf = TfidfVectorizer(max_features=20_000)
X_train_tf = tfidf.fit_transform(X_train)   # fit on train only
X_test_tf = tfidf.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tf, y_train)

# 5. Evaluate
pred = model.predict(X_test_tf)
print(f"Accuracy: {accuracy_score(y_test, pred):.4f}")
print(classification_report(y_test, pred, target_names=le.classes_))

# 6. Save artifacts
joblib.dump(model, "model.joblib")
joblib.dump(tfidf, "tfidf.joblib")
joblib.dump(le, "label_encoder.joblib")
```

</details>

The full version (with the `predict_complaint()` helper) is in [`complaint_classifier.py`](complaint_classifier.py).

## 🔮 Predicting on New Text

```python
print(predict_complaint("I was charged a late fee even though I paid on time."))
# -> predicted product category
```

## 📁 Project Structure

```
customer-complaint-classifier-nlp/
├── complaint_classifier.py   # full pipeline: sample → clean → train → evaluate → save
├── README.md
├── requirements.txt
└── (generated) model.joblib, tfidf.joblib, label_encoder.joblib
```

## 🧠 What I Learned

- Why **data quality decides everything**: an earlier attempt on a templated support-ticket dataset stayed near random-guess accuracy (~20%) with four different algorithms, while real complaint text reached ~80%.
- How **Bag-of-Words, TF-IDF, and one-hot encoding** differ and when each applies.
- Avoiding **data leakage** by fitting vectorizers on training data only.
- Cleaning noisy real-world text with regex + NLTK.

## 🔭 Future Improvements

- Compare **Bag-of-Words vs TF-IDF** and other models (Naive Bayes, Linear SVM, Random Forest).
- Add **n-grams** (`ngram_range=(1, 2)`) and hyperparameter tuning.
- Merge overlapping categories (e.g. the credit reporting group) to lift per-class F1.
- Try transformer models (e.g., DistilBERT) for higher accuracy.
- Deploy as a small **Streamlit / FastAPI** app.

## 👤 Author

**Hammad**, Computer Science student & independent developer
GitHub: [@hammad268](https://github.com/hammad268)

---

<div align="center">⭐ If you found this useful, consider starring the repo!</div>
