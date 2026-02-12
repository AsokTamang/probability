# 📧 Spam Detection using Naive Bayes (From Scratch)

## 📌 Overview
This project implements a **Spam Detection model from scratch** using the **Naive Bayes algorithm**.  
The goal is to classify emails as **Spam** or **Ham (Not Spam)** based on the words they contain, without relying on high-level ML libraries such as `sklearn`.

The notebook walks through the **entire ML pipeline**:
- Text preprocessing
- Word frequency modeling
- Probability estimation
- Laplace smoothing
- Prediction
- Model evaluation

---

## 🧠 Key Concepts Used
- Naive Bayes Theorem
- Conditional Probability
- Bag-of-Words Model
- Laplace (Add-One) Smoothing
- Handling unseen words
- Confusion Matrix metrics
- Accuracy evaluation

---

## Code File
spam_detection.ipynb

---

## ⚙️ How the Model Works

### 1️⃣ Text Preprocessing
- Lowercasing
- Tokenization
- Stopword removal
- Punctuation removal

This ensures the text is clean and standardized before modeling.

---

### 2️⃣ Training Phase
- Counts word frequencies separately for **spam** and **ham**
- Stores:
  - Word → class frequency
  - Total number of spam and ham emails

---

### 3️⃣ Probability Estimation
Using Naive Bayes assumption:

\[
P(\text{email} \mid \text{class}) = \prod P(\text{word} \mid \text{class})
\]

Laplace smoothing is applied to prevent zero probabilities.

---

### 4️⃣ Prediction
For each email:
- Compute likelihood for **spam**
- Compute likelihood for **ham**
- Assign the class with higher probability

---

### 5️⃣ Evaluation
Model performance is measured using:
- **True Positives**
- **True Negatives**
- **Accuracy**

\[
\text{Accuracy} = \frac{TP + TN}{\text{Total Observations}}
\]

---

## 📊 Results
The model outputs:
- Number of correctly classified spam emails
- Number of correctly classified ham emails
- Overall accuracy of the classifier

This gives a clear understanding of how well the model performs on unseen data.

---

## 🚀 Why This Project Matters
- Demonstrates **core ML fundamentals**
- Avoids black-box ML libraries
- Strengthens understanding of probabilistic models
- Excellent foundation for more advanced NLP models

---

## 🛠 Technologies Used
- Python
- NumPy
- Pandas
- NLTK
- Jupyter Notebook

---

## 📈 Future Improvements
- Use log-probabilities to avoid numerical underflow
- Add precision, recall, and F1-score
- Compare with `sklearn` Naive Bayes
- Extend to bigram or TF-IDF features





