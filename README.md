# 📰 Bangla News Article Classifier

**📘 Kaggle Notebook Reference:**  
https://www.kaggle.com/code/jvedsaqib/minor-project-classification  

**📎 Detailed Documentation (Google Drive):**  
https://drive.google.com/file/d/1V3aPWrFe8x2uyxQdZfenYAxX-uGiY_rG/view?usp=sharing  

---

## Table of Contents

- [Motivation](#motivation)  
- [Datasets](#datasets)  
- [Preprocessing Pipeline](#preprocessing-pipeline)  
- [Feature Engineering](#feature-engineering)  
- [Model Architecture](#model-architecture)  
- [Evaluation & Results](#evaluation--results)  
- [Technologies & Environment](#technologies--environment)  
- [Future Directions](#future-directions)  
- [Contributing](#contributing)  
- [References & Acknowledgements](#references--acknowledgements)  

---

## Motivation

With the surge of online Bangla news content, automatic classification becomes essential for **search**, **recommendation**, and **media analytics**.  
This project aims to build a robust pipeline to classify Bengali news articles into categories like:

> National · Politics · Education · Sports · Science · Kolkata · State · International

By combining traditional ML and deep learning approaches, we explore how to adapt NLP workflows for less-resourced languages like Bangla.

---

## Datasets

We built a consolidated corpus using multiple public datasets:

| Source | Description | Link |
|---|---|---|
| Shironaam (Hugging Face) | Bengali news headlines collection | https://huggingface.co/datasets/dialect-ai/shironaam |
| IndicNLP News Articles (Kaggle) | Full-length articles across topics | https://www.kaggle.com/datasets/csoham/classification-bengali-news-articles-indicnlp |
| Potrika Newspaper (Kaggle) | Rich coverage from Bangla press | https://www.kaggle.com/datasets/sabbirhossainujjal/potrika-bangla-newspaper-datasets |

These datasets were merged, cleaned, and rebalanced into a unified training corpus.

---

## Preprocessing Pipeline

Implemented across two notebooks:  
- `cleaning-potrika-dataset.ipynb`  
- `work-on-potrika-3-1.ipynb`  

Key steps:

- **Cleaning & Deduplication** – Remove duplicates, nulls, non-text noise  
- **Normalization** – Strip punctuation, numbers, mixed-language tokens  
- **Stopword Removal** – Using a Bengali stopword list (e.g. from stopwords-bn)  
- **Balancing** – Upsample or curate underrepresented classes  
- **Data Split**  
  - 50 % — Training  
  - 25 % — Validation  
  - 25 % — Test  

Each notebook’s workflow is documented with comments and intermediate outputs.

---

## Feature Engineering

### Statistical Features  
- **TF (Term Frequency)**  
- **IDF (Inverse Document Frequency)**  
- **TF-IDF Vectors**

These representations help classical ML models grasp token-level importance.

### Deep Learning Features  
- **Tokenizer → Sequence of Integers**  
- **Padding / Truncation** → Uniform length inputs for LSTM  

This bridges raw text to sequence models.

---

## Model Architecture

We trained and compared multiple models:

| Model | Framework | Purpose |
|---|---|---|
| Logistic Regression | scikit-learn | Baseline linear classifier |
| Naïve Bayes | scikit-learn | Probabilistic text classifier |
| Random Forest | scikit-learn | Ensemble tree-based method |
| SVM (Support Vector Machine) | scikit-learn | Margin-based classifier on vector space |
| LSTM | TensorFlow / Keras | Sequence-aware deep network for text |

Each model is wrapped in a training / evaluation script with hyperparameter tuning support.

---

## Evaluation & Results

We measured performance using:

- Accuracy  
- Precision  
- Recall  
- F1-score  

🏆 **Top Model**: LSTM — achieved ~**91 % accuracy** on test data.

In detailed analysis, classical models perform well on balanced categories; however, their performance degrades for underrepresented labels. Deep learning, particularly LSTM, handles contextual nuances better.

---

## Technologies & Environment

- **Language & Libraries**: Python, Pandas, NumPy  
- **ML / NLP**: scikit-learn, TensorFlow, Keras, NLTK  
- **Visualization**: Matplotlib, Seaborn, WordCloud  
- **Environments**:  
  - Jupyter Notebook  
  - Kaggle Notebook (see link above)  
  - Google Colab  

We recommend setting up a virtual environment and pinning package versions via `pip freeze` for reproducibility.

---

## Future Directions

1. **Mitigate Class Imbalance**  
   - SMOTE, data augmentation, or backtranslation  
2. **Pretrained Transformer Integration**  
   - Use models like BanglaBERT / IndicBERT / mT5  
3. **Deploy as a Real-Time Service**  
   - API / Web app for live classification  
4. **Support Multi-Label Classification**  
   - Articles with overlapping topics  
5. **Add Sentiment & Tone Analysis**  
   - Complement category labeling with sentiment tags  

---

## Contributing

Contributions, issue reports, and pull requests are welcome! 🎉  
Please follow these steps:

1. Fork the repo  
2. Create a feature or bug-fix branch  
3. Write tests / validation where needed  
4. Submit a pull request with a clear description  

Please adhere to the existing code style, add documentation for any new modules, and reference issues where applicable.

---

## References & Acknowledgements

- [Dialect AI Shironaam Dataset](https://huggingface.co/datasets/dialect-ai/shironaam)  
- [IndicNLP Bengali News Articles](https://www.kaggle.com/datasets/csoham/classification-bengali-news-articles-indicnlp)  
- [Potrika Bangla Newspaper Dataset](https://www.kaggle.com/datasets/sabbirhossainujjal/potrika-bangla-newspaper-datasets)  
- [Bengali Stopwords – stopwords-bn GitHub](https://github.com/stopwords-iso/stopwords-bn)  
- **Libraries & Tools**: scikit-learn, TensorFlow, Keras, NLTK  


---


[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  
[![Python Version](https://img.shields.io/badge/python-3.x-yellow.svg)](https://www.python.org/)  


