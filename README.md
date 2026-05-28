# Mental Health Sentiment Analysis using DistilBERT and PEFT

## Overview

This project focuses on Mental Health Sentiment Analysis using a fine-tuned pretrained DistilBERT model with Parameter-Efficient Fine-Tuning (PEFT). The objective is to classify mental health related textual data into multiple sentiment/status categories using Natural Language Processing (NLP) and Deep Learning techniques.

The project explores transformer-based language models for efficient text classification while reducing computational and memory requirements through PEFT.

---

## Features

* Pretrained DistilBERT transformer model
* Parameter-Efficient Fine-Tuning (PEFT)
* Multi-class text classification
* NLP preprocessing pipeline
* Embedding generation and analysis
* Model evaluation and performance metrics
* Scalable and efficient training approach

---

## Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* PEFT
* Scikit-learn
* Pandas
* NumPy
* Jupyter Notebook

---

## Project Structure

```bash
Mental_Health_Sentiment_Analysis/
│
├── Dataset/
├── notebooks/
├── models/
├── Output/
├── requirements.txt
├── README.md
└── src/
```

---

## Dataset

The dataset contains textual data related to mental health sentiment/status classification.

Note:
Large datasets and generated embedding files were excluded from GitHub due to repository size limitations.

---

## Model

This project uses:

* DistilBERT as the pretrained transformer model
* PEFT (Parameter-Efficient Fine-Tuning) for efficient adaptation

The model was trained and evaluated on mental health textual data to classify sentiment/status categories.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/hassan335/Mental_Health_Sentiment_Analysis.git
```

Navigate to the project directory:

```bash
cd Mental_Health_Sentiment_Analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the training script:

```bash
python train.py
```

Run inference/testing:

```bash
python predict.py
```

---

## Results

The model demonstrated effective performance in mental health sentiment classification using transformer-based embeddings and PEFT optimization techniques.

Evaluation metrics may include:

* Accuracy
* Precision
* Recall
* F1-Score

---

## Future Improvements

* Hyperparameter tuning
* Larger dataset integration
* Model deployment using Flask or FastAPI
* Real-time sentiment analysis dashboard
* Additional transformer model comparisons

---

## Author

Mohammad Hassan

---

## License

This project is intended for educational and research purposes.
