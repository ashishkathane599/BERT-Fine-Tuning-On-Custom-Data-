# BERT Fine-Tuning on Custom Data

This repository contains notebooks and datasets for fine-tuning **BERT** and **DistilBERT** models using the [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) library. The project focuses on text classification tasks, specifically **SMS Spam Detection** and **Sentiment Analysis**.

## 📁 Repository Structure

* **`BERT.ipynb`**: Main notebook for fine-tuning the base BERT model for sequence classification.
* **`Sentiment_analysisj_BERT.ipynb`**: Notebook dedicated to training BERT specifically for sentiment analysis tasks.
* **`fine_tune_DistiBERT.ipynb`**: A lightweight implementation using DistilBERT, optimized for faster training and lower memory usage.
* **`SMSSpamCollection.csv` / `.txt`**: The raw dataset used for training the spam detection classifier.

## 🚀 Getting Started

### 1. Prerequisites
You will need Python installed along with the following libraries:
```bash
pip install transformers datasets torch pandas scikit-learn
```

### 2. Dataset
The project utilizes the `SMSSpamCollection` dataset, which contains SMS messages labeled as `ham` (legitimate) or `spam`. 
* **Preprocessing:** The text is cleaned and tokenized using the `BertTokenizer`.
* **Loading:** Data is processed via Pandas and converted into Hugging Face `Dataset` objects for compatibility with the Trainer API.

### 3. Training Workflow
The fine-tuning process generally follows these steps:
1.  **Model Initialization:** Loading `BertForSequenceClassification` with pre-trained weights from `bert-base-uncased`.
2.  **Tokenization:** Converting text into `input_ids` and `attention_mask` tensors.
3.  **Hyperparameters:** * **Learning Rate:** $2 \times 10^{-5}$
    * **Batch Size:** 16 or 32
    * **Epochs:** 3-5 (standard for BERT fine-tuning)
4.  **Trainer API:** Leveraging the `Trainer` class to manage the training loop, evaluation, and model checkpoints.

## 📊 Evaluation
The models are evaluated on a held-out test set. Key metrics tracked include:
* **Accuracy**
* **Precision & Recall**
* **F1-Score** (especially important for the imbalanced Spam dataset)

## 🛠️ Usage
To replicate the results, open the desired notebook in **Jupyter Notebook** or **Google Colab**. 

```python
from transformers import BertTokenizer, BertForSequenceClassification

# Quick load example
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
```

## 📜 License
This project is open-source. Feel free to use the code for your own research or production-level NLP applications.

