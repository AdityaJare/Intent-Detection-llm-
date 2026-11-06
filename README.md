# 🎯 Intent Detection with DistilBERT

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20%26%20Tokenizer-yellow.svg)](https://huggingface.co/adityajare/intent-detection-distilbert)
[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Training%20Notebook-blue?logo=kaggle)](https://www.kaggle.com/code/adityajare/intent-detection-distilbert)
[![Python Version](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-4.40%2B-orange)](https://huggingface.co/docs/transformers)
[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Natural Language Understanding (NLU) solution for **Intent Detection** powered by a fine-tuned **DistilBERT** transformer model. The model classifies spoken or typed user queries into specific action intents with high accuracy and low latency, accompanied by an interactive **Streamlit** web application for real-time testing.

---

## 📌 Project Links

- **🤗 Hugging Face Model & Tokenizer:** [adityajare/intent-detection-distilbert](https://huggingface.co/adityajare/intent-detection-distilbert)
- **📓 Kaggle Training Notebook:** [Intent Detection with DistilBERT on Kaggle](https://www.kaggle.com/code/adityajare/intent-detection-distilbert)

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Supported Intents](#-supported-intents)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation (Using pip / venv)](#1-installation-using-pip--venv)
  - [Installation (Using uv)](#2-installation-using-uv)
- [Running the Web Application](#-running-the-web-application)
- [Programmatic Usage (Python)](#-programmatic-usage-python)
  - [Using Hugging Face Pipeline](#option-a-using-transformers-pipeline)
  - [Using PyTorch & AutoModel](#option-b-using-pytorch--automodel)
- [Model Details & Training](#-model-details--training)
- [Deployment](#-deployment)
- [Contributing & License](#-license)

---

## 🌟 Overview

Intent classification is a fundamental building block in modern conversational AI, intelligent personal assistants (e.g., Alexa, Siri, Google Assistant), and customer support bots. 

This project fine-tunes **DistilBERT** (`distilbert-base-uncased`) on the benchmark intent detection dataset. DistilBERT reduces BERT's size by 40% while retaining 97% of its language understanding capabilities and running 60% faster, making it an ideal choice for latency-critical intent prediction tasks.

---

## ✨ Key Features

- **⚡ Fast & Lightweight:** Powered by DistilBERT for low-latency inference on both CPU and GPU.
- **🤗 Hub Integration:** Model checkpoints and tokenizer files are hosted directly on Hugging Face Hub for seamless loading.
- **🖥️ Interactive UI:** Built-in Streamlit web interface for testing queries with instant feedback, predicted intent labels, and confidence probability scores.
- **📊 Robust Classification:** Pre-trained on diverse real-world intent categories (music, weather, bookings, search, and ratings).
- **🛠️ Production Ready:** Easy to embed into chatbot pipelines, microservices, or FastAPI/Flask endpoints.

---

## 🏷️ Supported Intents

The fine-tuned model classifies input text into **7 intent categories**:

| Label ID | Intent Name | Description / Sample Query |
| :---: | :--- | :--- |
| `0` | **`AddToPlaylist`** | *"add this track to my workout playlist"* |
| `1` | **`BookRestaurant`** | *"book a table at the italian restaurant for tonight"* |
| `2` | **`GetWeather`** | *"what is the weather in New York this weekend?"* |
| `3` | **`PlayMusic`** | *"play some relaxing lo-fi jazz music"* |
| `4` | **`RateBook`** | *"give this book 5 out of 5 stars"* |
| `5` | **`SearchCreativeWork`**| *"search for the movie Inception"* |
| `6` | **`SearchScreeningEvent`** | *"find movie screening times for Avatar near me"* |

---

## 📂 Project Structure

```text
IntentDetectionLLM/
├── main.py              # Streamlit web application & inference UI
├── pyproject.toml       # Project configuration and package dependencies
├── uv.lock              # Reproducible lockfile (uv package manager)
├── .python-version      # Target Python version specification (3.12)
├── .gitignore           # Git ignore rules
└── README.md            # Project documentation and guide
```

---

## 🚀 Getting Started

### Prerequisites

- **Python:** 3.10 or higher (Python 3.12 recommended)
- **Git:** Installed on your local machine
- (Optional) CUDA-enabled GPU for accelerated inference / fine-tuning

### 1. Installation (Using `pip` / `venv`)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/IntentDetectionLLM.git
   cd IntentDetectionLLM
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # On macOS/Linux
   python -m venv .venv
   source .venv/bin/activate

   # On Windows (PowerShell)
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```bash
   pip install streamlit torch transformers
   ```

### 2. Installation (Using `uv`)

If you prefer [`uv`](https://github.com/astral-sh/uv) (fast Python package installer):

```bash
uv venv
# Activate the venv as shown above, then sync
uv sync
```

---

## 🖥️ Running the Web Application

Launch the Streamlit app with:

```bash
streamlit run main.py
```

Once running, open your browser at `http://localhost:8501`. 

### What You Can Do in the App:
- Enter any custom natural language prompt (e.g., *"reserve a table for four at 7 PM"*).
- Click **Predict Intent** to see the predicted intent category with confidence percentage.
- Expand the **Probability Breakdown** to inspect softmax scores across all 7 classes.

---

## 💻 Programmatic Usage (Python)

You can load and use the model in your own Python projects or API endpoints:

### Option A: Using Transformers Pipeline

```python
from transformers import pipeline

# Load pipeline directly from Hugging Face Hub
classifier = pipeline(
    "text-classification",
    model="adityajare/intent-detection-distilbert",
    tokenizer="adityajare/intent-detection-distilbert"
)

result = classifier("play some jazz songs")
print(result)
# Output: [{'label': 'LABEL_3', 'score': 0.998}]
```

### Option B: Using PyTorch & AutoModel

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Intent label mapping
INTENT_LABELS = {
    0: "AddToPlaylist",
    1: "BookRestaurant",
    2: "GetWeather",
    3: "PlayMusic",
    4: "RateBook",
    5: "SearchCreativeWork",
    6: "SearchScreeningEvent",
}

repo_id = "adityajare/intent-detection-distilbert"

tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = AutoModelForSequenceClassification.from_pretrained(repo_id)
model.eval()

query = "reserve a table at the rooftop restaurant tonight"

# Note: Set return_token_type_ids=False since DistilBERT does not require token_type_ids
inputs = tokenizer(
    query,
    return_tensors="pt",
    truncation=True,
    padding=True,
    return_token_type_ids=False
)

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
probabilities = torch.softmax(logits, dim=-1)[0]
pred_class_id = torch.argmax(probabilities).item()
confidence = probabilities[pred_class_id].item()

print(f"Query: {query}")
print(f"Predicted Intent: {INTENT_LABELS[pred_class_id]}")
print(f"Confidence: {confidence * 100:.2f}%")
```

---

## 🔬 Model Details & Training

- **Base Architecture:** `distilbert-base-uncased`
- **Model Checkpoints:** [adityajare/intent-detection-distilbert](https://huggingface.co/adityajare/intent-detection-distilbert)
- **Training Environment:** Kaggle Notebook with GPU acceleration ([view training notebook](https://www.kaggle.com/code/adityajare/intent-detection-distilbert))
- **Objective:** Cross-Entropy loss over 7 intent classes
- **Key Hyperparameters:**
  - Optimizer: `AdamW`
  - Max Sequence Length: 64 tokens
  - Evaluation Metric: Accuracy, Macro F1-Score

---

## ☁️ Deployment

### Hugging Face Spaces (Streamlit)
1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces) with SDK set to **Streamlit**.
2. Push `main.py` and `pyproject.toml` (or `requirements.txt`).
3. Your application is live instantly with free cloud hosting!

### Streamlit Community Cloud
1. Push your repository to GitHub.
2. Visit [share.streamlit.io](https://share.streamlit.io) and connect your repository.
3. Set the entrypoint to `main.py` and click **Deploy**.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 👤 Author

**Aditya Jare**
- Hugging Face: [@adityajare](https://huggingface.co/adityajare)
- Kaggle: [@adityajare](https://www.kaggle.com/adityajare)

---

⭐ *If you find this project useful, don't forget to star the repository and heart the model on Hugging Face!*

