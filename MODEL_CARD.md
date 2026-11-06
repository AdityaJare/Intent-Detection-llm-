---
language:
- en
license: mit
tags:
- intent-detection
- text-classification
- distilbert
- natural-language-understanding
- nlu
- conversational-ai
- pytorch
- transformers
datasets:
- snips_built_in_intents
metrics:
- accuracy
- f1
pipeline_tag: text-classification
widget:
- text: "play some relaxing lo-fi music"
  example_title: "PlayMusic"
- text: "book a table at the nearest sushi restaurant for tonight"
  example_title: "BookRestaurant"
- text: "what is the weather forecast for tomorrow in New York"
  example_title: "GetWeather"
- text: "add this song to my workout playlist"
  example_title: "AddToPlaylist"
- text: "give this book 5 out of 5 stars"
  example_title: "RateBook"
- text: "search for the movie Inception"
  example_title: "SearchCreativeWork"
- text: "find movie screening times for Avatar"
  example_title: "SearchScreeningEvent"
---

# 🤖 Intent Detection DistilBERT

A fine-tuned **DistilBERT** (`distilbert-base-uncased`) sequence classification model for **Intent Detection** in Natural Language Understanding (NLU). 

The model classifies user queries into 7 distinct action intents with low inference latency, making it ideal for chatbots, voice assistants, and interactive conversational agents.

---

## 🔗 Project Resources

- **🌐 Live Demo (Streamlit App):** [intent-detection-llm-aj.streamlit.app](https://intent-detection-llm-aj.streamlit.app/)
- **💻 GitHub Repository:** [AdityaJare/Intent-Detection-llm-](https://github.com/AdityaJare/Intent-Detection-llm-)
- **📓 Kaggle Training Notebook:** [adityajare/intent-detection-distilbert](https://www.kaggle.com/code/adityajare/intent-detection-distilbert)

---

## 🏷️ Supported Intent Classes

| Class ID | Intent Name | Example Utterance |
| :---: | :--- | :--- |
| `0` | **`AddToPlaylist`** | *"add this track to my workout playlist"* |
| `1` | **`BookRestaurant`** | *"book a table at the italian restaurant for tonight"* |
| `2` | **`GetWeather`** | *"what is the weather in New York this weekend?"* |
| `3` | **`PlayMusic`** | *"play some relaxing lo-fi jazz music"* |
| `4` | **`RateBook`** | *"give this book 5 out of 5 stars"* |
| `5` | **`SearchCreativeWork`** | *"search for the movie Inception"* |
| `6` | **`SearchScreeningEvent`** | *"find movie screening times for Avatar near me"* |

---

## 🚀 Quick Start & Usage

### 1. Using Hugging Face `pipeline`

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="adityajare/intent-detection-distilbert",
    tokenizer="adityajare/intent-detection-distilbert"
)

result = classifier("play some jazz songs")
print(result)
```

### 2. Native PyTorch & Transformers

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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

query = "reserve a table for four at the rooftop restaurant tonight"

inputs = tokenizer(
    query,
    return_tensors="pt",
    truncation=True,
    padding=True,
    return_token_type_ids=False
)

with torch.no_grad():
    outputs = model(**inputs)

probabilities = torch.softmax(outputs.logits, dim=-1)[0]
pred_class_id = torch.argmax(probabilities).item()
confidence = probabilities[pred_class_id].item()

print(f"Predicted Intent: {INTENT_LABELS[pred_class_id]} ({confidence * 100:.2f}%)")
```

---

## ⚙️ Model Details

- **Base Architecture:** `distilbert-base-uncased`
- **Number of Parameters:** ~66 Million
- **Precision:** Float32
- **Hardware:** Trained on Kaggle GPU acceleration
- **Frameworks:** PyTorch, Hugging Face Transformers

---

## 📄 License

This model is licensed under the [MIT License](https://opensource.org/licenses/MIT).

## 👤 Author

**Aditya Jare**
- Hugging Face: [@adityajare](https://huggingface.co/adityajare)
- Kaggle: [@adityajare](https://www.kaggle.com/adityajare)
- GitHub: [@AdityaJare](https://github.com/AdityaJare)
