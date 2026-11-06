import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

st.set_page_config(page_title="Intent Detection App", page_icon="🤖", layout="centered")

st.title("🤖 Intent Detection with DistilBERT")
st.write("Type a query below to classify its intent using your fine-tuned model.")

# Human-readable label mapping for the 7 trained intents
INTENT_LABELS = {
    0: "AddToPlaylist",
    1: "BookRestaurant",
    2: "GetWeather",
    3: "PlayMusic",
    4: "RateBook",
    5: "SearchCreativeWork",
    6: "SearchScreeningEvent",
}

@st.cache_resource
def load_model():
    repo_id = "adityajare/intent-detection-distilbert"
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForSequenceClassification.from_pretrained(repo_id)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

user_input = st.text_input("Enter your query:", "play some relaxing music")

if st.button("Predict Intent", type="primary"):
    if not user_input.strip():
        st.warning("Please enter a valid text query.")
    else:
        inputs = tokenizer(
            user_input,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_token_type_ids=False,
        )

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)[0]
        predicted_class_id = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class_id].item() * 100

        prediction = INTENT_LABELS.get(
            predicted_class_id,
            model.config.id2label.get(str(predicted_class_id), f"LABEL_{predicted_class_id}")
        )

        st.success(f"**Predicted Intent:** `{prediction}` (Confidence: {confidence:.2f}%)")

        with st.expander("📊 Detailed Probability Distribution"):
            for idx, prob in enumerate(probabilities):
                label_name = INTENT_LABELS.get(idx, f"Class {idx}")
                prob_pct = prob.item() * 100
                st.write(f"**{label_name}:** {prob_pct:.2f}%")
                st.progress(float(prob))