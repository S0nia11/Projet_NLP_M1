import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import base64

def set_background(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
    css = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_background("app/App.png")

model_path = "models/emotion_distilbert"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

emotion_labels = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]

st.title("Détecteur d'Émotions")
text = st.text_area("Saisis un texte ici :", "")

if st.button("Prédire"):
    if text.strip():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        threshold = 0.5
        predicted = (probs >= threshold).astype(int)

        st.subheader("Émotions détectées")
        any_emotion = False
        for emotion, score, pred in zip(emotion_labels, probs, predicted):
            if pred:
                st.markdown(f"- **{emotion}** (score : `{score:.2f}`)")
                any_emotion = True

        if not any_emotion:
            st.info("Aucune émotion détectée avec un score supérieur à 0.5.")
    else:
        st.warning("Veuillez saisir un texte.")
