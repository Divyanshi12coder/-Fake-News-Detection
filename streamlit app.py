import streamlit as st
from src.preprocessing import clean_text
from src.utils import load_model

st.title("📰 Fake News Detector")

model = load_model("models/logistic_model.pkl")
vectorizer = load_model("models/tfidf_vectorizer.pkl")

text = st.text_area("Enter news article text:")
if st.button("Predict"):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    label = "REAL" if prediction == 1 else "FAKE"
    st.success(f"The article is predicted to be: **{label}**")