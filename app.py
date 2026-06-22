# app.py
import streamlit as st
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# UI
st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detector")
st.subheader("Powered by Machine Learning")

news = st.text_area("Paste your news headline or article here:", height=200)

if st.button("🔍 Detect"):
    if news.strip() == "":
        st.warning("Please enter some text!")
    else:
        vec = vectorizer.transform([news])
        result = model.predict(vec)[0]
        confidence = model.predict_proba(vec)[0]

        if result == 1:
            st.success(f"✅ REAL NEWS — Confidence: {confidence[1]*100:.1f}%")
        else:
            st.error(f"🚨 FAKE NEWS — Confidence: {confidence[0]*100:.1f}%")
