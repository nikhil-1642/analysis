import streamlit as st
from gtts import gTTS
import os
import io
import numpy as np
from PIL import Image
import speech_recognition as sr
from pydub import AudioSegment

# NLP imports
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from collections import Counter
import re

# Try to import heavy modules (optional)
try:
    from rembg import remove
except ImportError:
    remove = None

try:
    from deepface import DeepFace
except ImportError:
    DeepFace = None

# --- Download nltk resources ---
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# --- Streamlit config ---
st.set_page_config(page_title="🧠 Unstructured Data Analysis", layout="wide")
st.title("🧠 Unstructured Data Analysis Platform")

# Tabs for each data type
tab1, tab2, tab3 = st.tabs(["🖼️ Image Analysis", "🎧 Audio Analysis", "📝 Text + NLP Analysis"])

# ========== IMAGE ANALYSIS ==========
with tab1:
    st.header("🖼️ Image Analysis")
    img_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if img_file:
        image = Image.open(img_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Background removal
        if remove and st.button("🪄 Remove Background"):
            with st.spinner("Processing background removal..."):
                img_bytes = io.BytesIO()
                image.save(img_bytes, format="PNG")
                output = remove(img_bytes.getvalue())
                result_img = Image.open(io.BytesIO(output))
                st.image(result_img, caption="Background Removed", use_container_width=True)

        # Face analysis
        if DeepFace and st.button("🔍 Analyze Faces"):
            with st.spinner("Analyzing faces..."):
                try:
                    analysis = DeepFace.analyze(np.array(image), actions=["emotion", "age", "gender"], enforce_detection=False)
                    st.subheader("🧩 Face Analysis Results")
                    st.json(analysis[0] if isinstance(analysis, list) else analysis)
                except Exception as e:
                    st.error(f"Error during face analysis: {e}")

# ========== AUDIO ANALYSIS ==========
with tab2:
    st.header("🎧 Audio Analysis")
    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])

    if audio_file:
        st.audio(audio_file, format="audio/wav")
        recognizer = sr.Recognizer()

        # Convert to wav if needed
        if audio_file.type != "audio/wav":
            sound = AudioSegment.from_file(audio_file)
            wav_buffer = io.BytesIO()
            sound.export(wav_buffer, format="wav")
            wav_buffer.seek(0)
        else:
            wav_buffer = audio_file

        # Speech-to-text
        if st.button("🗣️ Convert Speech to Text"):
            with sr.AudioFile(wav_buffer) as source:
                audio_data = recognizer.record(source)
                try:
                    text = recognizer.recognize_google(audio_data)
                    st.success("✅ Speech Transcription Successful")
                    st.text_area("Transcribed Text", text, height=150)
                except Exception as e:
                    st.error(f"Speech recognition failed: {e}")

# ========== TEXT + NLP ANALYSIS ==========
with tab3:
    st.header("📝 Text + NLP Analysis")

    text_input = st.text_area("Enter text for analysis", height=200)
    col1, col2 = st.columns(2)

    with col1:
        if st.button("💬 Sentiment Analysis"):
            if text_input.strip():
                blob = TextBlob(text_input)
                sentiment = blob.sentiment
                st.write("**Polarity:**", sentiment.polarity)
                st.write("**Subjectivity:**", sentiment.subjectivity)
                if sentiment.polarity > 0:
                    st.success("🙂 Positive Sentiment")
                elif sentiment.polarity < 0:
                    st.error("☹️ Negative Sentiment")
                else:
                    st.info("😐 Neutral Sentiment")

        if st.button("🔊 Text to Speech"):
            if text_input.strip():
                tts = gTTS(text=text_input, lang='en')
                tts.save("tts_output.mp3")
                st.audio("tts_output.mp3")
                st.success("✅ Audio Generated Successfully")

    with col2:
        if st.button("🧾 Summarize Text"):
            words = text_input.split()
            if len(words) < 50:
                st.warning("Text too short to summarize meaningfully.")
            else:
                summary = " ".join(words[:len(words)//2]) + "..."
                st.write("**Summary:**")
                st.write(summary)

        if st.button("🧠 Keyword Extraction"):
            text_clean = re.sub(r'[^A-Za-z\s]', '', text_input.lower())
            tokens = nltk.word_tokenize(text_clean)
            filtered = [w for w in tokens if w not in stopwords.words('english')]
            freq = Counter(filtered)
            st.subheader("📊 Top Keywords")
            st.write(freq.most_common(10))
