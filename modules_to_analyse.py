import streamlit as st
from gtts import gTTS
import os
import speech_recognition as sr
from pydub import AudioSegment
import io
import numpy as np
from PIL import Image

# Try importing optional modules
try:
    from rembg import remove
except ImportError:
    remove = None

try:
    from deepface import DeepFace
except ImportError:
    DeepFace = None

st.title("🧠 Unstructured Data Analysis")

tab1, tab2, tab3 = st.tabs(["🖼️ Image Analysis", "🎧 Audio Analysis", "📝 Text Analysis"])

# ---------------------- IMAGE ANALYSIS ----------------------
with tab1:
    st.title("🖼️ Image Analysis")

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        img = Image.open(uploaded_image).convert("RGB")
        st.image(img, caption="Uploaded Image", width=300)
        img_array = np.array(img)

        col1, col2, col3, col4 = st.columns(4)

        # Face detection
        with col1:
            if st.button("Detect Face"):
                if DeepFace:
                    try:
                        detection = DeepFace.detectFace(img_array, enforce_detection=True)
                        st.success("✅ Face detected!")
                        st.image(detection, caption="Detected Face", use_column_width=True)
                    except Exception as e:
                        st.error(f"Detection failed: {e}")
                else:
                    st.warning("DeepFace not available on this platform.")

        # Age & Gender
        with col2:
            if st.button("Detect Age & Gender"):
                if DeepFace:
                    try:
                        analysis = DeepFace.analyze(img_array, actions=['age', 'gender'], enforce_detection=True)
                        st.success("✅ Age & Gender detected!")
                        st.write(f"**Age:** {analysis[0]['age']}")
                        st.write(f"**Gender:** {analysis[0]['dominant_gender']}")
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")
                else:
                    st.warning("DeepFace not available.")

        # Emotion
        with col3:
            if st.button("Detect Emotion"):
                if DeepFace:
                    try:
                        analysis = DeepFace.analyze(img_array, actions=['emotion'], enforce_detection=True)
                        st.success("✅ Emotion detected!")
                        st.write(f"**Emotion:** {analysis[0]['dominant_emotion']}")
                    except Exception as e:
                        st.error(f"Emotion detection failed: {e}")
                else:
                    st.warning("DeepFace not available.")

        # Background Removal
        with col4:
            if remove:
                output_image = remove(img)
                st.image(output_image, caption="BG Removed", width=300)
            else:
                st.warning("Background removal not supported in this environment.")

# ---------------------- AUDIO ANALYSIS ----------------------
with tab2:
    st.header("🗣️ Text to Speech")
    text = st.text_area("Enter text to convert to speech:")
    if st.button("Convert to Audio"):
        if text.strip():
            tts = gTTS(text, lang='en')
            tts.save("output.mp3")
            st.audio("output.mp3")
        else:
            st.warning("Please enter some text.")

    st.header("🎧 Speech to Text")
    uploaded_audio = st.file_uploader("Upload audio file (wav, mp3, m4a)", type=["wav","mp3","m4a"])
    if uploaded_audio:
        audio_bytes = uploaded_audio.read()
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        wav_io = io.BytesIO()
        audio.export(wav_io, format="wav")
        wav_io.seek(0)
        st.audio(wav_io, format="audio/wav")

        if st.button("Transcribe Audio"):
            recognizer = sr.Recognizer()
            wav_io.seek(0)
            with sr.AudioFile(wav_io) as source:
                audio_data = recognizer.record(source)
            with st.spinner("Transcribing..."):
                try:
                    text_output = recognizer.recognize_google(audio_data)
                    st.success("✅ Transcription complete!")
                    st.write(text_output)
                except Exception as e:
                    st.error(f"Error: {e}")
