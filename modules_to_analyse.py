import streamlit as st
from gtts import gTTS
import os
import speech_recognition as sr
from pydub import AudioSegment
import io
import numpy as np
from PIL import Image

# Optional heavy libs (use try/except for Streamlit Cloud safety)
try:
    from rembg import remove
except ImportError:
    remove = None

try:
    from deepface import DeepFace
except ImportError:
    DeepFace = None

st.set_page_config(page_title="Unstructured Data Analysis", layout="wide")
st.title("🧠 Unstructured Data Analysis")

tab1, tab2, tab3 = st.tabs(["🖼️ Image Analysis", "🎧 Audio Analysis", "📝 Text Analysis"])

# ========== IMAGE ANALYSIS TAB ==========
with tab1:
    st.header("Image Analysis")

    uploaded_img = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_img:
        image = Image.open(uploaded_img)
        st.image(image, caption="Original Image", use_container_width=True)

        # Background removal
        if remove:
            if st.button("🪄 Remove Background"):
                with st.spinner("Removing background..."):
                    img_bytes = io.BytesIO()
                    image.save(img_bytes, format="PNG")
                    output = remove(img_bytes.getvalue())
                    result_img = Image.open(io.BytesIO(output))
                    st.image(result_img, caption="Background Removed", use_container_width=True)
        else:
            st.warning("⚠️ rembg not available. Skipping background removal.")

        # Face analysis
        if DeepFace:
            if st.button("🔍 Analyze Faces"):
                with st.spinner("Analyzing faces..."):
                    try:
                        analysis = DeepFace.analyze(np.array(image), actions=["emotion", "age", "gender"], enforce_detection=False)
                        st.subheader("🧩 Face Analysis Results")
                        st.json(analysis[0] if isinstance(analysis, list) else analysis)
                    except Exception as e:
                        st.error(f"Error during face analysis: {e}")
        else:
            st.warning("⚠️ DeepFace not available. Skipping face analysis.")

# ========== AUDIO ANALYSIS TAB ==========
with tab2:
    st.header("Audio Analysis")

    uploaded_audio = st.file_uploader("Upload audio file", type=["mp3", "wav", "m4a"])
    if uploaded_audio:
        st.audio(uploaded_audio, format="audio/wav")
        recognizer = sr.Recognizer()

        # Convert to wav if needed
        if uploaded_audio.type != "audio/wav":
            sound = AudioSegment.from_file(uploaded_audio)
            wav_buffer = io.BytesIO()
            sound.export(wav_buffer, format="wav")
            wav_buffer.seek(0)
        else:
            wav_buffer = uploaded_audio

        # Transcribe audio
        if st.button("🗣️ Convert Speech to Text"):
            with sr.AudioFile(wav_buffer) as source:
                audio_data = recognizer.record(source)
                try:
                    text = recognizer.recognize_google(audio_data)
                    st.success("✅ Transcription Complete")
                    st.write(text)
                except Exception as e:
                    st.error(f"Speech recognition failed: {e}")

# ========== TEXT ANALYSIS TAB ==========
with tab3:
    st.header("Text-to-Speech Converter")

    input_text = st.text_area("Enter text to convert to speech", height=150)
    if st.button("🔊 Convert to Speech"):
        if input_text.strip():
            tts = gTTS(text=input_text, lang='en')
            tts.save("speech.mp3")
            st.audio("speech.mp3")
            st.success("✅ Audio Generated Successfully")
        else:
            st.warning("Please enter text to convert.")
