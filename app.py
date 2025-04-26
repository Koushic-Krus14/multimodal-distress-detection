import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
import torch
import torch.nn.functional as F
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import os

# ========== PAGE CONFIG ==========
st.set_page_config(page_title="Multimodal Distress Detection", layout="centered")

# ========== LOAD MODELS ==========
audio_model = load_model("models/Distress_model.h5")
audio_labels = ['male_stressful', 'female_stressful', 'male_non_stressful', 'female_non_stressful']
label_encoder = LabelEncoder()
label_encoder.fit(audio_labels)

tokenizer = AutoTokenizer.from_pretrained("bhadresh-savani/bert-base-go-emotion")
text_model = AutoModelForSequenceClassification.from_pretrained("bhadresh-savani/bert-base-go-emotion")
asr = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")

# ========== EMOTION GROUPS ==========
distress_emotions = ["anger", "annoyance", "disappointment", "disapproval", "disgust", "fear", "grief", "nervousness", "remorse", "sadness"]
non_distress_emotions = ["joy", "amusement", "approval", "caring", "curiosity", "desire", "excitement", "gratitude", "love", "optimism", "pride", "relief", "surprise", "admiration", "neutral"]

# ========== FEATURE EXTRACTION ==========
def extract_features(file_path, input_duration=3, sr=22050*2):
    X, sample_rate = librosa.load(file_path, res_type="kaiser_fast", duration=input_duration, sr=sr, offset=0.5)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
    return X, sample_rate, mfccs

# ========== VISUALIZATIONS ==========
def plot_waveform(X):
    fig, ax = plt.subplots()
    ax.plot(X)
    ax.set_title("Audio Waveform")
    ax.set_xlabel("Time (samples)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

def plot_mfcc(X, sr):
    mfccs = librosa.feature.mfcc(y=X, sr=sr, n_mfcc=13)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(mfccs, x_axis='time', ax=ax)
    fig.colorbar(img, ax=ax)  # Pass 'img' to colorbar
    ax.set_title("MFCCs")
    st.pyplot(fig)

def plot_spectrogram(X, sr):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(X)), ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(D, x_axis='time', y_axis='log', sr=sr, ax=ax)
    fig.colorbar(img, ax=ax)  # Pass 'img' to colorbar
    ax.set_title("Spectrogram")
    st.pyplot(fig)

def plot_chroma(X, sr):
    chroma = librosa.feature.chroma_stft(y=X, sr=sr)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', sr=sr, ax=ax)
    fig.colorbar(img, ax=ax)  # Pass 'img' to colorbar
    ax.set_title("Chromagram")
    st.pyplot(fig)

def plot_zcr(X):
    zcr = librosa.feature.zero_crossing_rate(y=X)
    fig, ax = plt.subplots()
    ax.plot(zcr[0])
    ax.set_title("Zero-Crossing Rate")
    st.pyplot(fig)

def plot_spectral_centroid(X, sr):
    spectral_centroid = librosa.feature.spectral_centroid(y=X, sr=sr)
    fig, ax = plt.subplots()
    ax.plot(spectral_centroid[0])
    ax.set_title("Spectral Centroid")
    st.pyplot(fig)

# ========== AUDIO PREDICTION ==========
def predict_audio_state(X, sr, mfccs):
    features = np.expand_dims(np.expand_dims(mfccs, axis=0), axis=2)
    pred = audio_model.predict(features)
    confidence = np.max(pred)
    label = label_encoder.inverse_transform([np.argmax(pred)])[0]
    state = "distress" if "stressful" in label else "non_distress"
    return label, state, confidence

# ========== TEXT PREDICTION ==========
def predict_text_state(file_path):
    asr_result = asr(file_path)
    text = asr_result['text']
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = text_model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)
    pred_class = torch.argmax(probs, dim=1).item()
    confidence = torch.max(probs).item()
    emotion = text_model.config.id2label[pred_class]

    if emotion in distress_emotions:
        state = "distress"
    elif emotion in non_distress_emotions:
        state = "non_distress"
    else:
        state = "unknown"

    return text, emotion, state, confidence

# ========== FUSION ==========
def multimodal_decision(audio_state, audio_conf, text_state, text_conf, threshold=0.75):
    if audio_conf >= threshold and text_conf >= threshold:
        return audio_state if audio_state == text_state else "uncertain"
    elif audio_conf >= threshold:
        return audio_state
    elif text_conf >= threshold:
        return text_state
    else:
        return "uncertain"

# ========== STREAMLIT UI ==========

st.title("🎧 Multimodal Distress Detection System")
st.markdown("Upload an audio file (.wav) to detect distress using audio and text models.")

audio_file = st.file_uploader("Upload a WAV file", type=["wav"])

if audio_file:
    with open("temp.wav", "wb") as f:
        f.write(audio_file.read())

    X, sr, mfccs = extract_features("temp.wav")

    st.audio("temp.wav")

    st.subheader("🎛️ Choose visualizations:")
    tabs = ["Waveform", "Spectrogram", "MFCCs", "Chroma", "ZCR", "Spectral Centroid"]
    selected_tab = st.selectbox("Choose a visualization:", tabs)

    if selected_tab == "Waveform":
        plot_waveform(X)
    elif selected_tab == "Spectrogram":
        plot_spectrogram(X, sr)
    elif selected_tab == "MFCCs":
        plot_mfcc(X, sr)
    elif selected_tab == "Chroma":
        plot_chroma(X, sr)
    elif selected_tab == "ZCR":
        plot_zcr(X)
    elif selected_tab == "Spectral Centroid":
        plot_spectral_centroid(X, sr)

    st.subheader("🔎 Predictions")
    audio_label, audio_state, audio_conf = predict_audio_state(X, sr, mfccs)
    st.write(f"**Audio Prediction:** {audio_label} → {audio_state} ({audio_conf:.2f})")

    text, emotion, text_state, text_conf = predict_text_state("temp.wav")
    st.write(f"**Transcribed Text:** `{text}`")
    st.write(f"**Text Emotion:** {emotion} → {text_state} ({text_conf:.2f})")

    final_state = multimodal_decision(audio_state, audio_conf, text_state, text_conf)
    st.success(f"✅ **Final Multimodal Prediction:** {final_state.upper()}")

    os.remove("temp.wav")
