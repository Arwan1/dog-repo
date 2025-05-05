import os
import uuid
import numpy as np
import sounddevice as sd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
import tensorflow as tf

# === CONFIG ===
AUDIO_DURATION = 2  # seconds
SAMPLE_RATE = 22050  # Hz

# === Classes ===
classes_dict = {
    0: 'Sad',
    1: 'Happy',
    2: 'Stress',
    3: 'Restless',
    4: 'Love',
    5: 'Lonely',
    6: 'Tired',
    7: 'Normal'
}

# === Load CNN Models ===
image_model_path = "C:/Users/OMOLP091/Documents/OMOTECH/RESEARCH/DOG-PATTERN-ARWAN_MAKHIJA/ISEF/MODELS/FULL_2_E_emotion_model_best.h5"
audio_model_path = "C:/Users/OMOLP091/Documents/OMOTECH/RESEARCH/DOG-PATTERN-ARWAN_MAKHIJA/ISEF/MODELS/EfficientNetB7_Model-FULL-2_emotion_model.h5"
image_model = tf.keras.models.load_model(image_model_path)
audio_model = tf.keras.models.load_model(audio_model_path)

# === Image Preprocessing ===
def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB").resize((180, 180))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0)

# === Audio Preprocessing (Spectrogram) ===
def preprocess_audio(audio, sr):
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db_resized = librosa.util.fix_length(mel_db, size=128, axis=1)  # Pad/crop to 128x128
    mel_db_resized = mel_db_resized / np.max(np.abs(mel_db_resized))  # Normalize
    mel_db_resized = np.expand_dims(mel_db_resized, axis=(0, -1))  # (1, 128, 128, 1)
    return mel_db_resized

# === Spectrogram Plot for UI ===
def plot_spectrogram(audio, sr):
    fig, ax = plt.subplots(figsize=(6, 3))
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, sr=sr, ax=ax, x_axis='time', y_axis='mel')
    ax.set(title='Mel-frequency spectrogram')
    fig.tight_layout()
    return fig

# === Streamlit App ===
def main():
    st.set_page_config(page_title="TAILSENSE: Dual CNN-based Emotion Detection")
    st.title("🐾 TAILSENSE: Canine Emotion Detection via Image and Audio")

    col1, col2 = st.columns(2)
    saved_image_path = None
    recorded_audio = None

    # === Image Capture ===
    with col1:
        st.subheader("📷 Capture Canine Image")
        image_data = st.camera_input("Take Picture")

        if image_data:
            try:
                img = Image.open(image_data)
                os.makedirs("saved_images", exist_ok=True)
                saved_image_path = f"saved_images/Canine_{uuid.uuid4().hex}.jpg"
                img.save(saved_image_path)
                st.image(saved_image_path, caption="Captured Canine Image")
                st.success("Image captured and saved.")
            except Exception as e:
                st.error(f"Error saving image: {e}")

    # === Audio Capture ===
    with col2:
        st.subheader("🎙️ Capture Audio Sample")
        if st.button("Capture Audio"):
            try:
                st.info("Recording audio for 2 seconds...")
                audio = sd.rec(int(AUDIO_DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
                sd.wait()
                recorded_audio = np.squeeze(audio)

                # Display Spectrogram
                fig = plot_spectrogram(recorded_audio, SAMPLE_RATE)
                st.pyplot(fig)
                st.success("Audio captured and spectrogram displayed.")
            except Exception as e:
                st.error(f"Audio recording failed: {e}")

    # === Image Prediction ===
    st.markdown("---")
    st.header("🧠 Predict Emotion from Image")
    if st.button("Predict Image Emotion"):
        if not saved_image_path or not os.path.exists(saved_image_path):
            st.warning("Please capture an image first.")
        else:
            try:
                inp = preprocess_image(saved_image_path)
                preds = image_model.predict(inp)[0]
                idx = int(np.argmax(preds))
                label = classes_dict[idx]
                conf = preds[idx]

                st.image(saved_image_path, caption="Prediction Image")
                st.success(f"Image Emotion: **{label}** (Confidence: {conf:.2f})")
            except Exception as e:
                st.error(f"Image prediction failed: {e}")

    # === Audio Prediction ===
    st.markdown("---")
    st.header("🧠 Predict Emotion from Audio")
    if st.button("Predict Audio Emotion"):
        if recorded_audio is None:
            st.warning("Please capture audio first.")
        else:
            try:
                spectro_input = preprocess_audio(recorded_audio, SAMPLE_RATE)
                preds = audio_model.predict(spectro_input)[0]
                idx = int(np.argmax(preds))
                label = classes_dict[idx]
                conf = preds[idx]

                st.success(f"Audio Emotion: **{label}** (Confidence: {conf:.2f})")
            except Exception as e:
                st.error(f"Audio prediction failed: {e}")

if __name__ == "__main__":
    main()