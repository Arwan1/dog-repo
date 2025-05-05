import os
import uuid
import cv2
import numpy as np
#import sounddevice as sd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
import tensorflow as tf

# === CONFIG ===
AUDIO_DURATION = 3  # seconds
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
image_model_path = "/home/omotec/Desktop/arwanisef/FULL_2_E_emotion_model_best.h5"
audio_model_path = "/home/omotec/Desktop/arwanisef/EfficientNetB7_Model-FULL-2_emotion_model.h5"
image_model = tf.keras.models.load_model(image_model_path)
audio_model = tf.keras.models.load_model(audio_model_path)

# === Preprocessing ===
def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB").resize((180, 180))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0)

def preprocess_audio(audio, sr):
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db_resized = librosa.util.fix_length(mel_db, size=128, axis=1)
    mel_db_resized = mel_db_resized / np.max(np.abs(mel_db_resized))
    mel_db_resized = np.expand_dims(mel_db_resized, axis=(0, -1))
    return mel_db_resized

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

    saved_image_path = None
    audio_fragment = None

    # === Video and Frame Capture ===
    st.subheader("📹 Video Feed (USB Camera)")
    run_camera = st.checkbox("Start Camera")
    captured_frame = st.button("Capture Frame")

    frame_placeholder = st.empty()
    captured_frame_path = "saved_images/captured_frame.jpg"

    if run_camera:
        os.makedirs("saved_images", exist_ok=True)
        cap = cv2.VideoCapture(0)

        while run_camera:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access USB camera.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", caption="Live Video Feed")

            if captured_frame:
                cv2.imwrite(captured_frame_path, frame)
                st.image(captured_frame_path, caption="Captured Frame")
                st.success("Frame captured and saved.")
                break

        cap.release()

    # === Image Prediction ===
    st.markdown("---")
    st.header("🧠 Predict Emotion from Captured Frame")
    if st.button("Predict Image Emotion"):
        if not os.path.exists(captured_frame_path):
            st.warning("Please capture a frame first.")
        else:
            try:
                inp = preprocess_image(captured_frame_path)
                preds = image_model.predict(inp)[0]
                idx = int(np.argmax(preds))
                label = classes_dict[idx]
                conf = preds[idx]
                st.image(captured_frame_path, caption="Prediction Frame")
                st.success(f"Image Emotion: **{label}** (Confidence: {conf:.2f})")
            except Exception as e:
                st.error(f"Image prediction failed: {e}")

    # === Audio Capture & Spectrogram ===
    st.markdown("---")
    st.subheader("🎙️ Audio Capture (USB Microphone)")
    if st.button("Capture 3s Audio Fragment"):
        pass
    """    
        try:
            st.info("Recording audio for 3 seconds...")
            audio = sd.rec(int(AUDIO_DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
            sd.wait()
            audio_fragment = np.squeeze(audio)
            fig = plot_spectrogram(audio_fragment, SAMPLE_RATE)
            st.pyplot(fig)
            st.success("Audio recorded and spectrogram displayed.")
        except Exception as e:
            st.error(f"Audio recording failed: {e}")
    """
    # === Audio Prediction ===
    st.header("🧠 Predict Emotion from Audio Fragment")
    if st.button("Predict Audio Emotion"):
        pass
        """
        if audio_fragment is None:
            st.warning("Please capture audio first.")
        else:
            try:
                spectro_input = preprocess_audio(audio_fragment, SAMPLE_RATE)
                preds = audio_model.predict(spectro_input)[0]
                idx = int(np.argmax(preds))
                label = classes_dict[idx]
                conf = preds[idx]
                st.success(f"Audio Emotion: **{label}** (Confidence: {conf:.2f})")
            except Exception as e:
                st.error(f"Audio prediction failed: {e}")
        """
if __name__ == "__main__":
    main()
