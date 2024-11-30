import os
import numpy as np
import librosa
import librosa.display
from keras.models import load_model
import gradio as gr

# Classes dictionary
audio_classes_dict = {
    0: 'dog_emotion_bark_10_affectionate', 
    1: 'dog_emotion_bark_1_alert',
    2: 'dog_emotion_bark_2_pain', 
    3: 'dog_emotion_bark_3_anxious',
    4: 'dog_emotion_bark_4_threatened', 
    5: 'dog_emotion_bark_5_angry',
    6: 'dog_emotion_bark_6_stressed', 
    7: 'dog_emotion_bark_7_defensive',
    8: 'dog_emotion_bark_8_stress_release', 
    9: 'dog_emotion_bark_9_friendly'
}

# Prediction function for audio
def audio_prediction(audio_path, model, classes_dict):
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=22050)

    # Extract features (e.g., MFCCs)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=64)
    mfccs_processed = np.mean(mfccs.T, axis=0)  # Take the mean to reduce dimensions

    # Expand dimensions for model input
    mfccs_exp_dim = np.expand_dims(mfccs_processed, axis=0)

    # Predict using the model
    prediction_probabilities = model.predict(mfccs_exp_dim)
    pred_class_index = np.argmax(prediction_probabilities)
    predicted_class = classes_dict[pred_class_index]
    predicted_probability = prediction_probabilities[0][pred_class_index]

    return f"{predicted_class} (Confidence: {predicted_probability:.2f})"

# Gradio wrapper for audio
def predict_audio(audio):
    model = load_model(r"C:\Users\OMOLP049\Documents\Arwan Makheja\Part2(SoundDetection)\sound_direction\sound_identification_model.h5")  # Replace with your model path
    return audio_prediction(audio, model, audio_classes_dict)

# Main function for Gradio UI
def main_audio():
    # Ensure the flagging directory exists
    flagging_dir = "flagged_audio_data"
    os.makedirs(flagging_dir, exist_ok=True)

    # Create Gradio interface for audio
    io = gr.Interface(
        fn=predict_audio,
        inputs=gr.Audio(label="Upload the audio file of Dog", type="filepath"),
        outputs=gr.Textbox(label="Dog Emotion Prediction"),
        allow_flagging="manual",
        flagging_options=["Save Audio for Review"],
        flagging_dir=flagging_dir,
        title="<b>Dog Emotion Detection from Audio</b>",
        description=(
            "Effortlessly detect the emotions of dogs based on their sounds! "
           
        ),
        theme=gr.themes.Base(
            primary_hue="blue",  # Primary color of the UI
            secondary_hue="green",  # Accent color
            neutral_hue="pink",  # Background and other elements
            text_size="lg",  # Increase text size
        ),
    )
    io.launch(share=True)

if __name__ == "__main__":
    main_audio()
