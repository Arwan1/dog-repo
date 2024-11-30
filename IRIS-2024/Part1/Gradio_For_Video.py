# import os
# import numpy as np
# import cv2
# from keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.applications.vgg19 import preprocess_input
# import gradio as gr

# # Classes dictionary
# classes_dict = {
#     0: "dog emotion video 10 affectionate",
#     1: "dog emotion video 1 alert",
#     2: "dog emotion video 2 pain",
#     3: "dog emotion video 3 anxious",
#     4: "dog emotion video 4 threatened",
#     5: "dog emotion video 5 angry",
#     6: "dog emotion video 6 stressed",
#     7: "dog emotion video 7 defensive",
#     8: "dog emotion video 8 stress release",
#     9: "dog emotion video 9 friendly",
# }

# # Cleaned class names (no underscores)
# classes_dict_cleaned = {key: value.replace('_', ' ') for key, value in classes_dict.items()}

# # Video prediction function
# def predict_video(video_path, model, classes_dict):
#     # Load the video
#     cap = cv2.VideoCapture(video_path)
#     frame_predictions = []

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Resize the frame to 224x224 (expected input size of the model)
#         frame_resized = cv2.resize(frame, (224, 224))
#         frame_array = img_to_array(frame_resized)
#         processed_frame = preprocess_input(frame_array)

#         # Expand dimensions to fit model input
#         frame_exp_dim = np.expand_dims(processed_frame, axis=0)

#         # Predict the emotion of the frame
#         prediction_probabilities = model.predict(frame_exp_dim)
#         pred_class_index = np.argmax(prediction_probabilities)
#         predicted_class = classes_dict[pred_class_index]
#         frame_predictions.append(predicted_class)

#     cap.release()

#     # Determine the most common prediction in the video
#     if frame_predictions:
#         most_common_prediction = max(set(frame_predictions), key=frame_predictions.count)
#     else:
#         most_common_prediction = "No frames detected"

#     return video_path, most_common_prediction


# # Gradio wrapper
# def predict(video):
#     model = load_model(
#         r"C:\Users\OMOLP049\Documents\Arwan Makheja\Part1(DogEmotionDetection)\Dog_Emotion_(ALEX)CNN.h5"
#     )
#     return predict_video(video.name, model, classes_dict_cleaned)


# # Main function for Gradio UI
# def main():
#     # Ensure the flagging directory exists
#     flagging_dir = "flagged_data"
#     os.makedirs(flagging_dir, exist_ok=True)

#     # Create Gradio interface
#     io = gr.Interface(
#         fn=predict,
#         inputs=gr.File(label="Upload the video of Dog", file_types=["video"]),
#         outputs=[
#             gr.Video(label="Uploaded Video"),
#             gr.Textbox(label="Dog Emotion Prediction"),
#         ],
#         allow_flagging="manual",
#         flagging_options=["Save"],
#         flagging_dir=flagging_dir,
#         title="Dog Emotion Detection",
#         description="Effortlessly Detecting the emotion of Dogs from videos.",
#         theme=gr.themes.Soft(),
#     )
#     io.launch(share=True)


# if __name__ == "__main__":
#     main()



import os
import numpy as np
import cv2
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg19 import preprocess_input
import gradio as gr

# Classes dictionary
classes_dict = {
    0: "dog emotion video 10 affectionate",
    1: "dog emotion video 1 alert",
    2: "dog emotion video 2 pain",
    3: "dog emotion video 3 anxious",
    4: "dog emotion video 4 threatened",
    5: "dog emotion video 5 angry",
    6: "dog emotion video 6 stressed",
    7: "dog emotion video 7 defensive",
    8: "dog emotion video 8 stress release",
    9: "dog emotion video 9 friendly",
}

# Cleaned class names (no underscores)
classes_dict_cleaned = {key: value.replace('_', ' ') for key, value in classes_dict.items()}

# Video prediction function
def predict_video(video_path, model, classes_dict):
    # Load the video
    cap = cv2.VideoCapture(video_path)
    frame_predictions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame to 224x224 (expected input size of the model)
        frame_resized = cv2.resize(frame, (224, 224))
        frame_array = img_to_array(frame_resized)
        processed_frame = preprocess_input(frame_array)

        # Expand dimensions to fit model input
        frame_exp_dim = np.expand_dims(processed_frame, axis=0)

        # Predict the emotion of the frame
        prediction_probabilities = model.predict(frame_exp_dim)
        pred_class_index = np.argmax(prediction_probabilities)
        predicted_class = classes_dict[pred_class_index]
        frame_predictions.append(predicted_class)

    cap.release()

    # Determine the most common prediction in the video
    if frame_predictions:
        most_common_prediction = max(set(frame_predictions), key=frame_predictions.count)
    else:
        most_common_prediction = "No frames detected"

    return video_path, most_common_prediction


# Gradio wrapper
def predict(video):
    model = load_model(
        r"C:\Users\OMOLP049\Documents\Arwan Makheja\Part1(DogEmotionDetection)\Dog_Emotion_(ALEX)CNN.h5"
    )
    return predict_video(video.name, model, classes_dict_cleaned)


# Main function for Gradio UI
def main():
    # Ensure the flagging directory exists
    flagging_dir = "flagged_data"
    os.makedirs(flagging_dir, exist_ok=True)

    # Create Gradio interface
    io = gr.Interface(
        fn=predict,
        inputs=gr.File(label="Upload the video of Dog", file_types=["video"]),
        outputs=[
            gr.Video(label="Uploaded Video"),
            gr.Textbox(label="Dog Emotion Prediction"),
        ],
        allow_flagging="manual",
        flagging_options=["Save"],
        flagging_dir=flagging_dir,
        title="Dog Emotion Detection",
        description="Effortlessly Detecting the emotion of Dogs from videos.",
        theme=gr.themes.Base(
            primary_hue="blue",     # Change primary color
            secondary_hue="cyan",   # Change accent color
            neutral_hue="teal",     # Change background and neutral elements
            text_size="lg",         # Set larger text size
        ),
    )
    io.launch(share=True)


if __name__ == "__main__":
    main()
