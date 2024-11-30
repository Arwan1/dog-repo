from numpy import *
import tensorflow as tf
from keras.models import load_model
import cv2

# Labels Dictionaries for CNN 

emotion_type_labels_dict = {'dog_emotion_image_10_affectionate':0,'dog_emotion_image_1_alert':1,'dog_emotion_image_2_pain':2,'dog_emotion_image_3_anxious':3,
               'dog_emotion_image_4_threatened':4, 'dog_emotion_image_5_angry':5,'dog_emotion_image_6_stressed':6,
               'dog_emotion_image_7_defensive':7, 'dog_emotion_image_8_stress_release':8, 'dog_emotion_image_9_friendly':9} 

# Scores for predicted class only
def prediction(image, loaded_model, emotion_type_labels_dict):
    # Load and preprocess the image
    image = cv2.resize(image, (224, 224))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = expand_dims(image_array, axis=0)
    image_scaled = image_array / 255.0

    # Make predictions using the loaded model
    predictions = loaded_model.predict(image_scaled)

    # Apply softmax to obtain probabilities
    probabilities = tf.nn.softmax(predictions).numpy()

    # Get the predicted class index
    predicted_class_index = argmax(predictions)

    # Display the predicted class name
    predicted_class = "Unknown"
    for class_name, class_num_label in emotion_type_labels_dict.items():
        if class_num_label == predicted_class_index:
            predicted_class = class_name

    # Get the probability and confidence score for the predicted class
    probability = probabilities[0, predicted_class_index]
    confidence_score = probability * 100

    return predicted_class, confidence_score

# Storing Paths of CNN models in Variables

cnn_model = r"C:\Users\OMOLP049\Documents\Arwan Makheja\Dog_Emotion_(ALEX)CNN.h5"

cnn_model_loaded = load_model(cnn_model)

# Function to resize frame
def resize_frame(frame, width, height):
    return cv2.resize(frame, (width, height))

# Specify the desired window size
window_width = 800
window_height = 600

# Open camera feed
cap = cv2.VideoCapture(0)

# Main loop to capture and process frames
while True:
    ret, frame = cap.read()  # Read frame from camera
    if not ret:
        break
    
    # Pass the loaded model instead of the model path
    predicted_class, confidence = prediction(frame, cnn_model_loaded, emotion_type_labels_dict)
    
    prediction_string = f"{predicted_class}, {confidence:.2f}%"
    cv2.putText(frame, prediction_string, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 0), 2)

    # Resize frame to desired size
    frame_resized = resize_frame(frame, window_width, window_height)
    
    # Display resized frame
    cv2.imshow('Frame', frame_resized)
    
    # Check for exit key
    if cv2.waitKey(500) & 0xFF == ord('q'):
        break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()











