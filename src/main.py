import cv2
import mediapipe as mp
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import os
import random
import string
import tensorflow as tf

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)
model = load_model('D:\\Repositorio\\RPS-project\\src\\models\\aug.h5')

# Nome das classes
class_names = ['rock', 'paper', 'scissor']  # Ajuste conforme necessário

# Main script to access webcam

def initialize_webcam():
    # Initialize the webcam
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        print("Failed to open webcam")
        return None
    return webcam

def capture_frame(webcam):
    # Capture a frame from the webcam
    ret, frame = webcam.read()
    if not ret:
        print("Failed to capture frame")
        return None
    return frame

def generate_random_string(length=33):
    # Combine letters and digits
    characters = string.ascii_letters + string.digits
    # Generate a random string of specified length
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string

# Example usage
random_string = generate_random_string()
print(random_string)

# Função para pré-processar a imagem da mão
def preprocess_image(image):
    img = Image.fromarray(image)
    img = img.resize((300, 200))  # Redimensiona a imagem para 300x200 pixels
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Adiciona uma dimensão para o batch
    return img_array

def process_frame(frame1):
    # Process the captured frame (perform any desired operations)
    # Example: Convert the frame to grayscale
    # gray_frame = cv2.cvtColor(frame, cv2.BORDER_REPLICATE)
    # frame2 = cv2.putText(gray_frame, f"label:", (300,300), cv2.FONT_HERSHEY_COMPLEX, 10, (255, 0, 0), 2, cv2.LINE_AA)

    # Initialize MediaPipe Drawing module for drawing landmarks
    image_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = hands.process(image_rgb)
    image_rgb.flags.writeable = True
    #image_rgb = cv2.rotate(image_rgb, cv2.ROTATE_90_CLOCKWISE)    
    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    #hand_landmarks = results.multi_hand_landmarks

    

    # Initialize MediaPipe Hands module
    #hands = mp_hands.Hands()

    
    
    # Check if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(image_rgb, hand_landmark, mp_hands.HAND_CONNECTIONS)
    
            # Extrair a área delimitadora da mão
            h, w, c = image_rgb.shape
            landmarks = np.array([[landmark.x * w, landmark.y * h] for landmark in hand_landmark.landmark], dtype=np.float32)  # Ensure the array is of type float32
            if landmarks.size > 0:  # Check if landmarks array is not empty
                bbox = cv2.boundingRect(landmarks)
                x, y, w, h = bbox
                hand_image = image_rgb[y:y + h, x:x + w].copy()
                # Pre-process the hand image
                preprocessed_image = preprocess_image(hand_image)

                with tf.device('/GPU:0'):
                    # Make the prediction
                    predictions = model.predict(preprocessed_image)
                predicted_label = np.argmax(predictions)

                # if predicted_label == 0:
                #     out_path = "D:\\Repositorio\\RPS-project\\src\\pedra"
                #     frame_name = generate_random_string()+'.jpg'
                #     cv2.imwrite(os.path.join(out_path, frame_name), hand_image)
                # elif predicted_label == 1:
                #     out_path = "D:\\Repositorio\\RPS-project\\src\\papel"
                #     frame_name = generate_random_string()+'.jpg'
                #     cv2.imwrite(os.path.join(out_path, frame_name), hand_image)
                # elif predicted_label == 2:
                #     out_path = "D:\\Repositorio\\RPS-project\\src\\tesoura"
                #     frame_name = generate_random_string()+'.jpg'
                #     cv2.imwrite(os.path.join(out_path, frame_name), hand_image)
                
                # Display the predicted class on the image
                cv2.putText(frame1, f'Prediction: {class_names[predicted_label]}',
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    return frame1

def display_frame(frame):
    # Display the processed frame
    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        return False
    return True

def main():
    webcam = initialize_webcam()
    if webcam is None:
        return

    while webcam.isOpened():
        success, image = webcam.read()
        
        if not success or image is None:
            print("Ignoring empty camera frame.")
            continue

        frame = capture_frame(webcam)
        if frame is None:
            break

        processed_frame = process_frame(frame)
        if not display_frame(processed_frame):
            break  # Exit the loop if 'q' is pressed

    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()