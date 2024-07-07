import cv2
import mediapipe as mp


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

def process_frame(frame1):
    # Initialize MediaPipe Hands module
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    # Initialize MediaPipe Drawing module for drawing landmarks
    mp_drawing = mp.solutions.drawing_utils
    # Process the captured frame (perform any desired operations)
    # Example: Convert the frame to grayscale
    
    t = hands.process(frame1)
    
    # gray_frame = cv2.cvtColor(frame, cv2.BORDER_REPLICATE)
    # frame2 = cv2.putText(gray_frame, f"label:", (300,300), cv2.FONT_HERSHEY_COMPLEX, 10, (255, 0, 0), 2, cv2.LINE_AA)
    
    # Check if hands are detected
    if t.multi_hand_landmarks:
        for hand_landmarks in t.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame1, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    

    return frame1

def display_frame(frame):
    # Display the processed frame
    cv2.imshow("Webcam", frame)
    cv2.waitKey(1)

def main():
    webcam = initialize_webcam()
    if webcam is None:
        return

    while True:
        frame = capture_frame(webcam)
        if frame is None:
            break

        processed_frame = process_frame(frame)
        display_frame(processed_frame)

    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()