import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# Load the TensorFlow model
model = tf.keras.models.load_model("buoi_model.h5")

# Load labels
with open("buoi_labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Function to detect the color of the fruit


def detect_color(roi):
    # Convert the ROI to HSV color space
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Calculate the average Hue value
    mean_hue = np.mean(hsv_roi[:, :, 0])  # Hue channel

    # Determine the color based on the Hue value
    if 0 <= mean_hue < 15 or 160 <= mean_hue <= 180:
        return "Red"
    elif 15 <= mean_hue < 35:
        return "Orange/Yellow"
    elif 35 <= mean_hue < 85:
        return "Green"
    elif 85 <= mean_hue < 130:
        return "Blue"
    elif 130 <= mean_hue < 160:
        return "Purple"
    else:
        return "Unknown"

# Video file path


video_path = "Citrus Fruits On The Table Stock Video.mp4"
cap = cv2.VideoCapture(video_path)

# Frame processing loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize and preprocess frame for the model
    size = (224, 224)  # Input size for Teachable Machine models
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    input_data = np.expand_dims(np.asarray(image, dtype=np.float32) / 255.0, axis=0)

    # Perform inference
    predictions = model.predict(input_data)[0]

    # Get the most confident prediction
    class_index = np.argmax(predictions)
    confidence = predictions[class_index]

    if confidence > 0.6:  # Confidence threshold
        # Get the label and confidence score
        label = class_names[class_index]

        # Convert the frame to HSV and create a mask for color detection
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_color = np.array([10, 100, 100])  # Adjust for the fruit color
        upper_color = np.array([25, 255, 255])
        mask = cv2.inRange(hsv_frame, lower_color, upper_color)

        # Find contours for detected fruit
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter small contours
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Extract ROI for color detection
                roi = frame[y:y + h, x:x + w]
                fruit_color = detect_color(roi)

                # Display label, confidence score, and color
                display_text = f"{label} ({fruit_color}): {confidence:.2f}"
                cv2.putText(
                    frame,
                    display_text,
                    (x, y - 10),  # Position text above the bounding box
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,  # Font size
                    (0, 255, 0),  # Text color (green)
                    2  # Thickness
                )

    # Display the video
    cv2.imshow("Fruit Detection", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
