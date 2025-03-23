import os
import traceback
import cv2
import numpy as np
from mtcnn import MTCNN
from deepface import DeepFace


def validate_dimensions(image, required_width=413, required_height=531):
    try:
        height, width, _ = image.shape
        is_valid = width == required_width and height == required_height
        message = f"Dimensions: {width}x{height} (should be {required_width}x{required_height})"
        return bool(is_valid), message
    except Exception as e:
        print(f"Error in validate_dimensions: {str(e)}")
        return False, f"Failed to check dimensions: {str(e)}"


def validate_background(image, threshold=180):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        white_pixels = np.sum(gray > threshold)
        total_pixels = gray.size
        white_percentage = (white_pixels / total_pixels) * 100
        is_valid = white_percentage > 80
        message = f"Background is {white_percentage:.1f}% white (should be >80%)"
        return bool(is_valid), message
    except Exception as e:
        print(f"Error in validate_background: {str(e)}")
        return False, f"Failed to check background: {str(e)}"


def validate_face_positioning(image):
    try:
        detector = MTCNN()
        results = detector.detect_faces(image)
        if not results:
            return False, "No face detected"
        return True, "Face positioning appears valid"
    except Exception as e:
        print(f"Error in validate_face_positioning: {str(e)}")
        return False, f"Failed to check face position: {str(e)}"


def validate_emotion(image_path):
    try:
        print(f"Starting emotion validation on: {image_path}")
        # Check if file exists
        if not os.path.exists(image_path):
            return False, "Image file not found"

        # Analyze emotion
        attributes = DeepFace.analyze(
            img_path=image_path, actions=["emotion"], enforce_detection=False
        )

        print(f"DeepFace analysis completed")

        # Handle multiple faces
        if isinstance(attributes, list):
            if not attributes:
                return False, "No face detected for emotion analysis"
            # Use the first face in the list
            attributes = attributes[0]

        emotion_scores = attributes["emotion"]
        print(f"Emotion scores: {emotion_scores}")

        # Check for neutral expression
        if emotion_scores["neutral"] < 0.5:
            dominant_emotion = max(emotion_scores, key=emotion_scores.get)
            return False, f"Expression not neutral (detected: {dominant_emotion})"

        return True, "Expression is neutral"
    except Exception as e:
        print(f"Exception in validate_emotion: {str(e)}")
        print(traceback.format_exc())
        return False, f"Emotion validation failed: {str(e)}"


def validate_head_covering(image_path):
    try:
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            return False, "Invalid image file"

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        if len(faces) == 0:
            return False, "No face detected"

        (x, y, w, h) = faces[0]

        aspect_ratio = w / h
        if aspect_ratio < 0.7 or aspect_ratio > 1.3:
            return False, "Head covering detected (abnormal aspect ratio)"

        # Analyze the upper region of the face for head coverings
        upper_face = gray[y : y + h // 4, x : x + w]
        upper_face_mean = upper_face.mean()

        if upper_face_mean < 100:
            return False, "Head covering detected (dark upper region)"

        return True, "No head covering detected"
    except Exception as e:
        return False, f"Head covering validation failed: {str(e)}"


# testing the file
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        test_image_path = sys.argv[1]
    else:
        test_image_path = "C:\\Users\\anves\\Downloads\\test2.jpg"
    print(f"Testing validation functions on: {test_image_path}")

    try:
        image = cv2.imread(test_image_path)
        if image is None:
            print(f"Error: Could not load image from {test_image_path}")
            sys.exit(1)

        print(f"Image loaded successfully. Shape: {image.shape}")

        print("Dimensions:", validate_dimensions(image))
        print("Background:", validate_background(image))

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print("Face Positioning:", validate_face_positioning(rgb_image))

        print("Emotion:", validate_emotion(test_image_path))
        print("Head Covering:", validate_head_covering(test_image_path))

    except Exception as e:
        print(f"Error during testing: {str(e)}")
        traceback.print_exc()
