from django.shortcuts import render,redirect
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import time
import os
from . models import CapturedImage
from django.conf import settings
model_path = os.path.join(settings.BASE_DIR,'best_model.h5') 
model = load_model(model_path)
# Load Haar Cascade for face detection
face_cascade_path = os.path.join(settings.BASE_DIR, 'haarcascade_frontalface_default.xml')
face_haar_cascade = cv2.CascadeClassifier(face_cascade_path)


# Create your views here.
def emotion_analysis(request):   
    cap = cv2.VideoCapture(0)
    # Parameters for capturing images
    capture_interval = 0.1
    capture_count = 0
    max_captures = 30
    min_confidence = 0.7  # Minimum confidence threshold to capture an image
    confidence_scores = []
    # Calculate the end time for capturing frames
    end_time = time.time() + max_captures
    while time.time() < end_time:
        ret, test_img = cap.read()
        if not ret:
           continue
        gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
            # Perform face detection
        faces_detected = face_haar_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces_detected:
            roi_gray = gray_img[y:y + h, x:x + w]

            # Perform emotion analysis on the detected face ROI
            roi_gray = cv2.resize(roi_gray, (224, 224))  # Resize for the emotion model input size
            roi_gray = roi_gray.astype("float") / 255.0  # Normalize pixel values
            roi_gray = img_to_array(roi_gray)
            roi_gray = np.expand_dims(roi_gray, axis=0)
            # Perform emotion prediction using your loaded deep learning model
            emotions = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')
            predicted_emotion = emotions[np.argmax(model.predict(roi_gray))]

            # Display the emotion and confidence score on the frame
            text = f"Emotion: {predicted_emotion}"
            cv2.putText(test_img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            confidence_score = np.max(model.predict(roi_gray))  # Confidence score for the predicted emotion
           # Check if confidence score is above the threshold and capture the image
            if confidence_score >= min_confidence and capture_count < max_captures:
                capture_count += 1
                timestamp = int(time.time())
                image_filename = f"captured_{capture_count}_{timestamp}.jpg"
                image_path =os.path.join(settings.MEDIA_ROOT,image_filename)
                cv2.imwrite(image_path, test_img)
                confidence_scores.append(confidence_score)
                captured_image = CapturedImage(
                        image=image_filename,
                        emotion=predicted_emotion,
                        confidence=confidence_score
                    )
                captured_image.save()
            # Display the processed frame
            resized_img = cv2.resize(test_img, (1000, 700))
            cv2.imshow('Facial emotion analysis', resized_img)

            # Check for user input to exit the loop (e.g., press 'q' to quit)
            if cv2.waitKey(10) == ord('q') or capture_count >= max_captures:
             break
    cap.release()
    cv2.destroyAllWindows()
    average_confidence_percentage=0
    # Calculate and print the average confidence level as a percentage
    if confidence_scores:
        average_confidence = sum(confidence_scores) / len(confidence_scores)
        average_confidence_percentage =int(average_confidence * 100)
        print(f"Average Confidence Level: {average_confidence_percentage:.2f}%")
    else:
        print("No images captured.")
    
    # Redirect to a page displaying captured images and emotions
    captured_images = CapturedImage.objects.all()  # Assuming you have a CapturedImage model
    
    return render(request,'emotion_analysis.html',{'captured_images': captured_images,'average_confidence_percentage': average_confidence_percentage})





  