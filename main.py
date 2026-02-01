import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import os
import urllib.request
import math

def calculate_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def draw_landmarks_and_connections(image, hand_landmarks):
    # Get image dimensions
    h, w, _ = image.shape
    
    # Convert landmarks to pixel coordinates
    points = []
    for lm in hand_landmarks:
        points.append((int(lm.x * w), int(lm.y * h)))
        
    # Connections (Indices based on Hand Landmark Model)
    # Thumb: 0-1-2-3-4
    # Index: 0-5-6-7-8
    # Middle: 0-9-10-11-12
    # Ring: 0-13-14-15-16
    # Pinky: 0-17-18-19-20
    
    # Colors
    connection_color = (0, 255, 0)
    landmark_color = (0, 0, 255)
    
    connections = [
        (0,1), (1,2), (2,3), (3,4), # Thumb
        (0,5), (5,6), (6,7), (7,8), # Index
        (5,9), (9,10), (10,11), (11,12), # Middle
        (9,13), (13,14), (14,15), (15,16), # Ring
        (13,17), (17,18), (18,19), (19,20), # Pinky
        (0,17) # Wrist to Pinky base
    ]
    
    # Draw connections
    for start_idx, end_idx in connections:
        cv2.line(image, points[start_idx], points[end_idx], connection_color, 2)
        
    # Draw landmarks
    for point in points:
        cv2.circle(image, point, 4, landmark_color, -1)

def main():
    model_path = 'hand_landmarker.task'
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

    if not os.path.exists(model_path):
        print(f"Downloading {model_path}...")
        try:
            urllib.request.urlretrieve(url, model_path)
            print("Download complete.")
        except Exception as e:
            print(f"Failed to download model: {e}")
            return

    # Initialize MediaPipe HandLandmarker
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir la caméra. Vérifiez qu'elle est connectée et n'est pas utilisée par un autre programme.")
        return

    # Start time for timestamping
    start_time = time.time()
    
    empty_frame_count = 0

    with HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                empty_frame_count += 1
                if empty_frame_count > 10:
                    print("Erreur: La caméra ne renvoie pas d'images. Arrêt du programme.")
                    break
                continue
            
            empty_frame_count = 0

            # Flip image for selfie view
            image = cv2.flip(image, 1)
            
            # Convert to RGB (MediaPipe needs RGB)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            
            # Calculate timestamp in ms
            timestamp_ms = int((time.time() - start_time) * 1000)
            
            # Detect
            detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            # Process results
            gesture = "Inconnu"
            
            # Helper to draw text
            cv2.rectangle(image, (0,0), (640, 40), (255, 255, 255), -1)
            
            if detection_result.hand_landmarks:
                for hand_landmarks in detection_result.hand_landmarks:
                    # Draw landmarks manually since 'drawing_utils' might be missing
                    draw_landmarks_and_connections(image, hand_landmarks)
                    
                    # Logic
                    lm = hand_landmarks
                    
                    # Tips: 4, 8, 12, 16, 20
                    # PIPs: 2, 6, 10, 14, 18
                    
                    tips = [4, 8, 12, 16, 20]
                    pips = [2, 6, 10, 14, 18]
                    
                    fingers_open = []
                    
                    # Fingers (Index to Pinky)
                    # Note: y increases downwards. Open finger -> tip.y < pip.y (if hand is up)
                    for i in range(1, 5):
                        if lm[tips[i]].y < lm[pips[i]].y:
                            fingers_open.append(True)
                        else:
                            fingers_open.append(False)
                            
                    # Thumb logic
                    # Check distance from pinky base (17)
                    dist_thumb_tip_pinky = calculate_distance(lm[4], lm[17])
                    dist_thumb_ip_pinky = calculate_distance(lm[3], lm[17])
                    
                    thumb_open = dist_thumb_tip_pinky > dist_thumb_ip_pinky
                    
                    count_open_non_thumb = sum(fingers_open)
                    
                    if count_open_non_thumb == 0 and not thumb_open:
                        gesture = "Poing (FIST)"
                    elif count_open_non_thumb == 0:
                        gesture = "Poing (FIST)" # Thumb out sometimes happens
                    elif count_open_non_thumb == 4:
                        gesture = "Main Ouverte (OPEN)"
                    elif fingers_open[0] and not any(fingers_open[1:]): # Index only
                        gesture = "Doigt Pointe (POINT)"
                    else:
                        gesture = "..."
                        
            else:
                 gesture = "Pas de main"

            cv2.putText(image, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 0, 0), 2, cv2.LINE_AA)
            
            cv2.imshow('Gesture Recognition (Tasks API)', image)
            
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
