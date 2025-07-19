import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  


import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)


pen_pos = [200, 200]
table_pos = [400, 300]
object_selected = None

cap = cv2.VideoCapture(0)

def is_pinch(landmarks):
    """Check if thumb tip and index tip are close (pinch gesture)."""
    thumb = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    dist = ((thumb.x - index.x) ** 2 + (thumb.y - index.y) ** 2) ** 0.5
    return dist < 0.05  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

 
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_pos = (int(index_tip.x * w), int(index_tip.y * h))

            
            cv2.circle(frame, index_pos, 10, (0, 255, 0), -1)

           
            if is_pinch(hand_landmarks.landmark):
                if object_selected is None:
                    
                    if pen_pos[0]-50 < index_pos[0] < pen_pos[0]+50 and pen_pos[1]-20 < index_pos[1] < pen_pos[1]+20:
                        object_selected = 'pen'
                    elif table_pos[0]-100 < index_pos[0] < table_pos[0]+100 and table_pos[1]-50 < index_pos[1] < table_pos[1]+50:
                        object_selected = 'table'
                
                if object_selected == 'pen':
                    pen_pos = list(index_pos)
                elif object_selected == 'table':
                    table_pos = list(index_pos)
            else:
                object_selected = None

    
    cv2.rectangle(frame, (pen_pos[0]-50, pen_pos[1]-20), (pen_pos[0]+50, pen_pos[1]+20), (255, 0, 0), -1)  # Pen
    cv2.rectangle(frame, (table_pos[0]-100, table_pos[1]-50), (table_pos[0]+100, table_pos[1]+50), (0, 0, 255), -1)  # Table

    cv2.putText(frame, "Pinch to grab objects", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Virtual Object Control", frame)

    if cv2.waitKey(1) & 0xFF == 27:  
        break

cap.release()
cv2.destroyAllWindows()
