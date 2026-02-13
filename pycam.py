import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Disable PyAutoGUI failsafe (move mouse to corner to stop)
pyautogui.FAILSAFE = True

class HandMouseController:
    def __init__(self):
        # Initialize MediaPipe Hand tracking
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Get screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Smoothing variables
        self.prev_x, self.prev_y = 0, 0
        self.smooth_factor = 0.5
        
        # Click detection
        self.click_threshold = 30  # Distance threshold for click gesture
        self.is_clicking = False
        self.is_rightclicking = False
        
    def get_finger_distance(self, landmarks, finger1_tip, finger2_tip):
        """Calculate distance between two finger tips"""
        x1, y1 = landmarks[finger1_tip].x, landmarks[finger1_tip].y
        x2, y2 = landmarks[finger2_tip].x, landmarks[finger2_tip].y
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return distance * 1000  # Scale up for easier threshold comparison
    
    def is_finger_up(self, landmarks, finger_tip, finger_dip):  
        return landmarks[finger_tip].y < landmarks[finger_dip].y

    def smooth_coordinates(self, x, y):
        """Apply smoothing to reduce jitter"""
        smooth_x = self.prev_x + (x - self.prev_x) * self.smooth_factor
        smooth_y = self.prev_y + (y - self.prev_y) * self.smooth_factor
        self.prev_x, self.prev_y = smooth_x, smooth_y
        return smooth_x, smooth_y
    
    def run(self):
        # Open webcam
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Hand Mouse Controller Started!")
        print("- Move your index finger to control the cursor")
        print("- Move Thumb towards the base of the index finger to click")
        print("- Press 'q' to quit")
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to grab frame")
                continue
            
            # Flip frame horizontally for mirror view
            frame = cv2.flip(frame, 1)
            frame_height, frame_width, _ = frame.shape
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    self.mp_draw.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    # Get index finger tip coordinates (landmark 8)
                    index_finger_tip = hand_landmarks.landmark[8]
                    x = int(index_finger_tip.x * self.screen_width)
                    y = int(index_finger_tip.y * self.screen_height)
                    
                    # Apply smoothing
                    smooth_x, smooth_y = self.smooth_coordinates(x, y)
                    
                    # Move mouse
                    pyautogui.moveTo(smooth_x, smooth_y, duration=0.1)
                    
                    index_up = self.is_finger_up(hand_landmarks.landmark, 8, 6)
                    middle_up = self.is_finger_up(hand_landmarks.landmark, 12, 10)


                    thumb_tip = 4
                    index_base = 5
                    distance = self.get_finger_distance(
                        hand_landmarks.landmark, 
                        thumb_tip, 
                        index_base
                    )
                    
                    # Click detection
                    if distance < self.click_threshold:
                        if (index_up and middle_up):
                            if not self.is_rightclicking:
                                pyautogui.rightClick()
                                self.is_rightclicking = True
                                cv2.putText(
                                    frame, 
                                    "RIGHT CLICK!", 
                                    (50, 100), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    1, 
                                    (0, 0, 255), 
                                    3
                                )
                        else:
                            if not self.is_clicking:
                                pyautogui.click()
                                self.is_clicking = True
                                cv2.putText(
                                    frame, 
                                    "LEFT CLICK!", 
                                    (50, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    1, 
                                    (0, 255, 0), 
                                    3
                                )
                    else:
                        self.is_clicking = False
                        self.is_rightclicking = False
                    
                    # Draw cursor position on frame
                    cursor_x = int(index_finger_tip.x * frame_width)
                    cursor_y = int(index_finger_tip.y * frame_height)
                    cv2.circle(frame, (cursor_x, cursor_y), 10, (0, 255, 255), -1)
            
            # Display instructions
            cv2.putText(frame, "Press 'q' to quit", (10, frame_height - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show the frame
            cv2.imshow('Hand Mouse Controller', frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()

if __name__ == "__main__":
    controller = HandMouseController()
    controller.run()