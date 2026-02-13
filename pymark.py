import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime

class ImageAnnotator:
    def __init__(self, image_path):
        # Load the image
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        height, width = self.original_image.shape[:2]
        max_dimension = 1800
        if max(height, width) < max_dimension:
            scale = max_dimension / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            self.original_image = cv2.resize(self.original_image, (new_width, new_height))
        
        # Create a copy for annotations
        self.annotated_image = self.original_image.copy()
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Drawing settings
        self.marker_mode = True
        self.save_mode = False
        self.color_change_mode = False
        self.current_color = (0, 0, 255)  # Red
        self.brush_size = 5
        self.prev_point = None
        
        # Cursor position
        self.cursor_pos = None
        
        # Colors
        self.colors = {
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
            'yellow': (0, 255, 255),
            'white': (255, 255, 255),
            'black': (0, 0, 0)
        }
        self.color_names = list(self.colors.keys())
        self.color_index = 0
        
        # Marker counter
        self.marker_count = 0
        
    def is_finger_up(self, landmarks, finger_tip_id, finger_pip_id):
        """Check if finger is extended"""
        return landmarks[finger_tip_id].y < landmarks[finger_pip_id].y
    
    def get_landmark_distance(self, landmarks, finger1, finger2):
        """Calculate distance between fingers"""
        x1, y1 = landmarks[finger1].x, landmarks[finger1].y
        x2, y2 = landmarks[finger2].x, landmarks[finger2].y
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2) * 1000
    
    def place_marker(self, x, y):
        self.marker_count += 1
        # Draw crosshair marker
        cv2.circle(self.annotated_image, (x, y), 15, self.current_color, 2)
        cv2.circle(self.annotated_image, (x, y), 3, self.current_color, -1)
        cv2.line(self.annotated_image, (x-12, y), (x+12, y), self.current_color, 2)
        cv2.line(self.annotated_image, (x, y-12), (x, y+12), self.current_color, 2)
        # Add number
        cv2.putText(self.annotated_image, str(self.marker_count),
                    (x+20, y-20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, self.current_color, 2)
        print(f"✓ Marker {self.marker_count} placed")

    def change_color(self):
        self.color_index = (self.color_index + 1) % len(self.color_names)
        self.current_color = self.colors[self.color_names[self.color_index]]
        print(f"🎨 Color: {self.color_names[self.color_index]}")
    
    def save_image(self):
        filename = f"annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(filename, self.annotated_image)
        print(f"💾 Saved to {filename}")
    
    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\n" + "="*60)
        print("IMAGE ANNOTATION - Hand Gesture Control")
        print("="*60)
        print("\n🎨 DRAWING MODES:")
        print("  • Marker Mode: Drop numbered markers")
        print("  • Draw Mode: Free-hand drawing")
        print("\n✋ GESTURES:")
        print("  • Index finger = Move cursor")
        print("  • Index + Pinch thumb = Draw/Place marker")
        print("\n⌨️  KEYBOARD CONTROLS:")
        print("  • D = Toggle Draw/Marker mode")
        print("  • C = Change color")
        print("  • + = Increase brush size")
        print("  • - = Decrease brush size")
        print("  • R = Reset (clear all annotations)")
        print("  • S = Save annotated image")
        print("  • Q = Quit")
        print("="*60 + "\n")
        
        # Create large window for image
        cv2.namedWindow('Annotated Image', cv2.WINDOW_NORMAL)
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue
            
            frame = cv2.flip(frame, 1)
            frame_height, frame_width, _ = frame.shape
            
            # Get image dimensions for scaling
            img_height, img_width = self.annotated_image.shape[:2]
            
            # Process hand tracking
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Get index finger position
                    index_tip = hand_landmarks.landmark[8]
                    index_up = self.is_finger_up(hand_landmarks.landmark, 8, 6)
                    middle_up = self.is_finger_up(hand_landmarks.landmark, 12, 10)
                    ring_up = self.is_finger_up(hand_landmarks.landmark, 16, 14)
                    pinky_up = self.is_finger_up(hand_landmarks.landmark, 20, 18)
                    
                    # Map camera coordinates to image coordinates
                    img_x = int(index_tip.x * img_width)
                    img_y = int(index_tip.y * img_height)
                    
                    # Clamp to image bounds
                    img_x = max(0, min(img_x, img_width - 1))
                    img_y = max(0, min(img_y, img_height - 1))
                    
                    # Store cursor position
                    self.cursor_pos = (img_x, img_y)
                    
                    # Check if pinching
                    thumb_tip = 4
                    index_base = 5
                    pinch_dist = self.get_landmark_distance(hand_landmarks.landmark, thumb_tip, index_base)
                    is_pinching = pinch_dist < 20
                    
                    if is_pinching:
                            # Place marker
                            if index_up and not middle_up and not ring_up and not pinky_up:
                                if not self.drawing_mode:
                                    self.place_marker(img_x, img_y)
                                    self.drawing_mode = True
                            if index_up and middle_up and not ring_up and not pinky_up:
                                if not self.color_change_mode:
                                    self.change_color()
                                    self.color_change_mode = True
                            if index_up and middle_up and ring_up and pinky_up:
                                if not self.save_mode:
                                    self.save_image()
                                    self.save_mode = True    
                    else:
                        self.prev_point = None
                        self.drawing_mode = False
                        self.color_change_mode = False
                        self.save_mode = False
            
            # Show annotated image with cursor
            display_image = self.annotated_image.copy()
            
            # Draw cursor on the image
            if self.cursor_pos:
                x, y = self.cursor_pos
                # Draw cursor circle
                cv2.circle(display_image, (x, y), 20, self.current_color, 2)
                cv2.circle(display_image, (x, y), 3, self.current_color, -1)
                # Draw cursor crosshair
                cv2.line(display_image, (x-15, y), (x+15, y), self.current_color, 1)
                cv2.line(display_image, (x, y-15), (x, y+15), self.current_color, 1)
            
            # Add UI overlay to annotated image
            mode_text = "MARKER MODE"
            cv2.putText(display_image, mode_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.current_color, 2)
            cv2.putText(display_image, f"Color: {self.color_names[self.color_index]}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.current_color, 2)
            if not self.marker_mode:
                cv2.putText(display_image, f"Brush: {self.brush_size}px", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow('Annotated Image', display_image)
            
            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.color_index = (self.color_index + 1) % len(self.color_names)
                self.current_color = self.colors[self.color_names[self.color_index]]
                print(f"🎨 Color: {self.color_names[self.color_index]}")
            elif key == ord('+') or key == ord('='):
                self.brush_size = min(20, self.brush_size + 2)
                print(f"🖌️  Brush size: {self.brush_size}px")
            elif key == ord('-') or key == ord('_'):
                self.brush_size = max(1, self.brush_size - 2)
                print(f"🖌️  Brush size: {self.brush_size}px")
            elif key == ord('r'):
                self.annotated_image = self.original_image.copy()
                self.marker_count = 0
                print("🔄 Reset annotations")
            elif key == ord('s'):
                filename = f"annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(filename, self.annotated_image)
                print(f"💾 Saved to {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python simple_image_annotator.py <image_path>")
        print("Example: python simple_image_annotator.py microscope_slide.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    annotator = ImageAnnotator(image_path)
    annotator.run()