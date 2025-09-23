import cv2
import numpy as np
import math
import time

class AugmentedCanvas:
    def __init__(self):
        self.canvas_width = 640
        self.canvas_height = 480
        self.detected_objects = []
        self.proximity_threshold = 150  # Distance in pixels for clustering
        self.base_flower_size = 20      # Base flower size
        self.max_flower_size = 60       # Maximum flower size
        self.question = ""
        self.timer_duration = 60  # Default 60 seconds
        self.start_time = None
        self.is_running = False
        self.fullscreen = False
        
    def detect_paper_objects(self, gray_frame):
        # Simple threshold to detect bright objects (white paper)
        _, thresh = cv2.threshold(gray_frame, 200, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Minimum size for paper
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # Look for square-ish objects (papers)
                if 0.5 < aspect_ratio < 2.0:  # Square-ish ratio
                    center_x = x + w // 2
                    center_y = y + h // 2
                    objects.append({
                        'type': 'paper',
                        'center': (center_x, center_y),
                        'bbox': (x, y, w, h),
                        'area': area
                    })
        
        return objects
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def count_nearby_notes(self, target_obj, all_objects):
        """Count how many notes are within proximity threshold of the target object"""
        nearby_count = 0
        target_center = target_obj['center']
        
        for obj in all_objects:
            if obj != target_obj:  # Don't count itself
                distance = self.calculate_distance(target_center, obj['center'])
                if distance <= self.proximity_threshold:
                    nearby_count += 1
        
        return nearby_count
    
    def calculate_flower_size(self, note_obj, all_objects):
        """Calculate flower size based on nearby notes clustering"""
        nearby_count = self.count_nearby_notes(note_obj, all_objects)
        
        # Scale flower size based on cluster density
        # More nearby notes = bigger flower
        size_multiplier = 1 + (nearby_count * 0.5)  # Each nearby note adds 50% to size
        new_size = min(self.base_flower_size * size_multiplier, self.max_flower_size)
        
        return int(new_size)
    
    def get_user_input(self):
        """Get question and timer duration from user via console"""
        print("\n=== Augmented Canvas Setup ===")
        
        try:
            # Get question
            question = input("Enter your question: ").strip()
            if not question:
                print("No question entered. Using default.")
                question = "What do you think about this?"
                
            # Get timer duration
            timer_input = input("Enter timer duration in seconds (default 60): ").strip()
            try:
                timer_duration = int(timer_input) if timer_input else 60
            except ValueError:
                print("Invalid timer input, using default 60 seconds")
                timer_duration = 60
        except EOFError:
            # If running in environment without interactive input, use defaults
            print("Using default values...")
            question = "What do you think about this?"
            timer_duration = 60
            
        self.question = question
        self.timer_duration = timer_duration
        print(f"Question: {self.question}")
        print(f"Timer: {self.timer_duration} seconds")
        return True
    
    def start_timer(self):
        """Start the timer"""
        self.start_time = time.time()
        self.is_running = True
    
    def get_remaining_time(self):
        """Get remaining time in seconds"""
        if not self.is_running or not self.start_time:
            return self.timer_duration
        
        elapsed = time.time() - self.start_time
        remaining = max(0, self.timer_duration - elapsed)
        return remaining
    
    def is_timer_expired(self):
        """Check if timer has expired"""
        return self.get_remaining_time() <= 0
    
    def draw_flower(self, canvas, center, size=20):
        x, y = center
        
        # Define colors (BGR format for OpenCV)
        petal_color = (147, 20, 255)    # Purple petals
        center_color = (0, 255, 255)    # Yellow center  
        stem_color = (0, 255, 0)        # Green stem
        
        # Flower petals (circles around center)
        petal_positions = [
            (x, y-size), (x+size//2, y-size//2), (x+size, y),
            (x+size//2, y+size//2), (x, y+size), (x-size//2, y+size//2),
            (x-size, y), (x-size//2, y-size//2)
        ]
        
        for petal_x, petal_y in petal_positions:
            cv2.circle(canvas, (petal_x, petal_y), size//3, petal_color, -1)
        
        # Flower center
        cv2.circle(canvas, (x, y), size//4, center_color, -1)
        
        # Stem
        cv2.line(canvas, (x, y+size//4), (x, y+size*2), stem_color, 3)
    
    def create_canvas(self, detected_objects, frame_shape):
        canvas = np.zeros((frame_shape[0], frame_shape[1], 3), dtype=np.uint8)
        
        # Display question at top center if available
        if self.question:
            text_size = cv2.getTextSize(self.question, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = (frame_shape[1] - text_size[0]) // 2
            cv2.putText(canvas, self.question, (text_x, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Display timer in top right corner if running
        if self.is_running:
            remaining = self.get_remaining_time()
            timer_text = f"Time: {int(remaining)}s"
            cv2.putText(canvas, timer_text, (frame_shape[1] - 150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Only show flowers if timer is running
        if self.is_running:
            # First, draw connection lines between clustered notes
            for i, obj1 in enumerate(detected_objects):
                if obj1['type'] == 'paper':
                    for obj2 in detected_objects[i+1:]:
                        if obj2['type'] == 'paper':
                            distance = self.calculate_distance(obj1['center'], obj2['center'])
                            if distance <= self.proximity_threshold:
                                # Draw a subtle connection line
                                cv2.line(canvas, obj1['center'], obj2['center'], 
                                       (100, 100, 100), 1)  # Gray connection line
            
            # Then draw flowers with dynamic sizing
            for obj in detected_objects:
                if obj['type'] == 'paper':
                    # Map camera position to canvas position
                    canvas_x = obj['center'][0]
                    canvas_y = obj['center'][1]
                    
                    # Calculate dynamic flower size based on nearby notes
                    flower_size = self.calculate_flower_size(obj, detected_objects)
                    
                    # Draw flower at object position with dynamic size
                    self.draw_flower(canvas, (canvas_x, canvas_y), flower_size)
        
        # Show start button if not running
        if not self.is_running:
            button_text = "Press SPACE to Start"
            text_size = cv2.getTextSize(button_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            button_x = (frame_shape[1] - text_size[0]) // 2
            button_y = frame_shape[0] // 2 + 50
            
            # Draw button background
            cv2.rectangle(canvas, (button_x - 20, button_y - 30), 
                         (button_x + text_size[0] + 20, button_y + 10), (0, 255, 0), -1)
            cv2.putText(canvas, button_text, (button_x, button_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Show stop button in bottom right if running
        if self.is_running:
            stop_text = "ESC to Stop"
            cv2.putText(canvas, stop_text, (frame_shape[1] - 120, frame_shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return canvas

def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
        
    augmented_canvas = AugmentedCanvas()
    
    # Get user input for question and timer
    if not augmented_canvas.get_user_input():
        print("Setup cancelled")
        return
    
    print("Controls:")
    print("- SPACE: Start timer and visualization")
    print("- ESC: Stop and exit")
    print("- Q: Quit")
    
    camera_window = 'Camera Feed'
    canvas_window = 'Augmented Canvas'
    
    # Create named windows
    cv2.namedWindow(camera_window, cv2.WINDOW_NORMAL)
    cv2.namedWindow(canvas_window, cv2.WINDOW_NORMAL)
    
    # Position windows side by side
    cv2.moveWindow(camera_window, 100, 100)
    cv2.moveWindow(canvas_window, 750, 100)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect objects in the frame
        detected_objects = augmented_canvas.detect_paper_objects(gray)
        
        # Create camera display with bounding boxes and info
        camera_display = frame.copy()
        
        # Add label to camera feed
        cv2.putText(camera_display, "Camera Feed", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(camera_display, f"Objects: {len(detected_objects)}", (10, camera_display.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw bounding boxes on camera feed
        for obj in detected_objects:
            x, y, w, h = obj['bbox']
            cv2.rectangle(camera_display, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Create canvas with overlays
        canvas = augmented_canvas.create_canvas(detected_objects, gray.shape)
        
        # Show both windows
        cv2.imshow(camera_window, camera_display)
        cv2.imshow(canvas_window, canvas)
        
        # Handle key presses from either window
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):  # Space bar
            if not augmented_canvas.is_running:
                augmented_canvas.start_timer()
        elif key == 27:  # ESC key
            if augmented_canvas.is_running:
                break  # Stop and exit
        
        # Check if timer expired
        if augmented_canvas.is_running and augmented_canvas.is_timer_expired():
            print("Timer expired!")
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()