import cv2
import numpy as np
import math

class AugmentedCanvas:
    def __init__(self):
        self.canvas_width = 640
        self.canvas_height = 480
        self.detected_objects = []
        self.proximity_threshold = 150  # Distance in pixels for clustering
        self.base_flower_size = 20      # Base flower size
        self.max_flower_size = 60       # Maximum flower size
        
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
                nearby_count = self.count_nearby_notes(obj, detected_objects)
                
                # Draw flower at object position with dynamic size
                self.draw_flower(canvas, (canvas_x, canvas_y), flower_size)
                
                # Add object info text with cluster info
                cv2.putText(canvas, f"Note (Cluster: {nearby_count + 1})", 
                           (obj['bbox'][0], obj['bbox'][1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return canvas

def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
        
    augmented_canvas = AugmentedCanvas()
    print("Press 'q' to quit. Place paper objects in front of camera to see flowers!")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect objects in the frame
        detected_objects = augmented_canvas.detect_paper_objects(gray)
        
        # Create canvas with overlays
        canvas = augmented_canvas.create_canvas(detected_objects, gray.shape)
        
        # Create side-by-side display
        height, width = frame.shape[:2]
        display = np.zeros((height, width * 2, 3), dtype=np.uint8)
        
        # Original color feed
        display[:height, :width] = frame
        
        # Augmented canvas
        display[:height, width:] = canvas
        
        # Add labels and object count
        cv2.putText(display, "Camera Feed", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display, "Augmented Canvas", (width + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display, f"Objects: {len(detected_objects)}", (10, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw bounding boxes on camera feed
        for obj in detected_objects:
            x, y, w, h = obj['bbox']
            cv2.rectangle(display[:height, :width], (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        cv2.imshow('Augmented Reality Canvas', display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()