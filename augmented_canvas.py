import cv2
import numpy as np
import math
import time

class AugmentedCanvas:
    def __init__(self):
        self.canvas_width = 640
        self.canvas_height = 480
        self.detected_objects = []
        self.proximity_threshold = 300  # Fixed distance in pixels for clustering (much larger)
        self.base_flower_size = 20      # Base flower size
        self.max_flower_size = 60       # Maximum flower size
        self.question = ""
        self.timer_duration = 60  # Default 60 seconds
        self.start_time = None
        self.is_running = False
        self.timer_expired = False
        self.show_debug = False
        self.fullscreen = False
        self.design_style = 'flowers'  # 'flowers' or 'rings'
        
    def detect_postit_objects(self, color_frame):
        # Convert BGR to HSV for better color detection
        hsv = cv2.cvtColor(color_frame, cv2.COLOR_BGR2HSV)
        
        # Define HSV ranges for post-it colors
        # Bright Pink post-its (pink can appear in two HSV ranges)
        pink_lower1 = np.array([140, 50, 50])   # Lower magenta/pink range
        pink_upper1 = np.array([180, 255, 255])
        pink_lower2 = np.array([0, 50, 50])     # Higher red/pink range  
        pink_upper2 = np.array([20, 255, 255])
        
        # Yellow post-its (broader range for different lighting)
        yellow_lower = np.array([15, 50, 50])
        yellow_upper = np.array([35, 255, 255])
        
        # Blue post-its (broader range)
        blue_lower = np.array([90, 50, 50])
        blue_upper = np.array([130, 255, 255])
        
        # Green post-its (broader range)
        green_lower = np.array([35, 50, 50])
        green_upper = np.array([85, 255, 255])
        
        # Create masks for each color
        pink_mask1 = cv2.inRange(hsv, pink_lower1, pink_upper1)
        pink_mask2 = cv2.inRange(hsv, pink_lower2, pink_upper2)
        pink_mask = cv2.bitwise_or(pink_mask1, pink_mask2)  # Combine pink ranges
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        
        # Combine all masks
        combined_mask = cv2.bitwise_or(pink_mask, yellow_mask)
        combined_mask = cv2.bitwise_or(combined_mask, blue_mask)
        combined_mask = cv2.bitwise_or(combined_mask, green_mask)
        
        # Clean up the mask with morphological operations
        # Use smaller kernel to avoid merging separate post-its
        kernel_small = np.ones((3,3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_small)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_small)
        
        # Apply erosion to separate touching objects, then dilate to restore size
        kernel_separate = np.ones((2,2), np.uint8)
        combined_mask = cv2.erode(combined_mask, kernel_separate, iterations=1)
        combined_mask = cv2.dilate(combined_mask, kernel_separate, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Reduced minimum size for post-its
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # Check if this might be multiple post-its merged together
                avg_postit_area = 2000  # Approximate area of a single post-it
                if area > avg_postit_area * 1.5:  # Likely multiple post-its
                    # Try to split using watershed or distance transform
                    contour_mask = np.zeros(combined_mask.shape, np.uint8)
                    cv2.drawContours(contour_mask, [contour], -1, 255, -1)
                    
                    # Use distance transform to find peaks (centers of post-its)
                    dist_transform = cv2.distanceTransform(contour_mask, cv2.DIST_L2, 5)
                    _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
                    sure_fg = np.uint8(sure_fg)
                    
                    # Find connected components (potential post-it centers)
                    n_labels, labels = cv2.connectedComponents(sure_fg)
                    
                    # Create individual objects for each component
                    for label in range(1, n_labels):
                        component_mask = (labels == label).astype(np.uint8) * 255
                        component_contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        if component_contours:
                            cx, cy, cw, ch = cv2.boundingRect(component_contours[0])
                            if cw * ch > 200:  # Minimum size for split component
                                center_x = cx + cw // 2
                                center_y = cy + ch // 2
                                
                                # Determine color for this component
                                roi = hsv[cy:cy+ch, cx:cx+cw]
                                pink_pixels = np.sum(cv2.inRange(roi, pink_lower1, pink_upper1)) + np.sum(cv2.inRange(roi, pink_lower2, pink_upper2))
                                yellow_pixels = np.sum(cv2.inRange(roi, yellow_lower, yellow_upper))
                                blue_pixels = np.sum(cv2.inRange(roi, blue_lower, blue_upper))
                                green_pixels = np.sum(cv2.inRange(roi, green_lower, green_upper))
                                
                                # Find the color with the most matching pixels
                                color_counts = {'pink': pink_pixels, 'yellow': yellow_pixels, 'blue': blue_pixels, 'green': green_pixels}
                                color = max(color_counts, key=color_counts.get)
                                
                                objects.append({
                                    'type': 'postit',
                                    'color': color,
                                    'center': (center_x, center_y),
                                    'bbox': (cx, cy, cw, ch),
                                    'area': cw * ch
                                })
                else:
                    # Single post-it - process normally
                    if 0.3 < aspect_ratio < 3.0:  # More lenient aspect ratio for post-its
                        center_x = x + w // 2
                        center_y = y + h // 2
                        
                        # Determine which color this post-it is
                        roi = hsv[y:y+h, x:x+w]
                        pink_pixels = np.sum(cv2.inRange(roi, pink_lower1, pink_upper1)) + np.sum(cv2.inRange(roi, pink_lower2, pink_upper2))
                        yellow_pixels = np.sum(cv2.inRange(roi, yellow_lower, yellow_upper))
                        blue_pixels = np.sum(cv2.inRange(roi, blue_lower, blue_upper))
                        green_pixels = np.sum(cv2.inRange(roi, green_lower, green_upper))
                        
                        # Choose the color with the most matching pixels
                        color_counts = {'pink': pink_pixels, 'yellow': yellow_pixels, 'blue': blue_pixels, 'green': green_pixels}
                        color = max(color_counts, key=color_counts.get)
                        
                        objects.append({
                            'type': 'postit',
                            'color': color,
                            'center': (center_x, center_y),
                            'bbox': (x, y, w, h),
                            'area': area
                        })
        
        # Store masks for debug visualization if needed
        self.debug_masks = {
            'pink': pink_mask,
            'yellow': yellow_mask, 
            'blue': blue_mask,
            'green': green_mask,
            'combined': combined_mask
        }
        
        return objects
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def calculate_relative_clustering_threshold(self, all_objects):
        """Calculate dynamic clustering threshold based on overall post-it distribution"""
        if len(all_objects) < 2:
            return 120  # Default threshold for single post-it
        
        # Calculate all pairwise distances
        distances = []
        for i, obj1 in enumerate(all_objects):
            for obj2 in all_objects[i+1:]:
                if obj1['type'] == 'postit' and obj2['type'] == 'postit':
                    distance = self.calculate_distance(obj1['center'], obj2['center'])
                    distances.append(distance)
        
        if not distances:
            return 120
        
        # Use a percentage of the average distance as clustering threshold
        # This way, only relatively close post-its cluster together
        avg_distance = sum(distances) / len(distances)
        clustering_threshold = avg_distance * 0.6  # 60% of average distance
        
        # Set reasonable bounds (between 50 and 200 pixels)
        clustering_threshold = max(50, min(200, clustering_threshold))
        
        return clustering_threshold

    def count_nearby_notes(self, target_obj, all_objects):
        """Count how many notes are within relative proximity threshold of the target object"""
        nearby_count = 0
        target_center = target_obj['center']
        
        # Calculate dynamic threshold based on distribution
        dynamic_threshold = self.calculate_relative_clustering_threshold(all_objects)
        
        for obj in all_objects:
            if obj != target_obj:  # Don't count itself
                distance = self.calculate_distance(target_center, obj['center'])
                if distance <= dynamic_threshold:
                    nearby_count += 1
        
        return nearby_count
    
    def group_postits_into_clusters(self, all_objects):
        """Group post-its into clusters based on fixed distance threshold"""
        postits = [obj for obj in all_objects if obj['type'] == 'postit']
        if not postits:
            return []
        
        clusters = []
        used_postits = set()
        
        for i, postit in enumerate(postits):
            if i in used_postits:
                continue
                
            # Start a new cluster with this post-it
            cluster = [postit]
            used_postits.add(i)
            
            # Find all post-its connected to this cluster (using fixed distance)
            to_check = [postit]
            while to_check:
                current = to_check.pop(0)
                
                # Check all remaining post-its
                for j, other_postit in enumerate(postits):
                    if j not in used_postits:
                        distance = self.calculate_distance(current['center'], other_postit['center'])
                        if distance <= self.proximity_threshold:  # Use fixed threshold
                            cluster.append(other_postit)
                            used_postits.add(j)
                            to_check.append(other_postit)
            
            clusters.append(cluster)
        
        return clusters
    
    def calculate_cluster_center(self, cluster):
        """Calculate the center point of a cluster of post-its"""
        if not cluster:
            return (0, 0)
        
        total_x = sum(postit['center'][0] for postit in cluster)
        total_y = sum(postit['center'][1] for postit in cluster)
        
        return (total_x // len(cluster), total_y // len(cluster))
    
    def calculate_flower_properties(self, note_obj, all_objects):
        """Calculate flower size and petal count based on nearby notes clustering"""
        nearby_count = self.count_nearby_notes(note_obj, all_objects)
        
        # Calculate petal count: 1 petal for isolated note, more for clustered notes
        # 1 note = 1 petal, 2-3 notes = 3 petals, 4-5 notes = 5 petals, etc.
        if nearby_count == 0:
            petal_count = 1  # Just this note
        elif nearby_count <= 2:
            petal_count = 3  # Small cluster
        elif nearby_count <= 4:
            petal_count = 5  # Medium cluster
        elif nearby_count <= 6:
            petal_count = 7  # Large cluster
        else:
            petal_count = min(12, 8 + nearby_count // 2)  # Very large cluster
        
        # Scale flower size based on cluster density
        # More nearby notes = bigger flower
        size_multiplier = 1 + (nearby_count * 0.3)  # Each nearby note adds 30% to size
        new_size = min(self.base_flower_size * size_multiplier, self.max_flower_size)
        
        return int(new_size), petal_count
    
    def detect_available_cameras(self):
        """Detect available cameras and return list of camera indices"""
        available_cameras = []
        
        print("Scanning for cameras (this may take a moment for network cameras)...")
        print("If you have an iPhone, make sure to approve the camera connection when prompted!")
        
        # Test cameras 0-10 (most systems won't have more than this)
        for i in range(10):
            print(f"Testing camera {i}...", end=" ", flush=True)
            cap = cv2.VideoCapture(i)
            
            if cap.isOpened():
                # Set properties for better network camera support
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                # Try multiple times with longer delays for network cameras
                success = False
                for attempt in range(6):  # Increased attempts
                    if attempt > 0:
                        print(".", end="", flush=True)
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        success = True
                        break
                    time.sleep(1.0)  # Increased wait time to 1 second
                
                if success:
                    available_cameras.append(i)
                    print(" ✓ Available")
                else:
                    print(" ✗ No signal (timeout)")
                cap.release()
            else:
                print(" ✗ Not found")
        
        return available_cameras
    
    def select_camera(self):
        """Let user select which camera to use"""
        print("\n=== Camera Selection ===")
        available_cameras = self.detect_available_cameras()
        
        if not available_cameras:
            print("No cameras detected automatically!")
            try:
                manual_input = input("Enter camera ID manually (e.g., 1, 2) or press Enter to exit: ").strip()
                if manual_input:
                    camera_id = int(manual_input)
                    print(f"Attempting to use camera {camera_id}")
                    return camera_id
                else:
                    return None
            except (ValueError, EOFError):
                print("Invalid input")
                return None
        
        if len(available_cameras) == 1:
            print(f"Using camera {available_cameras[0]} (only available camera)")
            return available_cameras[0]
        
        print("Available cameras:")
        for i, cam_id in enumerate(available_cameras):
            print(f"  {i + 1}. Camera {cam_id}")
        print(f"  {len(available_cameras) + 1}. Enter custom camera ID")
        
        try:
            choice = input(f"Select option (1-{len(available_cameras) + 1}, default 1): ").strip()
            if not choice:
                choice = "1"
            
            choice_idx = int(choice) - 1
            if choice_idx == len(available_cameras):
                # Custom camera ID option
                custom_id = input("Enter camera ID: ").strip()
                try:
                    camera_id = int(custom_id)
                    print(f"Using custom camera {camera_id}")
                    return camera_id
                except ValueError:
                    print("Invalid camera ID, using first available camera")
                    return available_cameras[0]
            elif 0 <= choice_idx < len(available_cameras):
                selected_camera = available_cameras[choice_idx]
                print(f"Selected camera {selected_camera}")
                return selected_camera
            else:
                print("Invalid choice, using first available camera")
                return available_cameras[0]
        except (ValueError, EOFError):
            print("Invalid input, using first available camera")
            return available_cameras[0]
    
    def get_user_input(self):
        """Get question, timer duration, and design style from user via console"""
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
                
            # Get design style
            print("\nDesign styles:")
            print("1. Flowers (petals represent clustered post-its)")
            print("2. Rings (concentric rings represent clustered post-its)")
            design_input = input("Choose design style (1 for flowers, 2 for rings, default 1): ").strip()
            if design_input == "2":
                design_style = 'rings'
            else:
                design_style = 'flowers'
                
        except EOFError:
            # If running in environment without interactive input, use defaults
            print("Using default values...")
            question = "What do you think about this?"
            timer_duration = 60
            design_style = 'flowers'
            
        self.question = question
        self.timer_duration = timer_duration
        self.design_style = design_style
        print(f"Question: {self.question}")
        print(f"Timer: {self.timer_duration} seconds")
        print(f"Design: {self.design_style}")
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
        if self.get_remaining_time() <= 0 and self.is_running:
            self.timer_expired = True
            self.is_running = False
        return self.timer_expired
    
    def draw_flower(self, canvas, center, size=20, petal_count=1, cluster_colors=None):
        x, y = center
        
        # Define colors (BGR format for OpenCV)
        if cluster_colors and len(cluster_colors) > 0:
            # Use the most common post-it color in the cluster for petals
            most_common_color = max(set(cluster_colors), key=cluster_colors.count)
            if most_common_color == 'pink':
                petal_color = (255, 0, 255)    # Magenta for pink post-its
            elif most_common_color == 'yellow':
                petal_color = (0, 255, 255)    # Cyan for yellow post-its
            elif most_common_color == 'blue':
                petal_color = (255, 0, 0)      # Blue for blue post-its
            elif most_common_color == 'green':
                petal_color = (0, 255, 0)      # Green for green post-its
            else:
                petal_color = (147, 20, 255)   # Default purple
        else:
            petal_color = (147, 20, 255)       # Default purple petals
            
        center_color = (0, 255, 255)    # Yellow center  
        stem_color = (0, 255, 0)        # Green stem
        
        # Ensure at least 1 petal, maximum 12 petals
        petal_count = max(1, min(petal_count, 12))
        
        # Calculate petal positions around the center
        import math
        petal_positions = []
        
        if petal_count == 1:
            # Single petal at the top
            petal_positions = [(x, y - size)]
        else:
            # Multiple petals arranged in a circle
            angle_step = 2 * math.pi / petal_count
            for i in range(petal_count):
                angle = i * angle_step - math.pi/2  # Start from top
                petal_x = x + int(size * math.cos(angle))
                petal_y = y + int(size * math.sin(angle))
                petal_positions.append((petal_x, petal_y))
        
        # Draw petals - make them larger and more visible for higher petal counts
        if petal_count <= 4:
            petal_size = max(size//4, 10)
        else:
            # For 5+ petals, make them larger to be more visible
            petal_size = max(size//3, 12)
        
        for petal_x, petal_y in petal_positions:
            cv2.circle(canvas, (petal_x, petal_y), petal_size, petal_color, -1)
        
        # Flower center (keep smaller so it doesn't cover petals)
        center_size = max(size//8, 4) + min(petal_count - 1, 3)  # Cap center growth
        cv2.circle(canvas, (x, y), center_size, center_color, -1)
        
        # Stem (thicker for more petals)
        stem_thickness = max(2, min(petal_count, 6))
        cv2.line(canvas, (x, y + center_size), (x, y + size * 2), stem_color, stem_thickness)
    
    def draw_rings(self, canvas, center, base_radius=30, cluster_colors=None):
        """Draw separate colored rings for each post-it with spacing between them"""
        x, y = center
        
        # Define default colors if no cluster colors provided
        if not cluster_colors or len(cluster_colors) == 0:
            cluster_colors = ['purple']  # Default
        
        # Ring properties
        ring_thickness = 8  # Thickness of each ring
        ring_spacing = 15   # Space between rings
        
        # Draw rings from inside to outside, one for each post-it
        for i, color in enumerate(cluster_colors):
            # Calculate radius for this ring (starting from center and moving outward)
            ring_radius = base_radius + (i * ring_spacing)
            
            # Determine color (BGR format for OpenCV)
            if color == 'pink':
                ring_color = (255, 0, 255)    # Magenta for pink post-its
            elif color == 'yellow':
                ring_color = (0, 255, 255)    # Cyan for yellow post-its
            elif color == 'blue':
                ring_color = (255, 0, 0)      # Blue for blue post-its
            elif color == 'green':
                ring_color = (0, 255, 0)      # Green for green post-its
            else:
                ring_color = (147, 20, 255)   # Default purple
            
            # Draw the ring (hollow circle)
            cv2.circle(canvas, (x, y), ring_radius, ring_color, ring_thickness)
        
        # Draw a small center dot to mark the cluster center
        cv2.circle(canvas, (x, y), 5, (255, 255, 255), -1)  # White center dot
    
    def create_canvas(self, detected_objects, frame_shape):
        canvas = np.zeros((frame_shape[0], frame_shape[1], 3), dtype=np.uint8)
        
        # Display question at top center if available
        if self.question:
            text_size = cv2.getTextSize(self.question, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = (frame_shape[1] - text_size[0]) // 2
            cv2.putText(canvas, self.question, (text_x, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Display timer or discussion message
        if self.is_running and not self.timer_expired:
            remaining = self.get_remaining_time()
            timer_text = f"Time: {int(remaining)}s"
            cv2.putText(canvas, timer_text, (frame_shape[1] - 150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        elif self.timer_expired:
            discussion_text = "Start Discussing!"
            text_size = cv2.getTextSize(discussion_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            text_x = (frame_shape[1] - text_size[0]) // 2
            cv2.putText(canvas, discussion_text, (text_x, frame_shape[0] // 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        # Show flowers if timer is running or has expired
        if self.is_running or self.timer_expired:
            # Group post-its into clusters using simple fixed distance
            clusters = self.group_postits_into_clusters(detected_objects)
            
            # Debug: show clustering info
            if self.show_debug:
                cv2.putText(canvas, f"Clusters: {len(clusters)}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                for i, cluster in enumerate(clusters):
                    cv2.putText(canvas, f"C{i}: {len(cluster)} post-its", (10, 90 + i*25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Connection lines removed - flowers now appear without connecting lines
            
            # Draw visualization based on selected design style
            for cluster in clusters:
                cluster_center = self.calculate_cluster_center(cluster)
                cluster_size = len(cluster)  # Number of post-its in cluster
                
                # Collect colors from all post-its in this cluster
                cluster_colors = [postit['color'] for postit in cluster]
                
                if self.design_style == 'rings':
                    # Calculate ring radius based on cluster size
                    base_radius = min(30 + (cluster_size - 1) * 10, 80)
                    self.draw_rings(canvas, cluster_center, base_radius, cluster_colors)
                else:  # default to flowers
                    # Calculate flower size based on cluster size
                    base_size = 25
                    flower_size = min(base_size + (cluster_size - 1) * 5, self.max_flower_size)
                    self.draw_flower(canvas, cluster_center, flower_size, cluster_size, cluster_colors)
                
                # Debug: show cluster info if debug mode is on
                if self.show_debug:
                    if self.design_style == 'rings':
                        debug_text = f"{cluster_size}r (rad:{base_radius})"
                    else:
                        debug_text = f"{cluster_size}p (size:{flower_size})"
                    cv2.putText(canvas, debug_text, (cluster_center[0] + 20, cluster_center[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Debug: show all pairwise distances between post-its
            if self.show_debug:
                postits = [obj for obj in detected_objects if obj['type'] == 'postit']
                y_offset = 200
                for i, obj1 in enumerate(postits):
                    for j, obj2 in enumerate(postits[i+1:], i+1):
                        distance = self.calculate_distance(obj1['center'], obj2['center'])
                        should_cluster = "YES" if distance <= self.proximity_threshold else "NO"
                        cv2.putText(canvas, f"D({i},{j}): {int(distance)}px -> {should_cluster}", 
                                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                        y_offset += 20
        
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
    
    def connect_to_iphone_camera(self, camera_id):
        """Special connection method for iPhone cameras with extended patience"""
        print(f"\n=== Connecting to iPhone Camera {camera_id} ===")
        print("IMPORTANT: On your iPhone:")
        print("1. Open the Camera app and keep it open")
        print("2. When prompted, tap 'Allow' for camera access")
        print("3. Keep your iPhone unlocked during the session")
        print("4. Stay on the same WiFi network")
        print("\nAttempting connection...")
        
        # Multiple connection attempts with different strategies
        for strategy in range(3):
            print(f"\nConnection attempt {strategy + 1}/3...")
            
            cap = cv2.VideoCapture(camera_id)
            
            if not cap.isOpened():
                print("  ✗ Camera device not found")
                continue
            
            # Strategy-specific settings
            if strategy == 0:
                # Standard approach
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FPS, 30)
            elif strategy == 1:
                # Lower settings for stability
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FPS, 15)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            else:
                # Minimal settings
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                cap.set(cv2.CAP_PROP_FPS, 10)
            
            print("  Waiting for iPhone approval and signal...", end="", flush=True)
            
            # Extended connection attempt with very long timeout
            success = False
            for attempt in range(20):  # 20 seconds total
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    # Verify we can read multiple frames
                    frame_count = 0
                    for verify in range(5):
                        ret2, frame2 = cap.read()
                        if ret2 and frame2 is not None:
                            frame_count += 1
                        time.sleep(0.1)
                    
                    if frame_count >= 3:  # At least 3 successful frames
                        success = True
                        print(" ✓ CONNECTED!")
                        print(f"  Frame size: {frame.shape}")
                        return cap
                    
                print(".", end="", flush=True)
                time.sleep(1.0)
            
            print(" ✗ Timeout")
            cap.release()
        
        print("\n❌ Failed to connect to iPhone camera")
        print("\nTroubleshooting:")
        print("- Make sure your iPhone is unlocked")
        print("- Try opening Camera app on iPhone first")
        print("- Check WiFi connection on both devices")
        print("- Try a different camera ID (0, 2, 3, etc.)")
        return None

def main():
    augmented_canvas = AugmentedCanvas()
    
    # Select camera
    camera_id = augmented_canvas.select_camera()
    if camera_id is None:
        print("No camera available")
        return
    
    # Use special iPhone connection method for network cameras
    if camera_id > 0:
        cap = augmented_canvas.connect_to_iphone_camera(camera_id)
        if cap is None:
            return
    else:
        # Standard connection for built-in cameras
        print(f"Connecting to camera {camera_id}...")
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        # Test built-in camera
        ret, test_frame = cap.read()
        if not ret or test_frame is None:
            print("Error: Camera opened but no video signal")
            cap.release()
            return
        
        print("✓ Built-in camera connected!")
        
    # Get user input for question and timer
    if not augmented_canvas.get_user_input():
        print("Setup cancelled")
        return
    
    print("Controls:")
    print("- SPACE: Start timer and visualization")
    print("- ESC: Stop and exit")
    print("- D: Toggle debug mode (show color detection masks)")
    print("- Q: Quit")
    
    camera_window = 'Camera Feed'
    canvas_window = 'Augmented Canvas'
    
    # Create named windows
    cv2.namedWindow(camera_window, cv2.WINDOW_NORMAL)
    cv2.namedWindow(canvas_window, cv2.WINDOW_NORMAL)
    
    # Position windows side by side
    cv2.moveWindow(camera_window, 100, 100)
    cv2.moveWindow(canvas_window, 750, 100)
    
    # Keep track of failed frame reads for connection recovery
    failed_reads = 0
    max_failed_reads = 5
    
    while True:
        ret, frame = cap.read()
        if not ret:
            failed_reads += 1
            print(f"Warning: Failed to read frame (attempt {failed_reads}/{max_failed_reads})")
            
            if failed_reads >= max_failed_reads:
                print("Error: Lost camera connection")
                if camera_id > 0:  # Network camera (iPhone)
                    print("iPhone may have disconnected. Try:")
                    print("1. Keep the Camera app open on your iPhone")
                    print("2. Ensure devices stay on the same network")
                    print("3. Re-run the program and re-approve camera access")
                break
            
            time.sleep(0.5)  # Wait before retrying
            continue
        else:
            failed_reads = 0  # Reset counter on successful read
        
        # Detect post-it objects in the frame (using color frame instead of gray)
        detected_objects = augmented_canvas.detect_postit_objects(frame)
        
        # Create camera display with bounding boxes and info
        camera_display = frame.copy()
        
        # Add label to camera feed
        cv2.putText(camera_display, "Camera Feed", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # Count post-its by color for debug info
        color_counts = {'pink': 0, 'yellow': 0, 'blue': 0, 'green': 0}
        for obj in detected_objects:
            color_counts[obj['color']] += 1
        
        cv2.putText(camera_display, f"Post-its: {len(detected_objects)} (P:{color_counts['pink']} Y:{color_counts['yellow']} B:{color_counts['blue']} G:{color_counts['green']})", 
                   (10, camera_display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw bounding boxes on camera feed with color coding
        for obj in detected_objects:
            x, y, w, h = obj['bbox']
            # Color code the bounding boxes based on post-it color
            if obj['color'] == 'pink':
                box_color = (255, 0, 255)  # Magenta for pink
            elif obj['color'] == 'yellow':
                box_color = (0, 255, 255)  # Cyan for yellow
            elif obj['color'] == 'blue':
                box_color = (255, 0, 0)    # Blue
            elif obj['color'] == 'green':
                box_color = (0, 255, 0)    # Green
            else:
                box_color = (255, 255, 255)  # White default
            
            cv2.rectangle(camera_display, (x, y), (x+w, y+h), box_color, 2)
            # Add color label
            cv2.putText(camera_display, obj['color'], (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        
        # Create canvas with overlays
        canvas = augmented_canvas.create_canvas(detected_objects, frame.shape)
        
        # Show both windows
        cv2.imshow(camera_window, camera_display)
        cv2.imshow(canvas_window, canvas)
        
        # Show debug window if enabled
        if augmented_canvas.show_debug and hasattr(augmented_canvas, 'debug_masks'):
            debug_display = np.zeros((frame.shape[0], frame.shape[1] * 2, 3), dtype=np.uint8)
            
            # Left side: combined mask
            debug_display[:, :frame.shape[1], 0] = augmented_canvas.debug_masks['combined']
            debug_display[:, :frame.shape[1], 1] = augmented_canvas.debug_masks['combined'] 
            debug_display[:, :frame.shape[1], 2] = augmented_canvas.debug_masks['combined']
            
            # Right side: individual color masks (4 colors now, so divide into quarters)
            h = frame.shape[0] // 4
            debug_display[:h, frame.shape[1]:, 0] = augmented_canvas.debug_masks['pink'][:h, :]  # Pink in red channel
            debug_display[h:2*h, frame.shape[1]:, 1] = augmented_canvas.debug_masks['yellow'][h:2*h, :]  # Yellow in green
            debug_display[2*h:3*h, frame.shape[1]:, 2] = augmented_canvas.debug_masks['blue'][2*h:3*h, :]  # Blue in blue
            debug_display[3*h:, frame.shape[1]:, 1] = augmented_canvas.debug_masks['green'][3*h:, :]  # Green in green
            
            cv2.putText(debug_display, "Combined Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(debug_display, "Pink Mask", (frame.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(debug_display, "Yellow Mask", (frame.shape[1] + 10, h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(debug_display, "Blue Mask", (frame.shape[1] + 10, 2*h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(debug_display, "Green Mask", (frame.shape[1] + 10, 3*h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Color Debug", debug_display)
        
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
        elif key == ord('d'):  # D key for debug
            augmented_canvas.show_debug = not augmented_canvas.show_debug
            if not augmented_canvas.show_debug:
                cv2.destroyWindow("Color Debug")
            print(f"Debug mode: {'ON' if augmented_canvas.show_debug else 'OFF'}")
        
        # Check if timer expired (but don't close, just update state)
        if augmented_canvas.is_running and augmented_canvas.is_timer_expired():
            print("Timer expired! Time to start discussing!")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()