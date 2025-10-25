# Brainstorming Collaboration Research Project

## Project Overview

This repository contains a comprehensive research project exploring brainstorming collaboration challenges and solutions in university settings. The project includes user research, analysis, and a proof-of-concept augmented reality prototype designed to enhance collaborative brainstorming experiences.

## Repository Structure

### [Documentation & Research](./artefacts/)
- **[Artifacts](./artefacts/)** - Promotional materials, posters, ethical disclaimers, and presentation materials used in the trade show

### [Research Data](./interviews/)
- **[Interview Scripts](./interviews/interview%20scripts.md)** - Comprehensive user interviews conducted with university students across multiple disciplines to understand brainstorming and collaboration challenges

### [Visual Assets](./assets/)
Contains all visual materials including:
- Affinity diagrams and theme analysis
- User research documentation photos
- Prototype sketches and design materials
- Process documentation imagery

### [Prototype](./augmented_canvas.py)
- **[Augmented Canvas](./augmented_canvas.py)** - Interactive Python-based prototype implementing computer vision and augmented reality features for collaborative brainstorming activities

## Project Purpose

This research project investigates the challenges students face in university brainstorming sessions and explores technological solutions to improve ideation, collaboration, and engagement. The findings are based on extensive user interviews across various disciplines including Engineering, Computer Science, Architecture, and Interaction Design.

## Research Methodology

Our research methodology included:

1. **User Interviews** - Conducted interviews with students from diverse academic backgrounds
2. **Thematic Analysis** - Identified key themes and patterns in brainstorming and collaboration challenges
3. **Prototype Development** - Created an interactive proof-of-concept solution for brainstorming enhancement
4. **User Testing** - Evaluated the prototype with target users in brainstorming contexts

## Key Findings

Based on our research, we identified several critical themes in brainstorming sessions:

- **Participation Barriers** - Students struggle with equal participation and idea contribution
- **Idea Organization** - Difficulty in clustering and building upon related concepts
- **Engagement Challenges** - Varying levels of participation and creative investment
- **Process Structure** - Need for better facilitation and time management
- **Visual Feedback** - Lack of immediate visual representation of collaborative progress

## Prototype: Augmented Collaboration Canvas

### Overview
The **Augmented Collaboration Canvas** is an interactive proof-of-concept prototype that uses computer vision and augmented reality to enhance brainstorming and ideation sessions. The system detects colored post-it notes placed on a physical canvas and provides real-time visual feedback through augmented overlays to improve idea clustering and participation visualization.

### Features
- **Real-time Object Detection** - Identifies colored post-it notes (pink, yellow, blue, green) using computer vision
- **Proximity-based Clustering** - Groups nearby ideas with visual connections
- **Dynamic Visual Feedback** - Displays flowers or rings that grow based on the number of clustered ideas
- **Timer Integration** - Built-in session timers for structured brainstorming activities
- **Flexible Display Modes** - Supports both windowed and full screen presentation modes
- **Customizable Aesthetics** - Choose between flower and ring visualization styles

### Use Cases
- **University Brainstorming** - Facilitate student idea generation with visual clustering
- **Design Workshops** - Enhance engagement through interactive feedback during ideation
- **Creative Sessions** - Support concept development and idea building
- **Research Studies** - Investigate brainstorming behavior and participation patterns

## Installation & Setup

### Prerequisites
- Python 3.7 or higher
- Webcam or external camera (built-in laptop cameras, USB cameras, or network cameras supported)
- macOS, Windows, or Linux operating system

### Dependencies
The prototype requires the following Python packages:
- `opencv-python` - Computer vision and camera input
- `numpy` - Numerical computations for image processing

### Installation Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/teamwork-collaboration.git
   cd teamwork-collaboration/teamwork-collaboration
   ```

2. **Set Up Virtual Environment** (Recommended)
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install opencv-python numpy
   ```

## Usage Instructions

### Running the Prototype
```bash
python3 augmented_canvas.py
```

The application will automatically detect and test available cameras when it starts. This process includes:

**Camera Detection Process:**
1. **Automatic Scanning** - The system tests camera indices 0-10 to find available devices
2. **Multiple Camera Types** - Supports built-in laptop cameras, USB cameras, and network cameras (including iPhone cameras via network)
3. **Connection Testing** - Each camera is tested with multiple attempts and longer delays for network cameras
4. **User Selection** - If multiple cameras are found, you'll be prompted to choose which one to use

**Camera Selection:**
- The system will display available cameras with their indices
- For network cameras (like iPhone cameras), make sure to approve the connection when prompted
- Choose the camera that provides the best view of your brainstorming workspace

### Interactive Controls
Once the application is running, use these keyboard shortcuts:

- **SPACE** - Start timer and visualization
- **ESC** - Stop and exit
- **I** - Toggle workflow guide
- **D** - Toggle debug mode (show color detection masks)
- **+/=** - Zoom in camera
- **-** - Zoom out camera
- **R** - Reset camera zoom to 1.0x
- **Q** - Quit

### Setting Up a Brainstorming Session

1. **Prepare Materials**
   - Gather colored post-it notes (pink, yellow, blue, green work best)
   - Ensure good lighting conditions
   - Position camera to capture the workspace

2. **Configure Session**
   - Use **I** to toggle the workflow guide for participants
   - Use **+/-** to adjust camera zoom for optimal viewing
   - Use **R** to reset camera zoom if needed

3. **Facilitate the Session**
   - Press **SPACE** to start the timer and visualization
   - Participants place post-it notes on the canvas
   - Watch as the system clusters related ideas with visual feedback
   - Use **D** to toggle debug mode if troubleshooting is needed

### Tips for Best Results
- Use bright, saturated colored post-it notes
- Ensure consistent lighting without strong shadows
- Place post-its flat against the surface
- Maintain adequate spacing between different idea clusters
- Position camera directly above or at an angle to minimize perspective distortion

## Technical Details

### Computer Vision Pipeline
1. **Color Space Conversion** - BGR to HSV for robust color detection
2. **Color Range Filtering** - Separate masks for each post-it color
3. **Morphological Operations** - Clean up noise and separate touching objects
4. **Contour Detection** - Identify individual post-it boundaries
5. **Clustering Algorithm** - Group nearby objects based on proximity threshold

### Configuration Options
The prototype includes several customizable parameters:
- `proximity_threshold` - Distance for clustering objects (default: 300 pixels)
- `base_flower_size` - Minimum visualization size (default: 20 pixels)
- `max_flower_size` - Maximum visualization size (default: 60 pixels)
- `timer_duration` - Default session length (default: 60 seconds)

## Login Credentials

**Note**: This prototype does not require any login credentials as it operates as a standalone application. Simply run the Python script to begin using the system.

## Troubleshooting

### Common Issues

**Camera not detected:**
- Ensure camera permissions are granted to the application
- Check that the camera is not being used by another application
- For network cameras (iPhone, etc.), ensure they're connected to the same network
- Try reconnecting USB cameras or restarting the application
- On macOS, check System Preferences > Security & Privacy > Camera
- The system automatically tests cameras 0-10, but you can manually specify a different index if needed

**Poor color detection:**
- Improve lighting conditions
- Use brighter, more saturated post-it colors
- Adjust HSV color ranges in the code if needed

**Performance issues:**
- Close other camera applications
- Reduce resolution or frame rate
- Ensure adequate system resources

## Support & Contact

For questions about the research project or technical issues with the prototype, please refer to the research documentation in this repository or contact the development team through the repository's issue tracker. 
