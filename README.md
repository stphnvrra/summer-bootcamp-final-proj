# Building People Counter System

A real-time people counting system built with Flask that uses computer vision to automatically track people entering and exiting a building. The system provides accurate occupancy monitoring, daily statistics, and session management through a web interface.

## Features

- **Real-time People Detection**: Uses MobileNet SSD model for person detection
- **Entry/Exit Tracking**: Intelligent line-crossing detection for both directions
- **Live Occupancy Monitoring**: Real-time current occupancy with peak tracking
- **Session Management**: Track counting sessions with detailed statistics
- **Daily Analytics**: Comprehensive daily summaries and usage patterns
- **Modern Web Interface**: Responsive design with live camera feed and statistics

## Installation

### 1. Clone and Setup
```bash
git clone <repository-url>
cd building-people-counter
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install flask flask-sqlalchemy flask-bootstrap opencv-python numpy
```

### 3. Verify Model Files
Ensure these files are in the project directory:
- `MobileNetSSD_deploy.prototxt`
- `MobileNetSSD_deploy.caffemodel`

### 4. Run the Application
```bash
python app.py
```
Navigate to `http://localhost:5000` and allow camera access.

## Usage

### Camera Setup
1. Position camera to monitor entry/exit point with clear view
2. System divides view into left (exit) and right (entry) zones
3. Counting line appears in center - people crossing this line are counted
4. Ensure good lighting and minimal background clutter

### Counting Process
- Green bounding boxes show detected people
- System tracks individuals across frames with unique IDs
- Line crossing determines entry (left→right) or exit (right→left)
- Real-time counters update showing entries, exits, and current occupancy

## Project Structure

```
building-people-counter/
├── app.py                          # Main Flask application
├── camera_controller.py            # People tracking and counting
├── camera.py                       # Simple detection demo
├── MobileNetSSD_deploy.prototxt    # Model configuration
├── MobileNetSSD_deploy.caffemodel  # Pre-trained model weights
├── templates/counter.html          # Web interface
├── static/                         # CSS/JS/images
└── instance/people_counter.db      # SQLite database (auto-created)
```

## Technical Details

### Computer Vision
- **Person Detection**: MobileNet SSD at 300x300 resolution
- **Tracking**: Euclidean distance-based tracking across frames
- **Line Crossing**: Detects when people cross the counting line
- **Direction**: Left→Right = Entry, Right→Left = Exit

### Database Schema
- **CountSession**: Track sessions with start/end times and totals
- **CountEvent**: Individual entry/exit events with timestamps
- **DailyCount**: Daily summaries with occupancy statistics

### API Endpoints
- `GET /` - Main counter interface
- `GET /video_feed` - Video streaming
- `GET /start_counter` - Initialize system
- `GET /check_count_events` - Get new events
- `GET /stats` - Statistics dashboard
- `POST /end_session` - End counting session

## Configuration

Edit `camera_controller.py` to modify:
```python
# Camera settings
self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Detection settings
self.min_confidence = 0.5
self.min_distance_for_tracking = 50
self.counting_line_x = frame_width // 2  # Center line position
```

## Troubleshooting

### Common Issues
- **Camera not working**: Check permissions and ensure no other apps are using camera
- **Inaccurate counting**: Adjust lighting, camera position, or confidence threshold
- **Performance issues**: Close other apps, reduce resolution, or use GPU acceleration
- **Database issues**: Check write permissions for `instance/` directory

### Debug Mode
```bash
export FLASK_DEBUG=1  # On Windows: set FLASK_DEBUG=1
python app.py
```

Test basic detection:
```bash
python camera.py  # Simple detection demo
```

## Use Cases

- **Building Management**: Track occupancy and usage patterns
- **Security**: Monitor access and emergency evacuation
- **Analytics**: Optimize staffing and energy management
- **Compliance**: Enforce capacity limits and safety regulations


## Acknowledgments

- **MobileNet SSD**: Google Research
- **OpenCV**: Computer vision library
- **Flask**: Web framework
- **Bootstrap**: Frontend framework

---
