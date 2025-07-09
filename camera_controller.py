import cv2
import numpy as np
import time
from datetime import datetime
import os
from threading import Thread, Event
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PeopleCounterCamera:
    def __init__(self):
        """Initialize the people counter camera system"""
        # Camera settings
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Load MobileNet SSD model for person detection
        self.prototxt = 'MobileNetSSD_deploy.prototxt'
        self.model = 'MobileNetSSD_deploy.caffemodel'
        self.net = cv2.dnn.readNetFromCaffe(self.prototxt, self.model)
        
        # Class labels for MobileNet SSD
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                       "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                       "dog", "horse", "motorbike", "person", "pottedplant",
                       "sheep", "sofa", "train", "tvmonitor"]
        
        # People counting state variables
        self.tracking_data = {}  # Store tracking info for detected people
        self.next_person_id = 1
        self.count_events = []  # Store count events to be saved to database
        self.backup_events = []  # Backup storage in case database fails
        self.max_backup_events = 1000  # Limit backup storage
        
        # Counting zones (will be set based on frame dimensions)
        self.entry_zone = None
        self.exit_zone = None
        self.counting_line_x = None
        
        # Tracking settings
        self.max_disappeared = 10  # Frames before removing tracker
        self.min_confidence = 0.5
        self.min_distance_for_tracking = 50  # Minimum distance for tracking
        
        logger.info("PeopleCounterCamera initialized successfully")
    
    def setup_counting_zones(self, frame_width, frame_height):
        """Set up entry/exit zones based on frame dimensions"""
        # Divide frame into left (exit) and right (entry) zones
        self.counting_line_x = frame_width // 2
        
        # Exit zone (left half)
        self.exit_zone = {
            'x1': 0,
            'y1': 0,
            'x2': self.counting_line_x,
            'y2': frame_height
        }
        
        # Entry zone (right half)
        self.entry_zone = {
            'x1': self.counting_line_x,
            'y1': 0,
            'x2': frame_width,
            'y2': frame_height
        }
        
        logger.info(f"Counting zones set up - Line at X: {self.counting_line_x}")
    
    def detect_people(self, frame):
        """Detect people in the frame using MobileNet SSD"""
        height, width = frame.shape[:2]
        
        # Set up zones if not already done
        if not hasattr(self, 'counting_line_x') or self.counting_line_x is None:
            self.setup_counting_zones(width, height)
        
        # Prepare input blob
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                   0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()
        
        people_detections = []
        
        # Process detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.min_confidence:
                idx = int(detections[0, 0, i, 1])
                if self.CLASSES[idx] == "person":
                    box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                    (startX, startY, endX, endY) = box.astype("int")
                    
                    # Calculate center point
                    centerX = (startX + endX) // 2
                    centerY = (startY + endY) // 2
                    
                    people_detections.append({
                        'confidence': float(confidence),
                        'bbox': [int(startX), int(startY), int(endX - startX), int(endY - startY)],
                        'center': (centerX, centerY)
                    })
        
        return people_detections
    
    def calculate_distance(self, center1, center2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def update_tracking(self, detections):
        """Update person tracking and detect entry/exit events"""
        current_time = time.time()
        
        # Match detections to existing tracks
        matched_tracks = set()
        unmatched_detections = list(range(len(detections)))
        
        for track_id, track_data in list(self.tracking_data.items()):
            best_match = None
            best_distance = float('inf')
            
            for i, detection in enumerate(detections):
                if i in matched_tracks:
                    continue
                
                distance = self.calculate_distance(track_data['last_center'], detection['center'])
                
                if distance < self.min_distance_for_tracking and distance < best_distance:
                    best_distance = distance
                    best_match = i
            
            if best_match is not None:
                # Update existing track
                detection = detections[best_match]
                old_center = track_data['last_center']
                new_center = detection['center']
                
                # Check for line crossing (entry/exit event)
                self.check_line_crossing(track_id, old_center, new_center, detection['confidence'])
                
                # Update track data
                track_data['last_center'] = new_center
                track_data['last_seen'] = current_time
                track_data['confidence'] = detection['confidence']
                
                matched_tracks.add(best_match)
                unmatched_detections.remove(best_match)
        
        # Create new tracks for unmatched detections
        for detection_idx in unmatched_detections:
            detection = detections[detection_idx]
            self.tracking_data[self.next_person_id] = {
                'last_center': detection['center'],
                'last_seen': current_time,
                'confidence': detection['confidence'],
                'created_time': current_time
            }
            self.next_person_id += 1
        
        # Remove old tracks
        tracks_to_remove = []
        for track_id, track_data in self.tracking_data.items():
            if current_time - track_data['last_seen'] > self.max_disappeared:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracking_data[track_id]
    
    def check_line_crossing(self, track_id, old_center, new_center, confidence):
        """Check if a person crossed the counting line and determine direction"""
        old_x = old_center[0]
        new_x = new_center[0]
        line_x = self.counting_line_x
        
        # Check if line was crossed
        if (old_x < line_x < new_x) or (old_x > line_x > new_x):
            direction = None
            
            if old_x < line_x < new_x:
                # Moved from exit zone to entry zone = ENTRY
                direction = 'IN'
            elif old_x > line_x > new_x:
                # Moved from entry zone to exit zone = EXIT
                direction = 'OUT'
            
            if direction:
                # Create count event
                event = {
                    'direction': direction,
                    'people_count': 1,
                    'confidence': confidence,
                    'timestamp': datetime.now(),
                    'track_id': track_id
                }
                
                self.count_events.append(event)
                logger.info(f"Count event: Person {track_id} - {direction} (confidence: {confidence:.2f})")
    
    def get_and_clear_events(self):
        """Get all count events and clear the buffer"""
        events = self.count_events.copy()
        
        # Backup events before clearing (in case database save fails)
        if events:
            self.backup_events.extend(events)
            # Keep backup within limits
            if len(self.backup_events) > self.max_backup_events:
                self.backup_events = self.backup_events[-self.max_backup_events:]
            logger.info(f"📦 Backed up {len(events)} events, total backup: {len(self.backup_events)}")
        
        self.count_events.clear()
        return events
    
    def get_backup_events(self):
        """Get backup events (in case of database recovery)"""
        return self.backup_events.copy()
    
    def clear_backup_events(self):
        """Clear backup events after successful database recovery"""
        cleared_count = len(self.backup_events)
        self.backup_events.clear()
        logger.info(f"🧹 Cleared {cleared_count} backup events")
        return cleared_count
    
    def draw_interface(self, frame):
        """Draw counting interface on frame"""
        height, width = frame.shape[:2]
        
        # Draw counting line
        cv2.line(frame, (self.counting_line_x, 0), (self.counting_line_x, height), (0, 255, 255), 3)
        
        # Draw zone labels
        cv2.putText(frame, "EXIT ZONE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "ENTRY ZONE", (self.counting_line_x + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Draw counting line label
        cv2.putText(frame, "COUNTING LINE", (self.counting_line_x - 100, height//2 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw tracking info
        active_tracks = len(self.tracking_data)
        pending_events = len(self.count_events)
        
        cv2.putText(frame, f"Active Tracks: {active_tracks}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Pending Events: {pending_events}", (10, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def generate_frames(self):
        """Generate frames for video streaming with people counting"""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Failed to read frame from camera")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect people
            people_detections = self.detect_people(frame)
            
            # Update tracking and detect events
            self.update_tracking(people_detections)
            
            # Draw person detection boxes and tracking
            for i, detection in enumerate(people_detections):
                bbox = detection['bbox']
                confidence = detection['confidence']
                center = detection['center']
                
                startX, startY = bbox[0], bbox[1]
                endX, endY = startX + bbox[2], startY + bbox[3]
                
                # Draw bounding box
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                
                # Draw center point
                cv2.circle(frame, center, 5, (255, 0, 0), -1)
                
                # Draw confidence
                label = f"Person: {confidence:.2f}"
                cv2.putText(frame, label, (startX, startY - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw tracking trails
            for track_id, track_data in self.tracking_data.items():
                center = track_data['last_center']
                cv2.circle(frame, center, 8, (255, 255, 0), 2)
                cv2.putText(frame, f"ID:{track_id}", (center[0] + 10, center[1] - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # Draw interface elements
            frame = self.draw_interface(frame)
            
            # People count display
            cv2.putText(frame, f"People detected: {len(people_detections)}", 
                       (10, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'cap'):
            self.cap.release()
        logger.info("PeopleCounterCamera resources cleaned up") 