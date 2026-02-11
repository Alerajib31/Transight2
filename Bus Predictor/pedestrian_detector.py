"""
Pedestrian Detection System for Bus Stop Monitoring
Uses YOLOv8 computer vision to count waiting passengers
Integrates with transit prediction API
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2
from ultralytics import YOLO

# Import requests with graceful fallback
try:
    import requests
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False
    print("[WARNING] requests library not installed - API integration disabled")
    print("          Install: pip install requests")

# ===========================
# CONFIGURATION
# ===========================

# Default API connection settings
API_SERVER_URL = "http://localhost:8000/api/sensors/update"
DATA_TRANSMISSION_INTERVAL = 5  # seconds between API updates
FRAME_PROCESSING_RATE = 2  # process every Nth frame
DISPLAY_WINDOW_WIDTH = 800  # pixels

# Detection parameters
PERSON_CLASS_ID = 0  # YOLO class ID for "person"
CONFIDENCE_THRESHOLD = 0.45
EXCLUSION_ZONE_RIGHT_PERCENT = 0.40  # Exclude right 40% of frame

# ===========================
# AI MODEL INITIALIZATION
# ===========================

print("[INIT] Loading YOLOv8 pedestrian detection model...")
detection_model = YOLO('yolo.pt')
print("[READY] Model loaded successfully")


# ===========================
# UTILITY FUNCTIONS
# ===========================

def locate_video_file(filename: str) -> Path:
    """
    Search for video file in multiple candidate locations
    Returns: Path object pointing to the video file
    Raises: FileNotFoundError if video not found
    """
    script_directory = Path(__file__).resolve().parent
    
    search_locations = [
        script_directory / "videos" / filename,
        script_directory / filename,
        Path.cwd() / filename,
        Path.cwd() / "videos" / filename
    ]
    
    for location in search_locations:
        if location.exists() and location.is_file():
            print(f"[INFO] Found video: {location}")
            return location
    
    # If not found, raise error with search details
    searched_paths = '\n  - '.join(str(loc) for loc in search_locations)
    raise FileNotFoundError(
        f"Video file '{filename}' not found.\n"
        f"Searched locations:\n  - {searched_paths}"
    )


def check_detection_zone(bbox_coords: Tuple, frame_dimensions: Tuple) -> bool:
    """
    Determine if detection is in valid waiting area
    Excludes right portion of frame where buses park
    
    Args:
        bbox_coords: Bounding box coordinates (x1, y1, x2, y2)
        frame_dimensions: (frame_height, frame_width)
    
    Returns:
        True if detection is in waiting zone, False otherwise
    """
    frame_height, frame_width = frame_dimensions
    x1, y1, x2, y2 = bbox_coords
    
    # Calculate center point of detection
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    
    # Define exclusion boundary (right side where bus stops)
    exclusion_boundary_x = frame_width * (1 - EXCLUSION_ZONE_RIGHT_PERCENT)
    
    # Reject detections beyond exclusion boundary
    if center_x > exclusion_boundary_x:
        return False
    
    # Accept detections in valid zone
    return True


def transmit_to_api(station_identifier: str, passenger_count: int) -> Optional[dict]:
    """
    Send detection data to backend API
    
    Args:
        station_identifier: Unique station ID
        passenger_count: Number of detected passengers
    
    Returns:
        API response dictionary or None if transmission fails
    """
    if not API_AVAILABLE:
        return None
    
    try:
        payload = {
            "station_id": station_identifier,
            "detected_people": passenger_count,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        }
        
        response = requests.post(
            API_SERVER_URL,
            json=payload,
            timeout=8
        )
        
        if response.status_code == 200:
            response_data = response.json()
            print(f"[API] Station: {station_identifier} | "
                  f"Count: {passenger_count} | "
                  f"Delay: {response_data.get('estimated_boarding_delay_minutes', 'N/A')}min")
            return response_data
        else:
            print(f"[ERROR] API returned status {response.status_code}")
            return None
            
    except requests.exceptions.Timeout:
        print("[ERROR] API request timeout")
        return None
    except requests.exceptions.RequestException as req_err:
        print(f"[ERROR] API communication failed: {req_err}")
        return None
    except Exception as general_err:
        print(f"[ERROR] Unexpected error: {general_err}")
        return None


def draw_visualization_overlay(frame, detections, zone_boundary_x):
    """
    Draw detection boxes and exclusion zone on frame
    
    Args:
        frame: OpenCV image array
        detections: List of detection results
        zone_boundary_x: X coordinate of exclusion zone boundary
    
    Returns:
        Annotated frame
    """
    frame_height, frame_width = frame.shape[:2]
    
    # Draw exclusion zone overlay (semi-transparent red)
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (int(zone_boundary_x), 0),
        (frame_width, frame_height),
        (0, 0, 255),
        -1
    )
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    
    # Draw exclusion zone border
    cv2.line(
        frame,
        (int(zone_boundary_x), 0),
        (int(zone_boundary_x), frame_height),
        (0, 0, 255),
        3
    )
    
    # Add zone label
    cv2.putText(
        frame,
        "EXCLUSION ZONE",
        (int(zone_boundary_x) + 10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )
    
    # Draw detection bounding boxes
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection['bbox'])
        color = (0, 255, 0) if detection['in_zone'] else (0, 165, 255)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Add confidence label
        label = f"{detection['confidence']:.2f}"
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )
    
    return frame


# ===========================
# MAIN DETECTION PIPELINE
# ===========================

def run_pedestrian_detection(
    station_id: str,
    video_path: str,
    display_output: bool = True
):
    """
    Main detection loop for pedestrian counting
    
    Args:
        station_id: Transit station identifier
        video_path: Path to video file for processing
        display_output: Show visualization window
    """
    print(f"\n{'='*60}")
    print(f"[START] Pedestrian Detection System")
    print(f"[CONFIG] Station ID: {station_id}")
    print(f"[CONFIG] Video: {video_path}")
    print(f"[CONFIG] API Updates: Every {DATA_TRANSMISSION_INTERVAL}s")
    print(f"{'='*60}\n")
    
    # Locate video file
    try:
        video_file_path = locate_video_file(video_path)
    except FileNotFoundError as e:
        print(f"[FATAL] {e}")
        sys.exit(1)
    
    # Open video stream
    video_capture = cv2.VideoCapture(str(video_file_path))
    
    if not video_capture.isOpened():
        print(f"[FATAL] Cannot open video file: {video_file_path}")
        sys.exit(1)
    
    # Get video properties
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"[VIDEO] Resolution: {frame_width}x{frame_height}")
    print(f"[VIDEO] FPS: {fps:.1f}")
    print(f"[VIDEO] Total Frames: {total_frames}")
    
    # Calculate exclusion zone boundary
    exclusion_boundary = frame_width * (1 - EXCLUSION_ZONE_RIGHT_PERCENT)
    
    # Timing variables
    last_api_update = time.time()
    frame_counter = 0
    current_passenger_count = 0
    
    print("\n[RUNNING] Press 'q' to quit, 's' to skip frame\n")
    
    # Main processing loop
    while True:
        success, frame = video_capture.read()
        
        if not success:
            print("[INFO] End of video - restarting...")
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        frame_counter += 1
        
        # Skip frames for performance
        if frame_counter % FRAME_PROCESSING_RATE != 0:
            continue
        
        # Run YOLO detection
        results = detection_model(frame, verbose=False)
        
        # Process detections
        valid_detections = []
        passenger_count = 0
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Extract detection data
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy()
                
                # Filter for person class with sufficient confidence
                if class_id != PERSON_CLASS_ID or confidence < CONFIDENCE_THRESHOLD:
                    continue
                
                # Check if detection is in valid waiting zone
                in_waiting_zone = check_detection_zone(
                    bbox,
                    (frame_height, frame_width)
                )
                
                if in_waiting_zone:
                    passenger_count += 1
                
                valid_detections.append({
                    'bbox': bbox,
                    'confidence': confidence,
                    'in_zone': in_waiting_zone
                })
        
        current_passenger_count = passenger_count
        
        # Periodic API transmission
        current_time = time.time()
        if current_time - last_api_update >= DATA_TRANSMISSION_INTERVAL:
            transmit_to_api(station_id, passenger_count)
            last_api_update = current_time
        
        # Display visualization
        if display_output:
            # Draw overlays
            annotated_frame = draw_visualization_overlay(
                frame.copy(),
                valid_detections,
                exclusion_boundary
            )
            
            # Add status information
            status_text = [
                f"Station: {station_id}",
                f"Waiting Passengers: {passenger_count}",
                f"Frame: {frame_counter}",
                f"Detections: {len(valid_detections)}"
            ]
            
            y_position = 30
            for text in status_text:
                cv2.putText(
                    annotated_frame,
                    text,
                    (10, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                cv2.putText(
                    annotated_frame,
                    text,
                    (10, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    1
                )
                y_position += 30
            
            # Resize for display
            display_frame = cv2.resize(
                annotated_frame,
                (DISPLAY_WINDOW_WIDTH, int(frame_height * DISPLAY_WINDOW_WIDTH / frame_width))
            )
            
            cv2.imshow('Pedestrian Detection', display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[EXIT] User requested quit")
                break
            elif key == ord('s'):
                print(f"[SKIP] Skipping ahead 30 frames")
                for _ in range(30):
                    video_capture.read()
    
    # Cleanup
    video_capture.release()
    cv2.destroyAllWindows()
    print("[CLEANUP] Resources released")


# ===========================
# COMMAND LINE INTERFACE
# ===========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pedestrian Detection for Transit Stations"
    )
    
    parser.add_argument(
        "--station",
        type=str,
        required=True,
        help="Station identifier (e.g., STATION_TM01)"
    )
    
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Video filename (e.g., 1.mp4)"
    )
    
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable visualization window"
    )
    
    args = parser.parse_args()
    
    run_pedestrian_detection(
        station_id=args.station,
        video_path=args.video,
        display_output=not args.no_display
    )
