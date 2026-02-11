"""
Multi-Camera Launcher for Pedestrian Detection System
Orchestrates multiple CV detector instances for simultaneous monitoring
"""

import subprocess
import sys
import time
from pathlib import Path

print("=" * 70)
print("  MULTI-STATION PEDESTRIAN DETECTION LAUNCHER")
print("=" * 70)

# ===========================
# CONFIGURATION
# ===========================

# Define station-camera mappings
CAMERA_DEPLOYMENTS = [
    {
        "station_id": "STATION_TM01",
        "video_file": "1.mp4",
        "description": "Temple Meads Station - Platform View"
    },
    {
        "station_id": "STATION_CB03",
        "video_file": "2.mp4",
        "description": "Cabot Circus - Main Stop"
    }
]

# ===========================
# PROCESS MANAGEMENT
# ===========================

def launch_detector_process(station_id: str, video_file: str, description: str):
    """
    Start a pedestrian detector process for a specific station
    """
    print(f"\n[LAUNCHING] {description}")
    print(f"            Station: {station_id}")
    print(f"            Video: {video_file}")
    
    try:
        # Build command
        command = [
            sys.executable,  # Python interpreter
            "pedestrian_detector.py",
            "--station", station_id,
            "--video", video_file
        ]
        
        # Launch subprocess
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print(f"[SUCCESS] Process started with PID: {process.pid}")
        return process
        
    except Exception as error:
        print(f"[ERROR] Failed to launch detector: {error}")
        return None


def main():
    """
    Main launcher function
    """
    print(f"\n[INFO] Configuring {len(CAMERA_DEPLOYMENTS)} camera stations...")
    
    # Verify pedestrian_detector.py exists
    detector_script = Path("pedestrian_detector.py")
    if not detector_script.exists():
        print(f"\n[FATAL] pedestrian_detector.py not found in current directory")
        print(f"[INFO] Please run this script from the project root directory")
        sys.exit(1)
    
    # Launch all detector processes
    active_processes = []
    
    for deployment in CAMERA_DEPLOYMENTS:
        process = launch_detector_process(
            deployment["station_id"],
            deployment["video_file"],
            deployment["description"]
        )
        
        if process:
            active_processes.append({
                "process": process,
                "station": deployment["station_id"],
                "description": deployment["description"]
            })
        
        # Small delay between launches
        time.sleep(1)
    
    print(f"\n{'='*70}")
    print(f"  DEPLOYMENT SUMMARY")
    print(f"{'='*70}")
    print(f"  Active Detectors: {len(active_processes)}")
    
    for idx, proc_info in enumerate(active_processes, 1):
        print(f"  {idx}. {proc_info['description']} (PID: {proc_info['process'].pid})")
    
    print(f"\n[INFO] All detectors running in background")
    print(f"[INFO] Press Ctrl+C to stop all processes")
    print(f"{'='*70}\n")
    
    # Monitor processes
    try:
        while True:
            time.sleep(5)
            
            # Check if any process has terminated
            for proc_info in active_processes:
                if proc_info["process"].poll() is not None:
                    print(f"\n[WARNING] Detector for {proc_info['station']} has stopped")
                    
    except KeyboardInterrupt:
        print(f"\n\n[SHUTDOWN] Stopping all detector processes...")
        
        # Terminate all processes
        for proc_info in active_processes:
            try:
                proc_info["process"].terminate()
                proc_info["process"].wait(timeout=5)
                print(f"[STOPPED] {proc_info['description']}")
            except Exception as error:
                print(f"[ERROR] Failed to stop {proc_info['station']}: {error}")
        
        print(f"\n[EXIT] All processes terminated")


if __name__ == "__main__":
    print(f"\n[START] Multi-Camera Launcher Initializing...")
    time.sleep(1)
    main()
