# Lab MoCap GUI Application

A graphical user interface for real-time human pose estimation and biomechanical analysis in the laboratory setting.

## Overview

The Lab MoCap GUI wraps the existing pose estimation pipeline into an intuitive interface with:
- Real-time video display with configurable overlays
- Multi-camera support (single or 4-camera stitched view)
- Live joint angle tracking and visualization
- Customizable display options for bounding boxes, keypoints, and skeleton connections

## Installation

1. Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

The GUI requires PyQt5, which has been added to the requirements.

## Running the Application

From the project root directory:
```bash
python scripts/lab_mocap_gui.py
```

Or from the scripts directory:
```bash
cd scripts
python lab_mocap_gui.py
```

## GUI Components

### Menu Bar

#### Input Menu
- **Configure Input**: Opens dialog to configure camera settings
  - **Camera Mode**: Choose between "Single Camera" or "All Cameras"
  - **Single Camera**: Select which camera (1-4) to use
  - **All Cameras**: Select which camera to use for angle computation
  - Changes require clicking "Apply" and will restart the video stream

#### Display Menu
- **Display Options**: Opens dialog to customize visual overlays
  - **Bounding Boxes**: Toggle display of detection bounding boxes
  - **Track IDs**: Toggle display of track ID labels
  - **Keypoints**: Toggle keypoint display and adjust size (1-10)
  - **Skeleton Connections**: Configure skeleton line display
    - Master toggle for all connections
    - Individual group controls (Left Leg, Right Leg, Left Arm, Right Arm, Torso, Head)
    - Color picker for each group
    - Thickness slider for each group (1-10)
  - **Reset to Defaults**: Restore default RTMPose visualization settings

### Main Display Area

The central area shows the live video feed with overlays:
- Bounding boxes around detected people (if enabled)
- Track IDs for each person (if enabled)
- Keypoints at joint locations (if enabled)
- Skeleton connections between keypoints (if enabled)

All overlays are configurable through the Display Options dialog.

### Left Sidebar - Angle Graphs

The left sidebar contains three angle graph panels for biomechanical analysis:

1. **Track ID Selector** (top)
   - Dropdown to select which person to track
   - Dynamically populated with currently detected track IDs
   - Click dropdown to refresh available IDs
   - Select "None" to disable angle tracking

2. **Three Graph Panels**
   - Each panel has:
     - Joint selector dropdown (Hip, Knee, or Elbow)
     - Real-time angle graph showing last 5 seconds
   - Default configuration:
     - Graph 1: Hip angle
     - Graph 2: Knee angle
     - Graph 3: Elbow angle

#### Joint Angle Definitions

All angles use left-side keypoints by default:

- **Hip Angle**: Shoulder → Hip → Knee (keypoints 5-11-13)
- **Knee Angle**: Hip → Knee → Ankle (keypoints 11-13-15)
- **Elbow Angle**: Shoulder → Elbow → Wrist (keypoints 5-7-9)

Angles are calculated as 0° = straight, 180° = fully bent.

#### Graph Behavior

- Graphs update in real-time when a track ID is selected
- If selected person is lost, graphs show empty data
- If required keypoints are not detected (confidence < 0.5), no angle is calculated
- Selecting a new track ID resets the time window
- Graphs maintain a 5-second rolling window

## Camera Configuration

The application uses hardcoded RTSP camera URLs:
- Camera 1: `rtsp://ubnt:ubnt@192.168.5.41:554/s0`
- Camera 2: `rtsp://ubnt:ubnt@192.168.5.45:554/s0`
- Camera 3: `rtsp://ubnt:ubnt@192.168.5.42:554/s0`
- Camera 4: `rtsp://ubnt:ubnt@192.168.5.48:554/s0`

### Camera Modes

**Single Camera Mode:**
- Displays feed from one selected camera
- Full resolution processing
- Angle computation uses this camera's data

**All Cameras Mode:**
- Displays all 4 cameras in a 2×2 grid (1920×1080 total)
- Each camera resized to 960×540
- Layout: Camera 1 (top-left), Camera 2 (top-right), Camera 3 (bottom-left), Camera 4 (bottom-right)
- Angle computation uses data from selected camera

## Technical Details

### Architecture

```
Main Window (GUI Thread)
├── Menu Bar (Input, Display)
├── Left Sidebar (Angle Graphs)
└── Main Display (Video Feed)

Video Processing Thread (Separate Thread)
├── Camera Capture
├── Detection (RTMDet-m)
├── Tracking (ByteTrack)
├── Pose Estimation (RTMPose-m)
└── Signal Emission → Main Window
```

### Performance

- Target: ~26 FPS real-time processing
- GPU acceleration via CUDA (ONNX Runtime)
- Threaded video processing prevents GUI freezing
- Efficient matplotlib integration for graphs

### File Structure

```
scripts/
├── lab_mocap_gui.py              # Main entry point
└── gui/
    ├── __init__.py
    ├── config.py                 # Configuration management
    ├── main_window.py            # Main window class
    ├── input_dialog.py           # Input configuration dialog
    ├── display_dialog.py         # Display options dialog
    ├── video_thread.py           # Video processing thread
    ├── angle_graph_widget.py     # Matplotlib graph widget
    └── angle_calculator.py       # Joint angle calculations
```

## Troubleshooting

### Application won't start
- Ensure PyQt5 is installed: `pip install PyQt5`
- Check that all dependencies are installed
- Verify CUDA is available for GPU acceleration

### No video feed
- Check camera RTSP URLs are accessible
- Verify network connectivity to cameras
- Try switching to a different camera in Input Configuration

### Graphs not updating
- Ensure a track ID is selected in the dropdown
- Check that the selected person is visible and tracked
- Verify keypoints are being detected (visible on video feed)

### Poor performance
- Close other GPU-intensive applications
- Try single camera mode instead of all cameras
- Check GPU memory usage

## Future Enhancements

Potential improvements for future versions:
- Configurable camera URLs (not hardcoded)
- Left/right side selection for angle calculations
- Export angle data to CSV
- Recording functionality
- Additional joint angle calculations
- Multi-person angle tracking
- Customizable graph time windows
- Performance metrics overlay

## Credits

Developed for the UniSA Biomechanics Laboratory
Based on the Lab MoCap pose estimation pipeline
