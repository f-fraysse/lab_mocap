# Lab Treadmill Analysis GUI

A specialized GUI for analyzing sagittal plane hip and knee kinematics during treadmill running.

## Overview

This GUI provides real-time visualization of left hip and knee joint angles, with recording capabilities and automatic gait analysis including footstrike detection and stride normalization.

## Features

- **Real-time Angle Visualization**: Two graphs displaying left hip and knee angles with fixed Y-axis ranges
- **Automatic Person Tracking**: Automatically tracks the person with the lowest track ID (no manual selection needed)
- **Recording Functionality**: Record angle data, ankle coordinates, and video with overlays
- **Automatic Gait Analysis**: Post-recording analysis includes:
  - Footstrike detection using Butterworth filtering and velocity analysis
  - Stride segmentation
  - Time-normalized stride plots (0-100%)
  - Comprehensive output graphs and CSV data

## Installation

Ensure you have all dependencies installed:

```bash
pip install -r requirements.txt
```

Additional dependencies for gait analysis:
- scipy (for signal processing and interpolation)
- matplotlib (for graph generation)

## Usage

### Starting the GUI

```bash
python scripts/lab_treadmill_gui.py
```

The GUI will automatically:
1. Initialize the GoPro camera on index 0 (1920x1080, 30 fps)
2. Start pose estimation and tracking
3. Display real-time hip and knee angles for the detected person

### Recording a Session

1. **Enter Recording Name**: Type a name in the text box (default: "test_00")
2. **Start Recording**: Click the red dot button to start recording
   - Button turns solid red during recording
   - Recording name input is disabled during recording
3. **Stop Recording**: Click the button again to stop
   - Analysis begins automatically in the background
   - GUI remains responsive during analysis

### Output Files

All outputs are saved to `/output/{recording_name}/`:

1. **CSV File**: `{recording_name}.csv`
   - Columns: frame, time_s, hip_angle, knee_angle, ankle_x, ankle_y
   
2. **Video File**: `{recording_name}.mp4`
   - Recorded video with pose overlays (keypoints and skeleton)
   
3. **Analysis Graphs** (if footstrikes detected):
   - `{recording_name}_hip_angles.png`: Hip angles vs frame with footstrike markers
   - `{recording_name}_knee_angles.png`: Knee angles vs frame with footstrike markers
   - `{recording_name}_hip_normalized.png`: All strides normalized to 0-100%
   - `{recording_name}_knee_normalized.png`: All strides normalized to 0-100%

## Graph Specifications

### Real-time Graphs
- **Left Hip**: Y-axis range -20° to +100°
- **Left Knee**: Y-axis range -10° to +160°
- Both graphs show 5 seconds of rolling history

### Angle Definitions
- **Hip Angle**: Shoulder - Hip - Knee
- **Knee Angle**: Hip - Knee - Ankle
- Angles calculated using COCO17 keypoint format
- 0° = straight, 180° = fully bent

## Gait Analysis Details

### Footstrike Detection Algorithm

1. **Filtering**: Butterworth low-pass filter (1st order, 10 Hz cutoff) applied to ankle X coordinate
2. **Peak Detection**: Local minima in horizontal ankle position (X coordinate) with:
   - Minimum prominence: 5 pixels
   - Minimum spacing: 0.25 seconds (7.5 frames at 30 fps)

### Stride Segmentation

- Each stride defined from footstrike n to footstrike n+1
- Data before first footstrike and after last footstrike are discarded
- Minimum 2 footstrikes required for stride analysis

### Normalization

- Each stride normalized to 0-100% using cubic interpolation
- 101 points per normalized stride (including both endpoints)
- All strides superimposed on normalized plots
- Mean stride shown in black

## Technical Details

### Camera Configuration
- **Device**: GoPro on index 0
- **Resolution**: 1920x1080
- **Frame Rate**: 30 fps
- **Codec**: MJPG

### Models Used
- **Detection**: RTMDet-m (640x640)
- **Pose Estimation**: RTMPose-m (192x256)
- **Tracking**: ByteTrack

### Processing Pipeline
1. Frame capture from GoPro
2. Person detection (RTMDet)
3. Multi-person tracking (ByteTrack)
4. Pose estimation (RTMPose)
5. Angle calculation for lowest track ID
6. Display update and optional recording

## Troubleshooting

### No Person Detected
- Ensure person is fully visible in frame
- Check lighting conditions
- Verify GoPro is properly connected

### Footstrikes Not Detected
- Ensure recording is at least 1 second long
- Check that person is running (not walking or standing)
- Verify ankle keypoint is consistently detected
- May need to adjust prominence parameter in `gait_analysis.py`

### Video Not Saving
- Check disk space in output directory
- Verify write permissions for output folder
- Ensure recording captured frames (check console output)

## File Structure

```
scripts/
├── lab_treadmill_gui.py              # Entry point
└── gui/
    ├── treadmill_main_window.py      # Main window with 2 graphs
    ├── treadmill_video_thread.py     # Video processing thread
    ├── treadmill_angle_graph.py      # Angle graph widget
    ├── recording_widget.py           # Recording controls
    ├── gait_analysis.py              # Footstrike detection & analysis
    ├── angle_calculator.py           # Joint angle calculations (shared)
    └── config.py                     # Configuration (shared)
```

## Comparison with Main GUI

| Feature | Main GUI | Treadmill GUI |
|---------|----------|---------------|
| Camera Input | Multiple options | GoPro only (index 0) |
| Angle Graphs | 3 configurable | 2 fixed (hip & knee) |
| Track Selection | Manual dropdown | Automatic (lowest ID) |
| Recording | No | Yes |
| Gait Analysis | No | Yes (automatic) |
| Use Case | General motion capture | Treadmill running analysis |

## Future Enhancements

Potential improvements:
- Adjustable footstrike detection parameters in GUI
- Real-time stride count display
- Configurable Y-axis ranges
- Export to other formats (Excel, MATLAB)
- Bilateral analysis (both legs)
- Additional gait metrics (stride length, cadence, etc.)

## Support

For issues or questions, refer to the main project documentation or contact the development team.
