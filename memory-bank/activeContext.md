# Active Context: Lab MoCap

## Current Work Focus

The Lab MoCap project has successfully transitioned from the volleyball HPE project to a biomechanics laboratory setup. Two comprehensive **GUI applications** have been developed:

1. **General Lab MoCap GUI** (`lab_mocap_gui.py`) - Multi-purpose pose analysis with configurable angle tracking
2. **Treadmill Analysis GUI** (`lab_treadmill_gui.py`) - Specialized for treadmill running gait analysis

**Latest Update (October 16, 2025)**: Successfully created **Treadmill Analysis GUI** with automatic gait analysis, footstrike detection, and stride normalization. Features signed angle calculations, recording functionality, and comprehensive post-recording analysis with 5 output graphs.

### Primary Achievements

1. **Successfully created and tested `lab_mocap_stream.py`** - Base RTSP streaming with HPE pipeline
2. **Developed `lab_mocap_2Dsquat.py`** - Specialized 2D squat analysis application with:
   - Real-time knee flexion angle calculation and display
   - Automatic squat repetition counting based on hip-knee position analysis
   - Enhanced visual feedback with large status indicators
   - Custom left leg visualization (hip-knee-ankle only)
   - 3-frame squat validation system for accuracy
3. **Created Treadmill Analysis GUI (`lab_treadmill_gui.py`)** - Complete gait analysis system:
   - Real-time hip and knee angle tracking with signed angles
   - Recording functionality with automatic post-recording analysis
   - Footstrike detection using ankle X position minima
   - Stride segmentation and normalization (0-100%)
   - Comprehensive output: 5 PNG graphs, CSV data, and video with overlays

### Current Performance Metrics

Based on initial testing with live RTSP streams:

| Component       | Avg Time (ms) | Performance Notes                           |
|-----------------|---------------|---------------------------------------------|
| **Total**       | **38.0**      | **~26 FPS** - Excellent real-time performance |
| Capture         | 12.4          | Includes RTSP stream latency                |
| Detection       | 21.8          | Primary processing component                |
| Tracking        | 0.1           | Very efficient (ByteTrack)                 |
| Pose Estimation | 1.9           | Highly optimized with batch processing     |
| Display/Logging | 1.2           | Minimal overhead                           |

### Camera Configuration System

Implemented clean configuration system with three input modes:
```python
CAMERA_MODE = "single_gopro"  # Options: "single_ip", "all_ip", "single_gopro"
SELECTED_CAMERA = 1           # Which IP camera (1-4) for single_ip mode
```

**Camera Input Modes:**
1. **Single IP Camera**: Processes one selected RTSP camera stream (1-4)
2. **All IP Cameras**: Processes all 4 RTSP cameras, automatically stitches into 2x2 grid (1920x1080 total)
3. **Single GoPro Camera** (Default): Processes GoPro Hero 12 via USB connection

### Camera Integration Details

**RTSP IP Cameras:**
- **Camera URLs**: 4 laboratory cameras with RTSP streams
- **Stream Handling**: Robust error handling for connection issues
- **Frame Stitching**: Automatic 2x2 grid layout for multi-camera mode
- **Resource Management**: Proper camera initialization and cleanup

**GoPro Hero 12 Camera:**
- **Connection**: USB via MSMF (Microsoft Media Foundation) backend
- **Resolution**: 1920x1080
- **Frame Rate**: 30 fps
- **Format**: MJPG (Motion JPEG)
- **Warmup**: 20 frames to skip splash/black frames
- **Camera Index**: 0 (configurable)
- **Implementation**: `GoProCam` class in `scripts/gui/test_gopro_stream.py`

## Recent Changes

1. **Created `lab_mocap_stream.py`**: Complete integration of RTSP streaming with HPE pipeline
2. **Developed `lab_mocap_2Dsquat.py`**: Specialized squat analysis application with dual functionality
3. **Preserved All Optimizations**: Maintained batch processing and performance optimizations from volleyball project
4. **Implemented Biomechanical Analysis**: Real-time knee angle calculation and squat repetition counting
5. **Enhanced Visual Feedback**: Large status indicators and custom skeleton visualization
6. **Added Custom Drawing System**: `draw_skeleton_custom()` function for selective keypoint display
7. **Added Knee Angle Time History Graph**: Real-time graph visualization in bottom right corner
   - Displays last 5 seconds of knee angle data
   - White background with black axes and thick blue plot line
   - Y-axis: -5° to 180° with gridlines
   - Current angle value displayed in top right of graph
   - Uses `collections.deque` for efficient rolling window storage
8. **Successful Testing**: Confirmed real-time performance with specialized analysis features
9. **Developed GUI Application (`lab_mocap_gui.py`)**: Complete PyQt5-based interface
   - Menu bar with Input and Display configuration dialogs
   - Main display area with real-time video and configurable overlays
   - Left sidebar with 3 angle graph panels (matplotlib integration)
   - Track ID selector for multi-person angle tracking
   - Threaded video processing to prevent GUI freezing
   - Modular architecture with separate components for dialogs, graphs, and processing
   - Fixed PyQt5 signal type compatibility issues (numpy array handling)
10. **Integrated GoPro Hero 12 Camera Support** (Latest - October 15, 2025):
    - Added third camera input mode: "Single GoPro Camera"
    - Renamed existing modes: "Single IP Camera" and "All IP Cameras"
    - Created `GoProCam` class in `scripts/gui/test_gopro_stream.py`
    - Modified `config.py` to support three camera modes with GoPro parameters
    - Updated `input_dialog.py` with GoPro radio button and mode selection
    - Enhanced `video_thread.py` to handle GoPro initialization and capture
    - GoPro configuration: Index 0, 1920x1080, 30fps, MJPG format, 20 warmup frames
    - GUI now defaults to GoPro camera on launch
    - Seamless integration with existing pose estimation pipeline

## Treadmill Analysis GUI Features (`lab_treadmill_gui.py`)

### Overview
Specialized GUI for analyzing sagittal plane hip and knee kinematics during treadmill running with automatic gait analysis.

### Real-time Features
- **2 Angle Graphs**: Left hip (-30° to +100°) and left knee (-10° to +160°)
- **Automatic Person Tracking**: Always tracks lowest detected track ID
- **Signed Angle Calculations**: 
  - Hip: Positive = extension, Negative = flexion
  - Knee: Positive = flexion, Negative = extension
  - Uses cross product to determine sign
- **Fixed GoPro Input**: Index 0, no camera selection needed

### Recording Functionality
- **Toggle Button**: Red dot icon for start/stop recording
- **Name Input**: Default "test_00", customizable
- **Data Collection**: Hip angles, knee angles, ankle coordinates, video frames
- **Automatic Analysis**: Triggered on recording stop

### Post-Recording Analysis
1. **Footstrike Detection**:
   - Butterworth low-pass filter (1st order, 10 Hz) on ankle X coordinate
   - Local minima detection with 0.25s minimum spacing
   - Prominence-based peak detection

2. **Stride Segmentation**:
   - Separates data into individual strides (footstrike n to n+1)
   - Discards data before first and after last footstrike

3. **Time Normalization**:
   - Each stride normalized to 0-100% using cubic interpolation
   - 101 points per normalized stride

### Output Files (saved to `/output/{recording_name}/`)
1. **CSV File**: Frame, time, hip angle, knee angle, ankle X, ankle Y
2. **Video File**: MP4 with pose overlays
3. **5 PNG Graphs**:
   - Ankle X position with footstrike markers
   - Hip angles with footstrike markers
   - Knee angles with footstrike markers
   - Hip normalized strides (0-100%) with ±1 SD shading and frame insets
   - Knee normalized strides (0-100%) with ±1 SD shading and frame insets

### Frame Insets Feature (October 16, 2025)
- **Third Stride Visualization**: Extracts 5 frames from the third detected stride
- **Key Time Points**: 0%, 25%, 50%, 75%, 100% of stride cycle
- **Visual Integration**: Frames displayed at top of normalized stride graphs
- **Reference Lines**: Black dotted vertical lines at each percentage point
- **Layout**: 14"x9" figure with 25% top panel for frames, 75% bottom for graph
- **Fallback**: Gracefully handles cases with <3 strides (no frame insets shown)

### Technical Implementation
- **Signed Angles**: Cross product method in `angle_calculator.py`
- **Gait Analysis**: Complete pipeline in `gait_analysis.py`
- **Recording Widget**: Custom PyQt5 widget with toggle button
- **Threaded Analysis**: Post-recording analysis runs in background thread

## Squat Analysis Features (`lab_mocap_2Dsquat.py`)

### Dual Analysis System
- **Knee Flexion Angle**: Real-time calculation using hip-knee-ankle keypoints
  - Formula: `180° - arccos(dot_product)` where 0° = straight leg
  - Displayed in time history graph (bottom right corner) showing last 5 seconds
  - Graph: white background, black axes, thick blue line, Y-axis: -5° to 180°
  - Current angle value shown in top right of graph (yellow text, black background)
  - Only calculated when all three keypoints have confidence > 0.5
  - Uses `deque` with 1000-entry capacity for efficient rolling window

- **Squat Repetition Counter**: Automatic counting based on biomechanical criteria
  - Uses hip-knee vertical position comparison
  - 3-frame consecutive validation prevents false positives
  - Large green "SQUAT" indicator (270x100 pixels) when validated
  - Enhanced rep counter with 50% larger text

### Technical Implementation Details
- **Custom Skeleton Drawing**: `draw_skeleton_custom()` function with selective rendering
- **State Machine Logic**: STANDING → SQUAT_VALIDATED → REP_COMPLETED → STANDING
- **Error Handling**: Graceful handling of missing keypoints and calculation errors
- **Performance**: Maintains ~26 FPS with additional analysis overhead
- **Configuration**: Uses camera 4 in single mode for focused analysis

## Technical Implementation

### Pipeline Architecture
```
RTSP Capture → Camera Selection/Stitching → Detection → Tracking → Pose Estimation → Display/Logging
```

### Key Features
- **Modular Camera Management**: Dynamic initialization based on configuration
- **Performance Profiling**: Comprehensive timing analysis with CSV logging
- **Visual Overlays**: Real-time performance metrics displayed on stream
- **Optional Data Logging**: HDF5 format for offline analysis (disabled by default)
- **Error Handling**: Robust stream failure recovery

### Models Used
- **Detection**: RTMDet-m (640x640 input)
- **Pose Estimation**: RTMPose-m (192x256 input)
- **Tracking**: ByteTrack with optimized parameters
- **Backend**: ONNX Runtime with CUDA acceleration

## GUI Application Architecture

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

### Key Components

**Input Dialog:**
- Camera mode selection (single/all)
- Camera number selection
- Angle computation camera (for all mode)
- Apply button restarts video processing

**Display Dialog:**
- Bounding boxes toggle
- Track IDs toggle
- Keypoints toggle with size slider
- Skeleton connections with grouped controls:
  - Left/Right Leg, Left/Right Arm, Torso, Head
  - Color picker for each group
  - Thickness slider for each group
- Reset to defaults button

**Angle Graphs:**
- Track ID selector (dynamically populated)
- 3 independent graph panels
- Joint selection: Hip, Knee, Elbow (left side only currently)
- 5-second rolling window display
- Matplotlib integration with real-time updates

### Technical Notes

**PyQt5 Signal Handling:**
- Signals require `object` type for numpy arrays
- Keypoints/scores converted to lists before emission
- Converted back to numpy arrays for processing
- Critical for thread-safe communication

**Threading:**
- Video processing runs in separate QThread
- Prevents GUI freezing during heavy computation
- Proper cleanup on window close

## Current Status

### ✅ Completed
- Real-time RTSP stream processing
- Multi-camera support with frame stitching
- Integrated HPE pipeline (detection → tracking → pose estimation)
- Performance optimization and profiling
- Clean configuration system
- **GUI Application with full functionality**
- **Real-time angle tracking and visualization**
- **Configurable display options**
- Memory bank updates for lab context

### 🔄 In Progress
- GUI testing and refinement
- Bug fixes for edge cases

### 📋 Future Work - GUI Enhancements
1. **Multi-GoPro Camera Support** (Next Priority)
   - Extend single GoPro to support up to 4 GoPro cameras
   - Threading pattern for parallel capture (reference implementation provided):
   ```python
   import threading
   
   def show(idx, title):
       cam = GoProCam(index=idx, size=(1920,1080), fps=30, fourcc='MJPG', warmup_frames=45)
       cam.open()
       while True:
           ok, frame = cam.read()
           if not ok: break
           cv2.imshow(title, frame)
           if cv2.waitKey(1) & 0xFF == ord('q'):
               break
       cam.close()
   
   for i in range(4):
       threading.Thread(target=show, args=(i, f"GoPro {i}"), daemon=True).start()
   
   cv2.waitKey(0)
   cv2.destroyAllWindows()
   ```
   - Implementation approach:
     * Each GoPro runs in separate daemon thread for parallel capture
     * Cameras identified by index 0-3
     * Synchronize frame collection before processing
     * Stitch into 2x2 grid (similar to "All IP Cameras" mode)
     * Feed stitched frame into existing pose estimation pipeline
   - New GUI mode: "All GoPro Cameras"

2. **Angle Graph Improvements**
   - Better handling of lost IDs / missing keypoints
   - Add left/right side selection for each joint (6 total options per dropdown)
   
3. **Input Dialog Bug Fix**
   - Fix error when changing camera number (currently shows error popup but continues working)
   
4. **Additional Features**
   - Export angle data to CSV
   - Recording functionality
   - Additional joint angle calculations
   - Multi-person angle tracking
   - Customizable graph time windows
   - Performance metrics overlay
   - Configurable camera URLs (not hardcoded)

## Key Considerations

1. **RTSP Stream Stability**: Monitor for connection drops and implement reconnection logic if needed
2. **Performance Scaling**: Current ~26 FPS is excellent for real-time analysis
3. **Camera Positioning**: Optimal placement for biomechanical analysis coverage
4. **Data Storage**: HDF5 logging available but disabled by default for performance
5. **Future Integration**: Design considerations for joint angle analysis implementation

## Questions and Next Steps

1. **Laboratory Validation**: How does the system perform with actual biomechanical movements?
2. **Camera Optimization**: Are there preferred camera positions for specific analyses?
3. **Data Requirements**: What specific biomechanical metrics need to be extracted?
4. **Integration Needs**: How will this connect with existing laboratory workflows?

The system is now ready for biomechanical research applications with excellent real-time performance and flexible camera configuration options.
