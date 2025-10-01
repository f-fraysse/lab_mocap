# Active Context: Lab MoCap

## Current Work Focus

The Lab MoCap project has successfully transitioned from the volleyball HPE project to a biomechanics laboratory setup. A comprehensive **GUI application** has been developed that wraps the existing functionality into an intuitive interface with real-time angle tracking and customizable visualization.

### Primary Achievements

1. **Successfully created and tested `lab_mocap_stream.py`** - Base RTSP streaming with HPE pipeline
2. **Developed `lab_mocap_2Dsquat.py`** - Specialized 2D squat analysis application with:
   - Real-time knee flexion angle calculation and display
   - Automatic squat repetition counting based on hip-knee position analysis
   - Enhanced visual feedback with large status indicators
   - Custom left leg visualization (hip-knee-ankle only)
   - 3-frame squat validation system for accuracy

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

Implemented clean configuration system:
```python
CAMERA_MODE = "single"  # Options: "single", "all"
SELECTED_CAMERA = 1     # Which camera (1-4) for single mode
```

**Single Mode**: Processes one selected camera stream
**All Mode**: Processes all 4 cameras, automatically stitches into 2x2 grid (1920x1080 total)

### RTSP Integration Details

- **Camera URLs**: 4 laboratory cameras with RTSP streams
- **Stream Handling**: Robust error handling for connection issues
- **Frame Stitching**: Automatic 2x2 grid layout for multi-camera mode
- **Resource Management**: Proper camera initialization and cleanup

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
1. **Angle Graph Improvements** (Priority)
   - Better handling of lost IDs / missing keypoints
   - Add left/right side selection for each joint (6 total options per dropdown)
   
2. **Input Dialog Bug Fix**
   - Fix error when changing camera number (currently shows error popup but continues working)
   
3. **Additional Features**
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
