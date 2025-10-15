# Project Progress: Lab MoCap

## What Works

### ✅ Real-time RTSP Processing Pipeline

1. **RTSP Stream Capture**
   - Successfully connects to and processes 4 laboratory RTSP camera streams
   - Handles individual camera failures gracefully
   - Efficient frame capture with error recovery (~12.4ms average including latency)

2. **Camera Configuration System**
   - Clean configuration-driven camera management with three input modes
   - **Single IP Camera mode**: Process one selected RTSP camera (1-4)
   - **All IP Cameras mode**: Process all 4 RTSP cameras with automatic 2x2 stitching
   - **Single GoPro Camera mode**: Process GoPro Hero 12 via USB (default)
   - Dynamic resource allocation based on selected mode
   - Seamless switching between camera sources via GUI

3. **Human Detection**
   - RTMDet model successfully detects humans in laboratory environment
   - Medium-sized model (RTMDet-m) provides good accuracy for biomechanics applications
   - **Current performance: ~21.8ms per frame** (optimized preprocessing)

4. **Subject Tracking**
   - ByteTrack algorithm successfully maintains subject IDs
   - Handles occlusions and multiple subjects effectively
   - Very efficient performance (~0.1ms per frame)
   - Consistent ID assignment throughout sessions

5. **Pose Estimation**
   - RTMPose model successfully estimates human poses in real-time
   - Medium-sized model (RTMPose-m) provides accurate keypoint detection
   - **Current performance: ~1.9ms per frame** (batch processing optimization)
   - Suitable for biomechanical analysis applications

6. **Real-time Display and Monitoring**
   - Live video display with pose overlays and bounding boxes
   - Real-time performance metrics displayed on stream
   - Comprehensive timing information for all pipeline components

### ✅ Specialized Biomechanical Analysis Applications

7. **2D Squat Analysis System (`lab_mocap_2Dsquat.py`)**
   - **Knee Flexion Angle Calculation**: Real-time computation using hip-knee-ankle keypoints
     - Custom formula: `180° - arccos(dot_product)` where 0° = straight leg
     - Time history graph in bottom right corner (last 5 seconds)
     - White background, black axes, thick blue plot line
     - Y-axis: -5° to 180° with gridlines at 0°, 45°, 90°, 135°, 180°
     - Current angle value displayed in top right (yellow text, black background)
     - Only calculated when all three keypoints have confidence > 0.5
     - Uses `collections.deque` with 1000-entry capacity for rolling window
   
   - **Automatic Squat Repetition Counting**: Biomechanically-based rep detection
     - Uses hip-knee vertical position comparison for squat validation
     - 3-frame consecutive validation system prevents false positives
     - State machine: STANDING → SQUAT_VALIDATED → REP_COMPLETED → STANDING
     - Large green "SQUAT" indicator (270x100 pixels) during validated squats
     - Enhanced rep counter with 50% larger text display

8. **Custom Skeleton Visualization System**
   - **`draw_skeleton_custom()` Function**: Selective keypoint and connection rendering
     - Configurable keypoint selection (supports left leg only visualization)
     - Customizable colors, line thickness, and keypoint sizes
     - Efficient rendering with enhanced visibility options
   - **Left Leg Focus**: Hip-knee-ankle visualization for squat analysis
     - Larger keypoints (radius=5) and thicker lines (width=3) for better visibility
     - Green color scheme for left leg components

### ✅ Performance Optimization

1. **Batch Processing Implementation**
   - RTMPose modified for batch inference of all detected subjects
   - Significant performance improvement over individual processing
   - Maintains accuracy while improving speed

2. **Comprehensive Profiling System**
   - Real-time performance monitoring with on-screen display
   - Detailed CSV logging of all timing components
   - Statistical analysis (min/max/average/median) for performance assessment

3. **GPU Acceleration**
   - ONNX Runtime with CUDA backend for optimal performance
   - Efficient memory management and resource utilization

### ✅ GUI Application

9. **Complete PyQt5 GUI Application (`lab_mocap_gui.py`)**
   - **Input Configuration Dialog**: Three camera mode selection
     - Single IP Camera with camera number selection (1-4)
     - All IP Cameras with angle computation camera selection
     - Single GoPro Camera (no additional configuration needed)
   - **Display Options Dialog**: Comprehensive visualization controls
     - Toggle bounding boxes, track IDs, keypoints, skeleton
     - Grouped skeleton controls (legs, arms, torso, head)
     - Color picker and thickness slider for each group
   - **Angle Tracking System**: Real-time joint angle visualization
     - Track ID selector (dynamically populated)
     - 3 independent matplotlib graph panels
     - Joint selection: Hip, Knee, Elbow (left side)
     - 5-second rolling window display
   - **Threaded Processing**: Prevents GUI freezing during computation
   - **Modular Architecture**: Separate components for maintainability

10. **GoPro Hero 12 Integration**
    - **USB Connection**: MSMF backend for Windows compatibility
    - **Configuration**: 1920x1080, 30fps, MJPG format
    - **Warmup System**: 20 frames to skip splash/black frames
    - **GoProCam Class**: Reusable camera interface in `test_gopro_stream.py`
    - **Seamless Integration**: Works with existing pose estimation pipeline
    - **Default Input**: GUI launches with GoPro as default camera source

### ✅ Development Environment

1. **Laboratory Integration**
   - Configured for biomechanics laboratory camera setup
   - RTSP stream handling optimized for laboratory network
   - GoPro USB camera support for portable/field use
   - Flexible configuration for different research scenarios

2. **Project Structure**
   - Clean separation between old volleyball code and new lab mocap system
   - Centralized configuration management
   - Proper resource cleanup and error handling
   - Modular GUI architecture with separate dialog components

## What's Left to Build

### 🔄 Current Focus: Laboratory Validation and Enhancement

1. **Laboratory Environment Testing** 🔄
   - Validate performance with actual biomechanical movements
   - Test system stability under various laboratory conditions
   - Optimize camera positioning for optimal coverage

2. **Joint Angle Calculation** 📋
   - Implement biomechanical analysis from pose keypoints
   - Calculate lower limb joint angles from RTMPose output
   - Integrate with existing pose estimation pipeline

3. **Advanced Data Analysis** 📋
   - Movement pattern analysis capabilities
   - Comparative study support
   - Longitudinal tracking and analysis

### 🔜 Future Enhancements

1. **Laboratory Workflow Integration**
   - Integration with existing laboratory data collection systems
   - Export formats compatible with biomechanical analysis tools
   - Automated data collection and processing workflows

2. **Advanced Analytics**
   - Movement quality assessment
   - Comparative analysis between subjects
   - Temporal analysis of movement patterns

3. **System Optimization**
   - Further performance improvements if needed
   - Enhanced error handling and recovery
   - Multi-threading for camera stream handling

## Current Status

### 🟢 Working Features

- ✅ RTSP stream capture and processing
- ✅ Multi-camera configuration system
- ✅ Real-time human detection with RTMDet
- ✅ Subject tracking with ByteTrack
- ✅ Pose estimation with RTMPose (batch processing)
- ✅ Real-time display with performance monitoring
- ✅ Optional HDF5 data logging
- ✅ Comprehensive performance profiling

### 🟡 In Progress

- 🔄 Laboratory environment validation
- 🔄 Performance monitoring under various conditions
- 🔄 System stability assessment

### 🟢 Ready for Enhancement

- 📋 Joint angle calculation implementation
- 📋 Advanced biomechanical analysis features
- 📋 Laboratory workflow integration

## Performance Metrics

Current performance with live RTSP streams:

| Component       | Avg Time (ms) | Performance Status | Notes                                    |
|-----------------|---------------|-------------------|------------------------------------------|
| **Total**       | **38.0**      | **🟢 Excellent**  | **~26 FPS - Real-time performance**     |
| Stream Capture  | 12.4          | 🟢 Good           | Includes RTSP network latency           |
| Detection       | 21.8          | 🟢 Good           | Primary processing component            |
| Tracking        | 0.1           | 🟢 Optimal        | Very efficient ByteTrack               |
| Pose Estimation | 1.9           | 🟢 Excellent      | Highly optimized batch processing      |
| Display/Logging | 1.2           | 🟢 Optimal        | Minimal overhead                       |

### System Requirements Met

- ✅ Real-time processing (>20 FPS target exceeded)
- ✅ Multi-camera support with flexible configuration
- ✅ Robust RTSP stream handling
- ✅ Laboratory environment compatibility
- ✅ Performance monitoring and profiling

## Project Transition Status

### ✅ Completed Transition

- ✅ Successfully migrated from volleyball HPE to lab mocap system
- ✅ Integrated RTSP streaming with optimized HPE pipeline
- ✅ Updated all memory bank documentation for lab context
- ✅ Created primary processing script (`lab_mocap_stream.py`)
- ✅ Validated real-time performance with live streams
- ✅ Developed complete GUI application with angle tracking
- ✅ Integrated GoPro Hero 12 camera support (October 15, 2025)

### 🔄 Cleanup in Progress

- 🔄 Memory bank consistency verification
- 🔄 Removal of unused volleyball-specific scripts
- 🔄 Final project organization

## Next Milestone

**Laboratory Validation and Joint Angle Implementation**

**Current Achievement**: Successfully created and tested real-time lab mocap system ✅

**Next Focus**: 
1. Validate system performance in actual laboratory biomechanical studies
2. Implement joint angle calculation from pose keypoints
3. Develop advanced biomechanical analysis features

**Success Criteria**:
- Reliable operation during actual laboratory sessions
- Accurate joint angle calculations for biomechanical analysis
- Integration with laboratory research workflows
- Demonstration of research value for biomechanics applications

The Lab MoCap system is now fully operational and ready for biomechanical research applications, with excellent real-time performance and comprehensive monitoring capabilities.
