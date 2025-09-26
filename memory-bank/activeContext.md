# Active Context: Lab MoCap

## Current Work Focus

The Lab MoCap project has successfully transitioned from the volleyball HPE project to a biomechanics laboratory setup. The current focus is on **real-time RTSP stream processing** with integrated human pose estimation for biomechanical analysis.

### Primary Achievement

Successfully created and tested `lab_mocap_stream.py` - a combined script that integrates:
- RTSP camera stream capture (4 laboratory cameras)
- RTMDet human detection
- ByteTrack multi-object tracking  
- RTMPose pose estimation with batch processing
- Real-time display with performance monitoring

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
2. **Preserved All Optimizations**: Maintained batch processing and performance optimizations from volleyball project
3. **Simplified Configuration**: Clean camera selection system (single/all modes only)
4. **Updated Memory Bank**: Removed volleyball context, added biomechanics lab focus
5. **Successful Testing**: Confirmed real-time performance with live RTSP streams

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

## Current Status

### ✅ Completed
- Real-time RTSP stream processing
- Multi-camera support with frame stitching
- Integrated HPE pipeline (detection → tracking → pose estimation)
- Performance optimization and profiling
- Clean configuration system
- Memory bank updates for lab context

### 🔄 In Progress
- Laboratory environment testing and validation
- Performance monitoring under various conditions

### 📋 Future Work
- Joint angle calculation from pose keypoints
- Advanced biomechanical analysis features
- Potential optimization for higher frame rates
- Integration with laboratory data collection systems

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
