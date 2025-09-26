# System Patterns: Lab MoCap

## Architecture Overview

The Lab MoCap system follows a modular real-time processing pipeline designed for biomechanics laboratory environments. The architecture maintains the optimized patterns from the volleyball project while adapting to RTSP stream processing.

## Core Pipeline Architecture

```
RTSP Stream Capture → Camera Management → Detection → Tracking → Pose Estimation → Display/Logging
```

### Processing Flow
1. **Stream Capture**: Multi-camera RTSP stream handling with error recovery
2. **Camera Management**: Dynamic initialization based on configuration mode
3. **Frame Processing**: Single camera or multi-camera stitching
4. **Detection**: RTMDet human detection with GPU acceleration
5. **Tracking**: ByteTrack multi-object tracking with ID persistence
6. **Pose Estimation**: RTMPose batch processing for efficiency
7. **Output**: Real-time display with performance overlays

## Key Design Patterns

### 1. Configuration-Driven Camera Management

**Pattern**: Centralized configuration with dynamic resource allocation
```python
CAMERA_MODE = "single" | "all"
SELECTED_CAMERA = 1-4  # For single mode
```

**Implementation**:
- `initialize_cameras()`: Dynamic camera initialization
- `capture_frame()`: Mode-aware frame capture
- `release_cameras()`: Proper resource cleanup

### 2. Modular Frame Processing

**Pattern**: Flexible frame handling based on camera configuration
- **Single Mode**: Direct frame processing from selected camera
- **All Mode**: Automatic frame stitching into 2x2 grid (1920x1080)

**Benefits**:
- Clean separation of concerns
- Easy mode switching
- Consistent processing pipeline regardless of input configuration

### 3. Optimized Pose Estimation Pipeline

**Pattern**: Batch processing for GPU efficiency (inherited from volleyball project)
- Collect all detected bounding boxes per frame
- Process entire batch in single RTMPose inference call
- Significantly reduces GPU overhead

**Performance Impact**:
- Pose estimation: ~1.9ms average (vs ~11ms in volleyball project)
- Maintains accuracy while improving speed

### 4. Comprehensive Performance Profiling

**Pattern**: Multi-level timing analysis
- **Component-level**: Overall pipeline timing (capture, detection, tracking, etc.)
- **Internal-level**: Detailed model timing (preprocess, inference, postprocess)
- **Statistical Analysis**: Min/max/average/median calculations

**Implementation**:
- Real-time display overlays
- CSV logging for detailed analysis
- Frame-by-frame profiling data

### 5. Specialized Biomechanical Analysis Pattern

**Pattern**: Application-specific analysis modules built on core pipeline
- **Base Pipeline**: Standard detection → tracking → pose estimation
- **Analysis Layer**: Specialized calculations and visualizations
- **Dual Functionality**: Multiple analysis types in single application

**Example Implementation (`lab_mocap_2Dsquat.py`)**:
```python
# Knee angle calculation
def calculate_knee_flexion_angle(hip, knee, ankle):
    # Vector-based angle calculation
    # Custom formula: 180° - arccos(dot_product)
    
# Squat repetition counting
# State machine: STANDING → SQUAT_VALIDATED → REP_COMPLETED
```

### 6. Custom Visualization System

**Pattern**: Selective and configurable skeleton rendering
```python
def draw_skeleton_custom(img, keypoints, scores, 
                        selected_keypoints=None,
                        selected_connections=None,
                        kpt_thr=0.5, radius=3, line_width=2,
                        keypoint_colors=None, connection_colors=None):
```

**Features**:
- **Selective Rendering**: Choose specific keypoints and connections
- **Configurable Appearance**: Custom colors, sizes, and line thickness
- **Performance Optimized**: Only renders necessary elements
- **Reusable**: Supports multiple analysis applications

### 7. State Machine Pattern for Movement Analysis

**Pattern**: Robust movement detection with validation
```python
# Squat detection state machine
consecutive_squat_frames = 0
in_squat_position = False
SQUAT_VALIDATION_FRAMES = 3  # Prevent false positives

# State transitions based on biomechanical criteria
if hip_below_knee:
    consecutive_squat_frames += 1
    if consecutive_squat_frames >= SQUAT_VALIDATION_FRAMES:
        in_squat_position = True
```

**Benefits**:
- **False Positive Prevention**: Multi-frame validation
- **Clear State Management**: Explicit state transitions
- **Extensible**: Easy to add new movement patterns

## Technical Implementation Patterns

### 1. RTSP Stream Handling

**Pattern**: Robust stream management with error recovery
```python
def capture_frame(cameras):
    for cam_id, cap in cameras.items():
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read from camera {cam_id}")
            return None
```

**Features**:
- Individual camera failure handling
- Graceful degradation
- Resource cleanup on exit

### 2. Frame Stitching Algorithm

**Pattern**: Consistent multi-camera layout
```python
def stitch_frames(frame_1, frame_2, frame_3, frame_4):
    # Resize to 960x540 each
    # Arrange in 2x2 grid: [1|2]
    #                      [3|4]
    # Total: 1920x1080
```

### 3. Model Integration Pattern

**Pattern**: Consistent model initialization and usage
- Path management through `paths.py`
- ONNX Runtime backend with CUDA acceleration
- Standardized input/output handling

**Models**:
- **RTMDet-m**: 640x640 input, human detection
- **RTMPose-m**: 192x256 input, pose estimation
- **ByteTrack**: Multi-object tracking with ID persistence

### 4. Data Flow Pattern

**Pattern**: Consistent data structures throughout pipeline
```python
# Detection output: bboxes, scores
# Tracking input: [x1, y1, x2, y2, score, class_id]
# Pose input: list of bboxes
# Pose output: keypoints, scores
```

## Performance Optimization Patterns

### 1. Batch Processing
- **Where**: Pose estimation stage
- **How**: Process all detected persons in single inference call
- **Benefit**: ~5x speed improvement over individual processing

### 2. Timing Measurement Strategy
- **Granular Timing**: Component-level and internal model timing
- **Statistical Analysis**: Skip first frame for stable statistics
- **Real-time Display**: Performance metrics overlaid on video

### 3. Memory Management
- **Resource Cleanup**: Proper camera and file handle management
- **Error Handling**: Graceful failure recovery
- **Optional Logging**: HDF5 logging disabled by default for performance

## Integration Patterns

### 1. Laboratory Environment Adaptation
- **Camera Configuration**: 4 RTSP cameras positioned around lab space
- **Processing Modes**: Single camera for focused analysis, multi-camera for comprehensive coverage
- **Real-time Feedback**: Live display with performance metrics

### 2. Data Output Patterns
- **Visual Output**: Real-time display with pose overlays and timing information
- **Data Logging**: Optional HDF5 format for offline analysis
- **Performance Logging**: CSV files with detailed timing data

### 3. Future Extension Points
- **Joint Angle Calculation**: Keypoint data ready for biomechanical analysis
- **Advanced Analytics**: Framework supports additional processing stages
- **Integration Hooks**: Designed for laboratory workflow integration

## Critical Implementation Details

### 1. Modified RTMlib Components
- **Batch Processing**: RTMPose modified for batch inference
- **Timing Integration**: Internal timing hooks for performance analysis
- **GPU Optimization**: CUDA acceleration with ONNX Runtime

### 2. ByteTrack Integration
- **Input Format**: `[x1, y1, x2, y2, score, class_id]`
- **Tracking Parameters**: Optimized for laboratory environment
- **ID Persistence**: Maintains consistent subject tracking

### 3. Error Handling Strategies
- **Stream Failures**: Continue processing with available cameras
- **Model Errors**: Graceful degradation with error reporting
- **Resource Management**: Proper cleanup in all exit scenarios

This architecture provides a robust, high-performance foundation for real-time biomechanical analysis while maintaining flexibility for future enhancements.
