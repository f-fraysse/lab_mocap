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

### 8. Signed Angle Calculation Pattern

**Pattern**: Cross product-based directional angle determination
```python
def calculate_angle(point1, point2, point3, signed=False):
    # Calculate vectors from vertex
    vec1 = p1 - p2
    vec2 = p3 - p2
    
    # Standard angle calculation
    angle_deg = 180 - np.degrees(np.arccos(dot_product))
    
    # If signed, use cross product for direction
    if signed:
        cross_product = vec1[0] * vec2[1] - vec1[1] * vec2[0]
        if cross_product < 0:
            angle_deg = -angle_deg
```

**Application**:
- **Hip Angle**: Positive = extension, Negative = flexion (sign reversed)
- **Knee Angle**: Positive = flexion, Negative = extension
- Enables biomechanically meaningful angle representations

### 9. Gait Analysis Pipeline Pattern

**Pattern**: Multi-stage post-processing for stride analysis
```python
# 1. Signal filtering
ankle_x_filtered = butterworth_filter(ankle_x, cutoff=10, fs=30, order=1)

# 2. Footstrike detection
footstrike_indices = detect_footstrikes(ankle_x_filtered, 
                                       prominence=5, 
                                       min_spacing=0.25)

# 3. Stride segmentation
strides = segment_strides(data, footstrike_indices)

# 4. Time normalization
normalized_stride = normalize_stride(stride_data, num_points=101)
```

**Components**:
- **Butterworth Filtering**: Low-pass filter (10 Hz) for noise reduction
- **Peak Detection**: Local minima in ankle X position
- **Stride Segmentation**: Footstrike n to footstrike n+1
- **Cubic Interpolation**: Normalize to 0-100% (101 points)

### 10. Recording and Analysis Pattern

**Pattern**: Threaded post-processing with comprehensive output
```python
def on_recording_stopped(self):
    # Run analysis in separate thread
    analysis_thread = Thread(target=self.analyze_and_save_recording)
    analysis_thread.start()

def analyze_and_save_recording(self):
    # Save CSV, video, perform gait analysis
    # Generate 5 PNG graphs
    # All outputs to /output/{recording_name}/
```

**Output Structure**:
- CSV: Frame-by-frame data
- MP4: Video with pose overlays
- 5 PNG graphs: Ankle X, hip angles, knee angles, normalized strides

### 11. Fixed Y-Axis Graph Pattern

**Pattern**: Biomechanically meaningful axis limits
```python
class TreadmillAngleGraph:
    def __init__(self, title, y_min, y_max):
        self.y_min = y_min
        self.y_max = y_max
        self.ax.set_ylim(y_min, y_max)
```

**Limits**:
- Hip: -30° to +100° (extension to flexion range)
- Knee: -10° to +160° (extension to flexion range)
- Consistent across real-time and saved graphs

### 12. Statistical Visualization Pattern

**Pattern**: Mean with standard deviation shading
```python
# Calculate statistics
mean = np.mean(data, axis=0)
std = np.std(data, axis=0)

# Plot shaded region
ax.fill_between(x, mean - std, mean + std, 
                color='gray', alpha=0.3, label='±1 SD')

# Plot mean line on top
ax.plot(x, mean, 'k-', linewidth=3, label='Mean')
```

**Benefits**:
- Visualizes stride-to-stride variability
- Identifies consistent vs. variable gait phases
- Standard biomechanics visualization

### 13. Frame Inset Visualization Pattern

**Pattern**: Video frame integration with time-series data
```python
from matplotlib.gridspec import GridSpec

# Create figure with dedicated frame panel
fig = plt.figure(figsize=(14, 9))
gs = GridSpec(2, 1, height_ratios=[1, 3], hspace=0.3)

# Frame panel at top
ax_frames = fig.add_subplot(gs[0])
ax_frames.axis('off')

# Main plot below
ax = fig.add_subplot(gs[1])

# Add frame insets using fig.add_axes()
for i, (frame, x_pos, pct) in enumerate(zip(frames, x_positions, percentages)):
    ax_inset = fig.add_axes([x_pos, y_pos, width, height])
    ax_inset.imshow(frame)
    ax_inset.axis('off')
    
    # Add vertical reference line
    ax.axvline(x=pct, linestyle=':', color='black', linewidth=1.5)
```

**Application**:
- **Third Stride Selection**: Uses stride 3 (index 2) for representative visualization
- **Key Time Points**: 0%, 25%, 50%, 75%, 100% of stride cycle
- **Frame Extraction**: `frame_idx = int(percentage * (len(frames) - 1) / 100)`
- **Layout**: 25% top panel for frames, 75% bottom for graph

**Benefits**:
- Provides visual context for biomechanical data
- Links kinematic curves to body positions
- Facilitates interpretation of gait patterns
- Standard presentation format for gait analysis

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
