# Technical Context: Lab MoCap

## Technologies Used

### Core Libraries and Frameworks

1. **RTMDet & RTMPose**
   - Part of the OpenMMLab ecosystem
   - State-of-the-art models for object detection and pose estimation
   - Accessed through RTMlib Python package
   - Modified for batch processing optimization

2. **ByteTrack**
   - State-of-the-art multi-object tracking algorithm
   - Modified version included in the project repository
   - Optimized for laboratory environment tracking

3. **ONNX Runtime**
   - Cross-platform inference engine for ONNX models
   - Used with CUDA backend for GPU acceleration
   - Enables real-time processing performance

4. **OpenCV (cv2)**
   - Used for RTSP stream handling, image processing, and visualization
   - Handles real-time frame capture and display
   - Frame stitching for multi-camera configurations

5. **HDF5 (h5py)**
   - Hierarchical data format for storing structured numerical data
   - Used to store detection, tracking, and pose results (optional)

### Programming Languages

- **Python**: Primary development language
- **C++**: Used by underlying libraries (ONNX Runtime, OpenCV, CUDA)

### Model Formats

- **ONNX**: Open Neural Network Exchange format
  - Used for both RTMDet and RTMPose models
  - Enables cross-platform deployment and optimization

## Development Setup

### Environment

- **Conda Environment**: `lab_mocap`
- **Python Version**: 3.10
- **IDE**: Visual Studio Code

### Hardware

- **Target Hardware**: CUDA-capable GPU for real-time processing
- **Camera Setup**: 4 RTSP cameras positioned around laboratory space
- **Network**: Laboratory network supporting RTSP streaming

### Repository Structure

```
lab_mocap/
├── ByteTrack/           # Forked + modified ByteTrack repo (tracking)
├── rtmlib/              # Modified RTMlib with batch processing and profiling
├── models/              # Model files (.onnx) for RTMPose and RTMDet
├── data/                # Sample data (if any)
├── output/
│   ├── h5/              # HDF5 outputs: IDs, bboxes, keypoints, scores
│   └── video/           # Video outputs (if recording enabled)
├── profiling_logs/      # CSV logs of detailed timing data
├── scripts/             # Main processing scripts
│   └── lab_mocap_stream.py  # Primary real-time processing script
├── memory-bank/         # Project documentation and context
├── paths.py             # Project-relative path definitions
└── requirements.txt     # Python dependencies
```

### Key Dependencies

- **rtmlib**: Modified wrapper for RTMDet and RTMPose models
- **onnxruntime-gpu**: GPU-accelerated inference engine
- **h5py**: HDF5 file interface (for optional data logging)
- **opencv-python**: Computer vision operations and RTSP handling
- **numpy**: Numerical operations

## Technical Constraints

### Hardware Constraints

1. **GPU Requirements**
   - CUDA-capable GPU required for real-time performance
   - Sufficient VRAM for model inference (typically 4GB+)

2. **Network Requirements**
   - Stable network connection for RTSP streams
   - Sufficient bandwidth for 4 simultaneous camera streams

### Software Constraints

1. **CUDA Compatibility**
   - CUDA version must be compatible with GPU driver
   - ONNX Runtime must be compatible with CUDA version
   - cuDNN version must be compatible with CUDA

2. **ONNX Runtime Limitations**
   - Some operations may be assigned to CPU instead of GPU
   - Optimization ongoing for maximum GPU utilization

3. **RTMlib Modifications**
   - Local copy of RTMlib included in the repository
   - Modified for batch processing optimization
   - Modified to include detailed profiling capabilities
   - Installed in development mode for easy modification

### Performance Constraints

Current performance metrics (live RTSP streams):
- **Total Processing**: ~38ms/frame (~26 FPS)
- **Detection**: ~21.8ms/frame (primary component)
- **Pose Estimation**: ~1.9ms/frame (highly optimized)
- **Stream Capture**: ~12.4ms/frame (includes RTSP latency)

## Laboratory Setup

### Camera Configuration

1. **RTSP Cameras**: 4 cameras with network streaming capability
2. **Camera URLs**: Configured for laboratory network
3. **Positioning**: Strategic placement around laboratory space
4. **Resolution**: Typically 1920x1080 per camera

### Processing Modes

1. **Single Camera Mode**
   - Process one selected camera (1-4)
   - Full resolution processing
   - Optimal for focused analysis

2. **Multi-Camera Mode**
   - Process all 4 cameras simultaneously
   - Automatic frame stitching into 2x2 grid
   - Comprehensive coverage of laboratory space

## Dependencies and Installation

### Prerequisites

1. **CUDA Toolkit**
   - Compatible with GPU driver
   - Required for GPU acceleration

2. **cuDNN**
   - Compatible with CUDA version
   - Required for optimal performance

### Installation Steps

1. **Create and activate conda environment**
   ```bash
   conda create -n lab_mocap python=3.10
   conda activate lab_mocap
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install ByteTrack**
   ```bash
   cd ByteTrack
   pip install -e .
   cd ..
   ```

4. **Install RTMlib**
   ```bash
   cd rtmlib
   pip install -e .
   cd ..
   ```

## Development Workflow

1. **Configuration**
   - Edit camera configuration in `scripts/lab_mocap_stream.py`
   - Set camera mode (single/all) and selected camera

2. **Model Setup**
   - Ensure ONNX models are in `/models` directory
   - RTMDet-m and RTMPose-m models required

3. **Execution**
   - Run `python scripts/lab_mocap_stream.py`
   - Monitor real-time performance metrics

4. **Data Analysis**
   - Optional HDF5 data logging for offline analysis
   - Performance profiling data in CSV format

## Performance Profiling

Comprehensive profiling system implemented:

1. **Real-time Monitoring**
   - Performance metrics displayed on live video stream
   - Component-level timing (capture, detection, tracking, pose)
   - Overall FPS calculation

2. **Detailed Logging**
   - CSV logs with per-frame timing data
   - Internal model profiling (preprocess, inference, postprocess)
   - Statistical analysis (min/max/average/median)

3. **Optimization Tracking**
   - Batch processing implementation for pose estimation
   - GPU utilization monitoring
   - Bottleneck identification

## Future Technical Considerations

1. **Joint Angle Calculation**
   - Implementation of biomechanical analysis from pose keypoints
   - Integration with existing pose estimation pipeline

2. **Advanced Analytics**
   - Movement pattern analysis
   - Comparative studies support
   - Longitudinal tracking capabilities

3. **Laboratory Integration**
   - Integration with existing laboratory data systems
   - Export formats for biomechanical analysis tools
   - Automated data collection workflows

4. **Performance Optimization**
   - Further GPU optimization
   - Stream processing improvements
   - Multi-threading for camera handling

The system is designed for real-time biomechanical analysis with emphasis on performance, reliability, and research integration.
