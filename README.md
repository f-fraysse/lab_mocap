# Lab MoCap

Real-time human pose estimation system for biomechanics laboratory environments using RTSP camera streams.

## Overview

Lab MoCap combines computer vision techniques (RTMDet detection, ByteTrack tracking, RTMPose estimation) to provide real-time human movement analysis without expensive motion capture equipment.

## Features

- **Real-time Processing**: ~26 FPS with live RTSP streams
- **Multi-camera Support**: Single camera or 4-camera stitched view
- **GPU Accelerated**: CUDA-optimized inference pipeline
- **Performance Monitoring**: Real-time metrics and detailed profiling
- **Optional Data Logging**: HDF5 format for offline analysis

## Quick Start

### Prerequisites

- CUDA-capable GPU
- Python 3.10
- 4 RTSP cameras (for multi-camera mode)

### Installation

```bash
# Clone repository
git clone https://github.com/f-fraysse/lab_mocap
cd lab_mocap

# Create conda environment
conda create -n lab_mocap python=3.10
conda activate lab_mocap

# Install dependencies
pip install -r requirements.txt

# Install ByteTrack
cd ByteTrack && pip install -e . && cd ..

# Install RTMlib
cd rtmlib && pip install -e . && cd ..
```

### Download Models

Detection and pose estimation models are not included in the repository. Download them manually:

1. Visit https://platform.openmmlab.com/deploee
2. Download the following ONNX models:
   - `rtmdet-m-640.onnx` (human detection)
   - `rtmpose-m-256-192.onnx` (pose estimation)
3. Save both models to the `/models/` subfolder

### Usage

1. **Configure cameras** in `scripts/lab_mocap_stream.py`:
   ```python
   CAMERA_MODE = "single"  # or "all"
   SELECTED_CAMERA = 1     # 1-4 for single mode
   ```

2. **Run the system**:
   ```bash
   python scripts/lab_mocap_stream.py
   ```

3. **Press 'q' to quit**

## Configuration

### Camera Modes

- **Single**: Process one camera (1920x1080)
- **All**: Process 4 cameras stitched into 2x2 grid (1920x1080 total)

### Data Logging

```python
record_results = True   # Enable HDF5 logging
```

## Performance

Current metrics with live RTSP streams:

| Component | Time (ms) | Notes |
|-----------|-----------|-------|
| **Total** | **38.0** | **~26 FPS** |
| Detection | 21.8 | RTMDet-m |
| Pose Estimation | 1.9 | RTMPose-m (batch) |
| Tracking | 0.1 | ByteTrack |
| Stream Capture | 12.4 | RTSP latency |

## Project Structure

```
lab_mocap/
├── scripts/
│   └── lab_mocap_stream.py    # Main processing script
├── models/                    # ONNX model files
├── rtmlib/                    # Modified RTMlib (batch processing)
├── ByteTrack/                 # Modified ByteTrack
├── memory-bank/               # Project documentation
└── profiling_logs/            # Performance data
```

## Future Development

- Joint angle calculation from pose keypoints
- Advanced biomechanical analysis
- Laboratory workflow integration

## Requirements

- RTMDet-m and RTMPose-m ONNX models
- CUDA toolkit and cuDNN
- Laboratory network with RTSP camera access

## License

See LICENSE file for details.
