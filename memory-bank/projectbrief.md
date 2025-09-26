# Lab MoCap Project Brief

## Project Overview

Lab MoCap is a computer vision project that combines object detection, multi-object tracking, and pose estimation to analyze human movement in an indoor biomechanics laboratory. The system processes live RTSP camera streams to automatically track subjects and analyze their movements for biomechanical research and analysis.

## Primary Goals

1. **Real-time Movement Analysis**: Develop a robust pipeline that processes live camera streams to track human subjects and extract pose data for biomechanical analysis.

2. **Multi-Camera Integration**: Support flexible camera configurations including single camera analysis and multi-camera (4-camera) stitched views for comprehensive movement capture.

3. **Subject Detection & Tracking**: Implement reliable detection and tracking of human subjects throughout recording sessions, maintaining consistent subject IDs.

4. **Pose Estimation**: Extract accurate skeletal pose data for each tracked subject to enable detailed biomechanical movement analysis.

5. **Performance Monitoring**: Maintain real-time processing capabilities with comprehensive performance profiling for optimization.

6. **Future: Joint Angle Analysis**: Develop capabilities to calculate lower limb joint angles from extracted keypoint positions for biomechanical assessment.

## Current Focus

The immediate focus is on establishing a stable real-time processing pipeline that can handle live RTSP camera streams while maintaining accurate detection, tracking, and pose estimation. The system currently achieves ~26 FPS processing speed with good accuracy.

## Technical Requirements

1. Process live RTSP camera streams from 4 laboratory cameras
2. Support both single camera and multi-camera (stitched) processing modes
3. Maintain consistent subject tracking IDs throughout sessions
4. Extract accurate pose keypoints for biomechanical analysis
5. Generate live display with visual overlays for real-time monitoring
6. Optional data logging (HDF5 format) for offline analysis
7. Comprehensive performance profiling and timing analysis

## Laboratory Setup

1. **Camera Configuration**: 4 RTSP cameras positioned around the laboratory space
2. **Target Hardware**: CUDA-capable GPU for real-time processing
3. **Processing Environment**: Indoor biomechanics laboratory with controlled lighting
4. **Subject Focus**: Human movement analysis for biomechanical research

## Success Criteria

1. Reliable subject detection and tracking in laboratory environment
2. Accurate pose estimation for tracked subjects
3. Real-time processing capability (~25+ FPS) for live analysis
4. Flexible camera configuration (single vs multi-camera modes)
5. Stable RTSP stream processing with minimal frame drops
6. Eventually: Accurate calculation of joint angles from pose data

## Current Status

- ✅ Functional real-time pipeline established
- ✅ RTSP stream integration completed
- ✅ Multi-camera support with frame stitching
- ✅ Performance optimization from volleyball project maintained
- ✅ Comprehensive timing and profiling system
- 🔄 Testing and validation in laboratory environment
- 📋 Future: Joint angle calculation implementation
