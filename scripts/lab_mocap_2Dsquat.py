import h5py
import cv2
import os
import time
import numpy as np
import csv
import statistics
from datetime import datetime
from pathlib import Path
from argparse import Namespace
from rtmlib import RTMDet, RTMPose, draw_skeleton
from yolox.tracker.byte_tracker import BYTETracker
from paths import MODEL_DIR, DATA_DIR, OUTPUT_VIDEO_DIR, OUTPUT_H5_DIR, ensure_output_dirs

ensure_output_dirs()

# COCO17 keypoint indices for left leg
LEFT_HIP_IDX = 11
LEFT_KNEE_IDX = 13
LEFT_ANKLE_IDX = 15

def calculate_knee_flexion_angle(hip, knee, ankle):
    """
    Calculate knee flexion angle from hip, knee, and ankle keypoints.
    
    Args:
        hip: (x, y) coordinates of left hip
        knee: (x, y) coordinates of left knee  
        ankle: (x, y) coordinates of left ankle
        
    Returns:
        angle_deg: Knee flexion angle in degrees (0° = straight leg, 90° = bent at 90°)
    """
    # Convert to numpy arrays for vector operations
    hip = np.array(hip)
    knee = np.array(knee)
    ankle = np.array(ankle)
    
    # Calculate vectors from knee to hip and knee to ankle
    vec1 = hip - knee  # knee to hip
    vec2 = ankle - knee  # knee to ankle
    
    # Calculate angle using dot product
    cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    # Clamp to avoid numerical errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    # Calculate angle in radians then convert to degrees
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    angle_deg = round(180 - angle_deg)
    
    return angle_deg


def draw_skeleton_custom(img, keypoints, scores, 
                        selected_keypoints=None,
                        selected_connections=None,
                        kpt_thr=0.5,
                        radius=3,
                        line_width=2,
                        keypoint_colors=None,
                        connection_colors=None):
    """
    Custom skeleton drawing function with selective keypoint and connection display.
    
    Args:
        img: Input image
        keypoints: Array of keypoint coordinates [N, 17, 2] for COCO17
        scores: Array of keypoint confidence scores [N, 17]
        selected_keypoints: List of keypoint indices to draw (None = all)
        selected_connections: List of connection tuples (idx1, idx2) to draw
        kpt_thr: Confidence threshold for keypoints
        radius: Keypoint circle radius
        line_width: Connection line thickness
        keypoint_colors: Dict of {keypoint_idx: (B,G,R)} colors
        connection_colors: Dict of {(idx1,idx2): (B,G,R)} colors
        
    Returns:
        img: Image with drawn skeleton
    """
    # Default COCO17 keypoint colors (left side green, right side orange)
    default_kpt_colors = {
        0: [51, 153, 255],   # nose
        1: [51, 153, 255],   # left_eye
        2: [51, 153, 255],   # right_eye
        3: [51, 153, 255],   # left_ear
        4: [51, 153, 255],   # right_ear
        5: [0, 255, 0],      # left_shoulder
        6: [255, 128, 0],    # right_shoulder
        7: [0, 255, 0],      # left_elbow
        8: [255, 128, 0],    # right_elbow
        9: [0, 255, 0],      # left_wrist
        10: [255, 128, 0],   # right_wrist
        11: [0, 255, 0],     # left_hip
        12: [255, 128, 0],   # right_hip
        13: [0, 255, 0],     # left_knee
        14: [255, 128, 0],   # right_knee
        15: [0, 255, 0],     # left_ankle
        16: [255, 128, 0]    # right_ankle
    }
    
    # Default connection colors
    default_conn_colors = {
        (11, 13): [0, 255, 0],    # left_hip to left_knee
        (13, 15): [0, 255, 0],    # left_knee to left_ankle
    }
    
    # Use provided colors or defaults
    kpt_colors = keypoint_colors if keypoint_colors else default_kpt_colors
    conn_colors = connection_colors if connection_colors else default_conn_colors
    
    # If no specific keypoints selected, use all
    if selected_keypoints is None:
        selected_keypoints = list(range(17))
    
    # If no specific connections selected, use default left leg connections
    if selected_connections is None:
        selected_connections = [(11, 13), (13, 15)]  # hip-knee, knee-ankle
    
    num_instances = keypoints.shape[0]
    
    for i in range(num_instances):
        kpts = keypoints[i]
        kpt_scores = scores[i]
        
        # Draw connections first (so keypoints appear on top)
        for (idx1, idx2) in selected_connections:
            if (kpt_scores[idx1] > kpt_thr and kpt_scores[idx2] > kpt_thr and
                idx1 in selected_keypoints and idx2 in selected_keypoints):
                
                pt1 = (int(kpts[idx1][0]), int(kpts[idx1][1]))
                pt2 = (int(kpts[idx2][0]), int(kpts[idx2][1]))
                
                # Get connection color
                conn_key = (idx1, idx2) if (idx1, idx2) in conn_colors else (idx2, idx1)
                color = conn_colors.get(conn_key, [0, 255, 0])  # Default green
                
                cv2.line(img, pt1, pt2, color, line_width)
        
        # Draw keypoints
        for idx in selected_keypoints:
            if kpt_scores[idx] > kpt_thr:
                pt = (int(kpts[idx][0]), int(kpts[idx][1]))
                color = kpt_colors.get(idx, [0, 255, 0])  # Default green
                cv2.circle(img, pt, radius, color, -1)
    
    return img

#---------- CONFIGURATION ------------------
# Camera Configuration
CAMERA_MODE = "single"  # Options: "single", "all"
SELECTED_CAMERA = 4     # Which camera to use in single mode (1-4)

# RTSP URLs
CAMERA_URLS = {
    1: "rtsp://ubnt:ubnt@192.168.5.41:554/s0",
    2: "rtsp://ubnt:ubnt@192.168.5.45:554/s0", 
    3: "rtsp://ubnt:ubnt@192.168.5.42:554/s0",
    4: "rtsp://ubnt:ubnt@192.168.5.48:554/s0"
}

# Data Logging (off by default)
record_results = False
OUT_H5_FILE = "lab_mocap_2Dsquat_data.h5"

# Timing Logging Configuration
enable_timing_logs = False  # Set to False to disable CSV timing logs

# Detection and tracking models
RTMDET_MODEL = 'rtmdet-m-640.onnx'
RTMPOSE_MODEL = 'rtmpose-m-256-192.onnx'

# RTMPose engine
device = 'cuda'
backend = 'onnxruntime'
#---------- CONFIGURATION ------------------

def stitch_frames(frame_1, frame_2, frame_3, frame_4):
    """Stitch 4 camera frames into a 2x2 grid"""
    # Resize each frame to 960x540
    size = (960, 540)
    f1 = cv2.resize(frame_1, size)
    f2 = cv2.resize(frame_2, size)
    f3 = cv2.resize(frame_3, size)
    f4 = cv2.resize(frame_4, size)

    # Top row: f1 | f2
    top = np.hstack((f1, f2))
    # Bottom row: f3 | f4
    bottom = np.hstack((f3, f4))

    # Combine rows
    combined = np.vstack((top, bottom))
    return combined

def initialize_cameras():
    """Initialize camera captures based on CAMERA_MODE"""
    cameras = {}
    
    if CAMERA_MODE == "single":
        if SELECTED_CAMERA not in CAMERA_URLS:
            raise ValueError(f"Invalid camera selection: {SELECTED_CAMERA}. Must be 1-4.")
        cameras[SELECTED_CAMERA] = cv2.VideoCapture(CAMERA_URLS[SELECTED_CAMERA])
        print(f"Initialized single camera: {SELECTED_CAMERA}")
        
    elif CAMERA_MODE == "all":
        for cam_id, url in CAMERA_URLS.items():
            cameras[cam_id] = cv2.VideoCapture(url)
        print("Initialized all 4 cameras")
        
    else:
        raise ValueError(f"Invalid CAMERA_MODE: {CAMERA_MODE}. Must be 'single' or 'all'.")
    
    # Enable OpenCV optimizations
    cv2.setUseOptimized(True)
    return cameras

def capture_frame(cameras):
    """Capture frame(s) based on camera configuration"""
    frames = {}
    
    for cam_id, cap in cameras.items():
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read from camera {cam_id}")
            return None
        frames[cam_id] = frame
    
    if CAMERA_MODE == "single":
        return frames[SELECTED_CAMERA]
    elif CAMERA_MODE == "all":
        return stitch_frames(frames[1], frames[2], frames[3], frames[4])

def release_cameras(cameras):
    """Release all camera resources"""
    for cap in cameras.values():
        cap.release()

# Create profiling logs directory and initialize CSV file (if enabled)
log_file = None
if enable_timing_logs:
    log_dir = "profiling_logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"lab_mocap_stream_{timestamp}.csv")

    # Initialize CSV log file with headers
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'frame_id',
            'det_total', 'det_preprocess', 'det_prep', 'det_model', 'det_postprocess',
            'pose_total', 'pose_preprocess', 'pose_prep', 'pose_model', 'pose_postprocess', 'pose_num_bboxes',
            'cap_time_ms', 'det_time_ms', 'track_time_ms', 'pose_time_ms', 'hdf5_time_ms',
            'disp_time_ms', 'csv_time_ms', 'draw_time_ms', 'total_frame_time_ms'
        ])

# Timing statistics tracking
det_timing_stats = {
    'total': [], 'preprocess': [], 'prep': [], 'model': [], 'postprocess': []
}

pose_timing_stats = {
    'total': [], 'preprocess': [], 'prep': [], 'model': [], 'postprocess': []
}

# Overall timing lists
cap_times_ms = []
det_times_ms = []
track_times_ms = []
pose_times_ms = []
hdf_times_ms = []
disp_times_ms = []
csv_times_ms = []
draw_times_ms = []
total_frame_times_ms = []

# Make the full model paths
RTMDET_MODEL = os.path.join(MODEL_DIR, RTMDET_MODEL)
RTMPOSE_MODEL = os.path.join(MODEL_DIR, RTMPOSE_MODEL)
OUT_H5_FILE = os.path.join(OUTPUT_H5_DIR, OUT_H5_FILE)

# Create results HDF5 file if logging enabled
if record_results:
    h5file = h5py.File(OUT_H5_FILE, "w")
track_id_index = {}

# Initialize cameras
cameras = initialize_cameras()

# Get frame dimensions from first capture
test_frame = capture_frame(cameras)
if test_frame is None:
    print("Failed to capture initial frame. Exiting.")
    release_cameras(cameras)
    exit(1)

height, width = test_frame.shape[:2]
fps = 30  # Assume 30 FPS for RTSP streams

print(f"Frame dimensions: {width}x{height}")
print(f"Camera mode: {CAMERA_MODE}")

# Initialize detector
detector = RTMDet(
    onnx_model=RTMDET_MODEL,
    model_input_size=(640, 640),
    backend=backend,
    device=device
)

# Initialize ByteTrack tracker
args = Namespace(
    track_thresh=0.5,
    match_thresh=0.8,
    track_buffer=30,
    frame_rate=fps,
    mot20=False,
    min_hits=3
)
tracker = BYTETracker(args)

# Initialize pose detector
pose_estimator = RTMPose(
    onnx_model=RTMPOSE_MODEL,
    model_input_size=(192, 256),
    backend=backend,
    device=device
)

print("Initialized detection and pose estimation models")
print("Press 'q' to quit")

# ------------ START LOOP OVER FRAMES --------------
frame_id = 0
global_start = time.time()
csv_duration_ms = 0.0
draw_duration_ms = 0.0

# Squat tracking state variables
squat_rep_count = 0
consecutive_squat_frames = 0
in_squat_position = False
SQUAT_VALIDATION_FRAMES = 3  # Require 3 consecutive frames for squat validation

try:
    while True:
        start_time = time.perf_counter()
        
        # Capture frame
        frame = capture_frame(cameras)
        if frame is None:
            print("Failed to capture frame, continuing...")
            continue
            
        frame_id += 1
        cap_time = time.perf_counter()

        # Step 1: Detection
        det_bboxes_scores, det_timing = detector(frame)
        det_bboxes, det_scores = det_bboxes_scores
        det_time = time.perf_counter()

        # Update detection timing statistics
        for key in det_timing:
            if key in det_timing_stats:
                det_timing_stats[key].append(det_timing[key])

        # Step 2: Format for ByteTrack
        if len(det_bboxes) > 0:
            dets_for_tracker = np.array([[*box, score, 0] for box, score in zip(det_bboxes, det_scores)])
        else:
            dets_for_tracker = np.empty((0, 6))

        # Step 3: Tracking
        tracks = tracker.update(dets_for_tracker, [height, width], (height, width))
        track_time = time.perf_counter()

        # Step 4: Prepare data for Pose Estimation and Drawing
        img_show = frame.copy()
        track_ids = []
        tracked_bboxes = []
        bbox_scores = []
        bbox_rects = []

        for track in tracks:
            if not track.is_activated:
                continue

            x1, y1, w, h = track.tlwh
            x2, y2 = x1 + w, y1 + h
            track_id = int(track.track_id)
            score = track.score if hasattr(track, "score") else 0.0

            track_ids.append(track_id)
            tracked_bboxes.append([x1, y1, x2, y2])
            bbox_scores.append(score)
            bbox_rects.append((x1, y1, x2, y2, track_id, score))

        # Step 5: Pose estimation
        pose_timing = {
            'total': 0, 'preprocess': 0, 'prep': 0, 'model': 0, 'postprocess': 0, 'num_bboxes': 0
        }
        keypoints_list = []
        scores_list = []

        if len(tracked_bboxes) > 0:
            keypoints_list, scores_list, pose_timing = pose_estimator(frame, tracked_bboxes)
            for key in pose_timing:
                if key in pose_timing_stats and key != 'num_bboxes':
                    pose_timing_stats[key].append(pose_timing[key])

        pose_time = time.perf_counter()

        # Step 6: Build HDF5 file (if enabled)
        if record_results:
            track_ids_array = np.array(track_ids)
            bboxes_array = np.array(tracked_bboxes)
            bbox_scores_array = np.array(bbox_scores)
            keypoints_array = np.array(keypoints_list)
            keypoint_scores_array = np.array(scores_list)

            if track_ids_array.size > 0:
                frame_group = h5file.create_group(f"frame_{frame_id:05d}")
                frame_group.create_dataset("track_ids", data=track_ids_array)
                frame_group.create_dataset("bboxes", data=bboxes_array)
                frame_group.create_dataset("bbox_scores", data=bbox_scores_array)
                frame_group.create_dataset("keypoints", data=keypoints_array)
                frame_group.create_dataset("keypoint_scores", data=keypoint_scores_array)

                for tid in track_ids:
                    if tid not in track_id_index:
                        track_id_index[tid] = []
                    track_id_index[tid].append(frame_id)

        hdf_time = time.perf_counter()

        # Step 7: Drawing - Custom left leg visualization and squat tracking
        for i, (keypoints, kpt_scores) in enumerate(zip(keypoints_list, scores_list)):
            # Draw custom skeleton (left leg only)
            img_show = draw_skeleton_custom(
                img_show,
                np.array([keypoints]),
                np.array([kpt_scores]),
                selected_keypoints=[LEFT_HIP_IDX, LEFT_KNEE_IDX, LEFT_ANKLE_IDX],
                selected_connections=[(LEFT_HIP_IDX, LEFT_KNEE_IDX), (LEFT_KNEE_IDX, LEFT_ANKLE_IDX)],
                kpt_thr=0.3,
                radius=5,  # Larger keypoints for better visibility
                line_width=3  # Thicker lines for better visibility
            )
            
            # Calculate and display knee flexion angle + squat tracking
            hip_score = kpt_scores[LEFT_HIP_IDX]
            knee_score = kpt_scores[LEFT_KNEE_IDX]
            ankle_score = kpt_scores[LEFT_ANKLE_IDX]
            
            # Only calculate angle and track squat if all three keypoints are confident
            if hip_score > 0.5 and knee_score > 0.5 and ankle_score > 0.5:
                hip_pos = keypoints[LEFT_HIP_IDX]
                knee_pos = keypoints[LEFT_KNEE_IDX]
                ankle_pos = keypoints[LEFT_ANKLE_IDX]
                
                # Calculate knee flexion angle
                try:
                    knee_angle = calculate_knee_flexion_angle(hip_pos, knee_pos, ankle_pos)
                    
                    # Position angle text near the knee with offset
                    text_x = int(knee_pos[0] + 20)
                    text_y = int(knee_pos[1] - 10)
                    
                    # Ensure text stays within image bounds
                    text_x = max(10, min(text_x, width - 150))
                    text_y = max(30, min(text_y, height - 10))
                    
                    # Draw angle text with background for better visibility
                    angle_text = f"{knee_angle:.0f} deg"
                    
                    # Get text size for background rectangle
                    (text_width, text_height), baseline = cv2.getTextSize(
                        angle_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                    
                    # Draw background rectangle
                    cv2.rectangle(img_show, 
                                (text_x - 5, text_y - text_height - 5),
                                (text_x + text_width + 5, text_y + baseline + 5),
                                (0, 0, 0), -1)  # Black background
                    
                    # Draw angle text
                    cv2.putText(img_show, angle_text, (text_x, text_y),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)  # Yellow text
                    
                except Exception as e:
                    # Handle any calculation errors gracefully
                    print(f"Error calculating knee angle: {e}")
                
                # Squat tracking based on hip-knee vertical position comparison
                # Compare vertical positions (y-coordinates)
                # Hip lower than knee means hip_y > knee_y (since y=0 is at top)
                hip_below_knee = hip_pos[1] > knee_pos[1]
                
                if hip_below_knee:
                    # Hip is below knee - potential squat position
                    consecutive_squat_frames += 1
                    
                    if consecutive_squat_frames >= SQUAT_VALIDATION_FRAMES and not in_squat_position:
                        # Squat validated after 3 consecutive frames
                        in_squat_position = True
                        print(f"Squat validated at frame {frame_id}")
                        
                else:
                    # Hip is above knee - standing position
                    if in_squat_position:
                        # Coming up from squat - count repetition
                        squat_rep_count += 1
                        in_squat_position = False
                        print(f"Squat rep #{squat_rep_count} completed at frame {frame_id}")
                    
                    # Reset consecutive frame counter
                    consecutive_squat_frames = 0

        # Draw bboxes and ID labels
        for (x1, y1, x2, y2, track_id, score) in bbox_rects:
            img_show = cv2.rectangle(img_show, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            label = f"ID: {track_id}"
            if score is not None:
                label += f" | {score:.2f}"
            img_show = cv2.putText(img_show, label, (int(x1), int(y1) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        disp_time = time.perf_counter()

        # Calculate timing durations
        cap_duration = (cap_time - start_time) * 1000
        det_duration = (det_time - cap_time) * 1000
        track_duration = (track_time - det_time) * 1000
        pose_duration = (pose_time - track_time) * 1000
        hdf5_duration = (hdf_time - pose_time) * 1000
        disp_duration = (disp_time - hdf_time) * 1000

        # CSV Write Timing (if enabled)
        csv_write_start_time = time.perf_counter()
        total_frame_time = cap_duration + det_duration + track_duration + pose_duration + hdf5_duration + disp_duration + csv_duration_ms + draw_duration_ms
        
        if enable_timing_logs:
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    frame_id,
                    det_timing.get('total', 0), det_timing.get('preprocess', 0), det_timing.get('prep', 0), 
                    det_timing.get('model', 0), det_timing.get('postprocess', 0),
                    pose_timing.get('total', 0), pose_timing.get('preprocess', 0), pose_timing.get('prep', 0), 
                    pose_timing.get('model', 0), pose_timing.get('postprocess', 0), pose_timing.get('num_bboxes', 0),
                    cap_duration, det_duration, track_duration, pose_duration, hdf5_duration,
                    disp_duration, csv_duration_ms, draw_duration_ms, total_frame_time
                ])
        
        csv_time = time.perf_counter()
        current_csv_duration = (csv_time - csv_write_start_time) * 1000

        # Draw squat status indicators in top-right corner
        # Bright green rectangle when in validated squat position (twice as large)
        if in_squat_position:
            cv2.rectangle(img_show, (width - 280, 10), (width - 10, 110), (0, 255, 0), -1)  # Bright green filled rectangle (twice as large)
            cv2.putText(img_show, "SQUAT", (width - 250, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)  # Black text (larger)
        
        # Repetition counter display (50% larger text)
        rep_text = f"Reps: {squat_rep_count}"
        cv2.putText(img_show, rep_text, (width - 200, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)  # White text (50% larger)

        # Draw timing overlays
        img_show = cv2.putText(img_show, f'Biomechanics Lab - 2D Squat Rep Counter - FRANCOIS FRAYSSE @ UniSA', (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 127, 0), 2)
        img_show = cv2.putText(img_show, f'Camera {SELECTED_CAMERA} - Squat Repetition Tracking', (10, 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 127, 0), 2)
        img_show = cv2.putText(img_show, f'cap: {cap_duration:.1f} ms', (10, 80), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        img_show = cv2.putText(img_show, f'det: {det_duration:.1f} ms', (10, 100), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        img_show = cv2.putText(img_show, f'track: {track_duration:.1f} ms', (10, 120), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        img_show = cv2.putText(img_show, f'pose: {pose_duration:.1f} ms', (10, 140), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        img_show = cv2.putText(img_show, f'hdf5: {hdf5_duration:.1f} ms', (10, 160), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        img_show = cv2.putText(img_show, f'disp: {disp_duration:.1f} ms', (10, 180), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        img_show = cv2.putText(img_show, f'csv: {csv_duration_ms:.1f} ms', (10, 200), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        img_show = cv2.putText(img_show, f'draw: {draw_duration_ms:.1f} ms', (10, 220), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        fps_display = 1000 / total_frame_time if total_frame_time > 0 else 0
        img_show = cv2.putText(img_show, f'total: {total_frame_time:.1f} ms ({fps_display:.0f} FPS)', 
                              (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display frame
        cv2.imshow('Lab MoCap - 2D Squat Analysis', img_show)
        
        draw_time = time.perf_counter()
        current_draw_duration = (draw_time - csv_time) * 1000

        # Store timing statistics (skip first frame)
        if frame_id > 1:
            cap_times_ms.append(cap_duration)
            det_times_ms.append(det_duration)
            track_times_ms.append(track_duration)
            pose_times_ms.append(pose_duration)
            hdf_times_ms.append(hdf5_duration)
            disp_times_ms.append(disp_duration)
            csv_times_ms.append(current_csv_duration)
            draw_times_ms.append(current_draw_duration)
            total_frame_times_ms.append(total_frame_time)

        # Update durations for next frame
        csv_duration_ms = current_csv_duration
        draw_duration_ms = current_draw_duration

        # Check for quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

except KeyboardInterrupt:
    print("\nInterrupted by user")

finally:
    # Cleanup
    release_cameras(cameras)
    cv2.destroyAllWindows()

    # Save track presence info to HDF5
    if record_results:
        index_group = h5file.create_group("track_presence")
        for tid, frames in track_id_index.items():
            index_group.create_dataset(str(tid), data=np.array(frames, dtype='int32'))
        h5file.close()

    finish_time = time.time()
    print(f"\nTotal runtime: {(finish_time - global_start):.1f} seconds")

    # Helper function for statistics
    def calculate_stats(times_list):
        if not times_list:
            return {'min': 0, 'max': 0, 'avg': 0, 'median': 0}
        return {
            'min': min(times_list),
            'max': max(times_list),
            'avg': sum(times_list) / len(times_list),
            'median': statistics.median(times_list)
        }

    # Calculate and print statistics
    overall_stats = {
        "Total": calculate_stats(total_frame_times_ms),
        "Capture": calculate_stats(cap_times_ms),
        "Detection": calculate_stats(det_times_ms),
        "Track": calculate_stats(track_times_ms),
        "Pose": calculate_stats(pose_times_ms),
        "Hdf": calculate_stats(hdf_times_ms),
        "Disp": calculate_stats(disp_times_ms),
        "Csv": calculate_stats(csv_times_ms),
        "Draw": calculate_stats(draw_times_ms),
    }

    print("\n===== OVERALL TIMING STATISTICS =====")
    print(f"{'Component':<10}\t{'Min (ms)':<10}\t{'Max (ms)':<10}\t{'Avg (ms)':<10}\t{'Median (ms)':<10}")
    print("-" * 60)
    for name, stats in overall_stats.items():
        print(f"{name:<10}\t{stats['min']:<10.1f}\t{stats['max']:<10.1f}\t{stats['avg']:<10.1f}\t{stats['median']:<10.1f}")

    # Print detailed detection timing statistics
    print("\n===== DETECTION TIMING STATISTICS =====")
    print(f"{'Component':<12}\t{'Min (ms)':<10}\t{'Max (ms)':<10}\t{'Avg (ms)':<10}\t{'Median (ms)':<10}")
    print("-" * 65)
    for key in det_timing_stats:
        times = det_timing_stats[key][1:]  # Skip first frame
        if times:
            min_val = min(times)
            max_val = max(times)
            avg_val = sum(times)/len(times)
            median_val = statistics.median(times)
            print(f"{key:<12}\t{min_val:<10.1f}\t{max_val:<10.1f}\t{avg_val:<10.1f}\t{median_val:<10.1f}")

    # Print detailed pose timing statistics
    print("\n===== POSE ESTIMATION TIMING STATISTICS =====")
    print(f"{'Component':<12}\t{'Min (ms)':<10}\t{'Max (ms)':<10}\t{'Avg (ms)':<10}\t{'Median (ms)':<10}")
    print("-" * 65)
    for key in pose_timing_stats:
        times = pose_timing_stats[key][1:]  # Skip first frame
        if times:
            min_val = min(times)
            max_val = max(times)
            avg_val = sum(times)/len(times)
            median_val = statistics.median(times)
            print(f"{key:<12}\t{min_val:<10.1f}\t{max_val:<10.1f}\t{avg_val:<10.1f}\t{median_val:<10.1f}")

    if enable_timing_logs:
        print(f"\nDetailed profiling data saved to: {log_file}")
    else:
        print("\nTiming logging disabled - no CSV file saved")
