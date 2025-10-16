"""
Simplified video processing thread for treadmill GUI
Only handles GoPro camera input
"""
from PyQt5.QtCore import QThread, pyqtSignal
import cv2
import numpy as np
import os
from argparse import Namespace
from rtmlib import RTMDet, RTMPose
from yolox.tracker.byte_tracker import BYTETracker
from paths import MODEL_DIR
from .test_gopro_stream import GoProCam


class TreadmillVideoThread(QThread):
    """Thread for processing video frames with pose estimation"""
    
    # Signals
    frame_ready = pyqtSignal(object, list, list, list)  # frame, track_data, keypoints, scores
    error_occurred = pyqtSignal(str)
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.running = False
        self.gopro_cam = None
        
        # Models
        self.detector = None
        self.tracker = None
        self.pose_estimator = None
        
        # Model paths
        self.rtmdet_model = os.path.join(MODEL_DIR, 'rtmdet-m-640.onnx')
        self.rtmpose_model = os.path.join(MODEL_DIR, 'rtmpose-m-256-192.onnx')
        
        # Recording state
        self.is_recording = False
        self.video_writer = None
        self.recording_data = {
            'hip_angles': [],
            'knee_angles': [],
            'ankle_coords': [],
            'frames': []
        }
        
    def initialize_models(self):
        """Initialize detection, tracking, and pose estimation models"""
        try:
            # Initialize detector
            self.detector = RTMDet(
                onnx_model=self.rtmdet_model,
                model_input_size=(640, 640),
                backend='onnxruntime',
                device='cuda'
            )
            
            # Initialize tracker
            args = Namespace(
                track_thresh=0.5,
                match_thresh=0.8,
                track_buffer=30,
                frame_rate=30,
                mot20=False,
                min_hits=3
            )
            self.tracker = BYTETracker(args)
            
            # Initialize pose estimator
            self.pose_estimator = RTMPose(
                onnx_model=self.rtmpose_model,
                model_input_size=(192, 256),
                backend='onnxruntime',
                device='cuda'
            )
            
            return True
        except Exception as e:
            self.error_occurred.emit(f"Failed to initialize models: {str(e)}")
            return False
            
    def initialize_camera(self):
        """Initialize GoPro camera"""
        try:
            # Initialize GoPro camera on index 0
            self.gopro_cam = GoProCam(
                index=0,
                size=(1920, 1080),
                fps=30,
                fourcc='MJPG',
                warmup_frames=20
            )
            self.gopro_cam.open()
            
            # Enable OpenCV optimizations
            cv2.setUseOptimized(True)
            
            return True
        except Exception as e:
            self.error_occurred.emit(f"Failed to initialize GoPro camera: {str(e)}")
            return False
            
    def release_camera(self):
        """Release camera resources"""
        if self.gopro_cam is not None:
            self.gopro_cam.close()
            self.gopro_cam = None
            
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        
    def start_recording(self, recording_name, output_path):
        """Start recording video and data"""
        self.is_recording = True
        self.recording_data = {
            'hip_angles': [],
            'knee_angles': [],
            'ankle_coords': [],
            'frames': []
        }
        
        # Initialize video writer
        video_path = os.path.join(output_path, f'{recording_name}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(video_path, fourcc, 30, (1920, 1080))
        
    def stop_recording(self):
        """Stop recording and return collected data"""
        self.is_recording = False
        
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        
        # Return copy of recording data
        data = self.recording_data.copy()
        self.recording_data = {
            'hip_angles': [],
            'knee_angles': [],
            'ankle_coords': [],
            'frames': []
        }
        return data
        
    def run(self):
        """Main processing loop"""
        # Initialize models
        if not self.initialize_models():
            return
            
        # Initialize camera
        if not self.initialize_camera():
            return
            
        self.running = True
        
        try:
            while self.running:
                # Capture frame from GoPro
                ret, frame = self.gopro_cam.read()
                if not ret:
                    continue
                
                height, width = frame.shape[:2]
                
                # Step 1: Detection
                det_bboxes_scores, _ = self.detector(frame)
                det_bboxes, det_scores = det_bboxes_scores
                
                # Step 2: Format for ByteTrack
                if len(det_bboxes) > 0:
                    dets_for_tracker = np.array([[*box, score, 0] 
                                                for box, score in zip(det_bboxes, det_scores)])
                else:
                    dets_for_tracker = np.empty((0, 6))
                
                # Step 3: Tracking
                tracks = self.tracker.update(dets_for_tracker, [height, width], (height, width))
                
                # Step 4: Prepare data for pose estimation
                track_data = []
                tracked_bboxes = []
                
                for track in tracks:
                    if not track.is_activated:
                        continue
                    
                    x1, y1, w, h = track.tlwh
                    x2, y2 = x1 + w, y1 + h
                    track_id = int(track.track_id)
                    score = track.score if hasattr(track, "score") else 0.0
                    
                    track_data.append({
                        'id': track_id,
                        'bbox': [x1, y1, x2, y2],
                        'score': score
                    })
                    tracked_bboxes.append([x1, y1, x2, y2])
                
                # Step 5: Pose estimation
                keypoints_list = []
                scores_list = []
                
                if len(tracked_bboxes) > 0:
                    keypoints_array, scores_array, _ = self.pose_estimator(frame, tracked_bboxes)
                    # Convert numpy arrays to lists for PyQt5 signal
                    keypoints_list = [kp.tolist() if isinstance(kp, np.ndarray) else kp for kp in keypoints_array]
                    scores_list = [sc.tolist() if isinstance(sc, np.ndarray) else sc for sc in scores_array]
                
                # Emit processed frame data
                self.frame_ready.emit(frame, track_data, keypoints_list, scores_list)
                
        except Exception as e:
            self.error_occurred.emit(f"Error in processing loop: {str(e)}")
        finally:
            self.release_camera()
            
    def stop(self):
        """Stop the processing thread"""
        self.running = False
        self.wait()
